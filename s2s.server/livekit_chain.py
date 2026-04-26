"""LiveKit chained voice agent using local ASR/TTS and Gemini text generation."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
import re
import uuid

from livekit import rtc
from livekit.agents import (
    Agent,
    AgentSession,
    AutoSubscribe,
    ChatContext,
    DEFAULT_API_CONNECT_OPTIONS,
    JobContext,
    JobProcess,
    NOT_GIVEN,
    WorkerOptions,
    cli,
    llm,
    stt,
    tts,
)
from livekit.agents.llm.llm import ChatChunk, ChoiceDelta
from livekit.agents.stt.stt import SpeechData, SpeechEvent, SpeechEventType
from livekit.agents.tts.tts import ChunkedStream
from livekit.agents.utils.audio import combine_frames
from livekit.plugins import silero

from asr import ASREngine, WhisperEngine
from llm import DEFAULT_MODEL, INTRO_GREETING, SYSTEM_PROMPT, create_gemini_client
from tts import TTSEngine


def load_asr_engine() -> ASREngine | WhisperEngine:
    """Load the configured ASR backend using the same env contract as FastAPI."""
    asr_backend = os.getenv("ASR_BACKEND", "w2v").strip().lower()
    whisper_path_env = os.getenv("ASR_WHISPER_PATH")
    w2v_path_env = os.getenv("ASR_W2V_PATH")

    if asr_backend in {"whisper", "sna-whisper", "sna-whisper-asr"}:
        return (
            WhisperEngine(whisper_path=Path(whisper_path_env))
            if whisper_path_env
            else WhisperEngine()
        )

    return ASREngine(w2v_path=Path(w2v_path_env)) if w2v_path_env else ASREngine()


class LocalASRSTT(stt.STT):
    """LiveKit STT adapter around the existing offline ASR engine."""

    def __init__(self, engine: ASREngine | WhisperEngine) -> None:
        super().__init__(
            capabilities=stt.STTCapabilities(
                streaming=False,
                interim_results=False,
            )
        )
        self._engine = engine
        self._language = os.getenv("LIVEKIT_STT_LANGUAGE", "sn")

    @property
    def model(self) -> str:
        return type(self._engine).__name__

    @property
    def provider(self) -> str:
        return "local"

    async def _recognize_impl(
        self,
        buffer,
        *,
        language=None,
        conn_options,
    ) -> SpeechEvent:
        frame = combine_frames(buffer)
        wav_bytes = frame.to_wav_bytes()
        text = await asyncio.to_thread(self._engine.transcribe, wav_bytes)
        cleaned = text.strip()
        del language, conn_options

        alternatives = (
            [SpeechData(language=self._language, text=cleaned, confidence=1.0)]
            if cleaned
            else []
        )

        return SpeechEvent(
            type=SpeechEventType.FINAL_TRANSCRIPT,
            request_id=uuid.uuid4().hex,
            alternatives=alternatives,
        )

    async def aclose(self) -> None:
        return None


class GeminiTextLLMStream(llm.LLMStream):
    """One-shot Gemini text generation stream for LiveKit."""

    def __init__(
        self,
        *,
        llm_adapter: "GeminiTextLLM",
        chat_ctx: ChatContext,
        tools: list[llm.Tool],
        conn_options,
    ) -> None:
        super().__init__(
            llm=llm_adapter,
            chat_ctx=chat_ctx,
            tools=tools,
            conn_options=conn_options,
        )
        self._adapter = llm_adapter

    async def _run(self) -> None:
        contents: list[dict[str, object]] = []
        for message in self.chat_ctx.messages():
            text = message.text_content
            if not text:
                continue
            if message.role not in ("user", "assistant"):
                continue

            contents.append(
                {
                    "role": message.role,
                    "parts": [{"text": text}],
                }
            )

        if not contents:
            contents.append({"role": "user", "parts": [{"text": "Mhoro."}]})

        config = self._adapter._config()
        response = await asyncio.to_thread(
            self._adapter._client.models.generate_content,
            model=self._adapter.model,
            contents=contents,
            config=config,
        )

        reply = (response.text or "").strip()
        reply = re.sub(r"\d+", "", reply)
        reply = re.sub(r"\[[^\]]*\]", " ", reply)
        reply = re.sub(r"\s{2,}", " ", reply).strip()

        if not reply:
            reply = "Ndapota dzokorora zvakare zvishoma."

        self._event_ch.send_nowait(
            ChatChunk(
                id=uuid.uuid4().hex,
                delta=ChoiceDelta(role="assistant", content=reply),
            )
        )


class GeminiTextLLM(llm.LLM):
    """LiveKit LLM adapter using Gemini text generation."""

    def __init__(self, model: str | None = None) -> None:
        super().__init__()
        self._client = create_gemini_client()
        self._model = model or os.getenv("GEMINI_TEXT_MODEL", DEFAULT_MODEL)

    @property
    def model(self) -> str:
        return self._model

    @property
    def provider(self) -> str:
        return "google"

    def _config(self):
        from google.genai import types

        return types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            temperature=0.5,
            max_output_tokens=256,
            thinking_config=types.ThinkingConfig(thinking_level="MINIMAL"),
        )

    def chat(
        self,
        *,
        chat_ctx: ChatContext,
        tools: list[llm.Tool] | None = None,
        conn_options=DEFAULT_API_CONNECT_OPTIONS,
        parallel_tool_calls=NOT_GIVEN,
        tool_choice=NOT_GIVEN,
        extra_kwargs=NOT_GIVEN,
    ) -> llm.LLMStream:
        del parallel_tool_calls, tool_choice, extra_kwargs
        return GeminiTextLLMStream(
            llm_adapter=self,
            chat_ctx=chat_ctx,
            tools=tools or [],
            conn_options=conn_options,
        )

    async def aclose(self) -> None:
        return None


class LocalTTSChunkedStream(ChunkedStream):
    """Chunked non-streaming TTS adapter around the existing TTS engine."""

    def __init__(
        self,
        *,
        tts_adapter: "LocalTTS",
        input_text: str,
        conn_options,
    ) -> None:
        super().__init__(
            tts=tts_adapter,
            input_text=input_text,
            conn_options=conn_options,
        )
        self._adapter = tts_adapter

    async def _run(self, output_emitter) -> None:
        wav_bytes = await asyncio.to_thread(
            self._adapter._engine.synthesize,
            self.input_text,
        )

        output_emitter.initialize(
            request_id=uuid.uuid4().hex,
            sample_rate=self._adapter.sample_rate,
            num_channels=self._adapter.num_channels,
            mime_type="audio/wav",
            frame_size_ms=120,
            stream=False,
        )
        output_emitter.push(wav_bytes)


class LocalTTS(tts.TTS):
    """LiveKit TTS adapter around the local non-streaming TTS model."""

    def __init__(self, engine: TTSEngine) -> None:
        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=False, aligned_transcript=False),
            sample_rate=engine._model.config.sampling_rate,
            num_channels=1,
        )
        self._engine = engine

    @property
    def model(self) -> str:
        return "mms-tts-sna"

    @property
    def provider(self) -> str:
        return "local"

    def synthesize(self, text: str, *, conn_options=DEFAULT_API_CONNECT_OPTIONS) -> ChunkedStream:
        return LocalTTSChunkedStream(
            tts_adapter=self,
            input_text=text,
            conn_options=conn_options,
        )

    async def aclose(self) -> None:
        return None


def prewarm(proc: JobProcess) -> None:
    """Load heavy local models once per worker process."""
    proc.userdata["vad"] = silero.VAD.load(
        min_silence_duration=0.45,
        prefix_padding_duration=0.35,
        activation_threshold=0.55,
    )
    proc.userdata["asr_engine"] = load_asr_engine()
    proc.userdata["tts_engine"] = TTSEngine()
    proc.userdata["gemini_llm"] = GeminiTextLLM()


async def entrypoint(ctx: JobContext) -> None:
    """Run the chained LiveKit voice agent."""
    await ctx.connect(auto_subscribe=AutoSubscribe.SUBSCRIBE_ALL)
    shutdown_event = asyncio.Event()
    ctx.add_shutdown_callback(lambda *_: _notify_shutdown(shutdown_event))

    session = AgentSession(
        vad=ctx.proc.userdata["vad"],
        turn_handling={
            "endpointing": {
                "mode": "dynamic",
                "min_delay": float(os.getenv("LIVEKIT_MIN_ENDPOINTING_DELAY", "0.5")),
                "max_delay": float(os.getenv("LIVEKIT_MAX_ENDPOINTING_DELAY", "1.8")),
            },
            "interruption": {
                "enabled": True,
                "mode": "adaptive",
                "min_duration": float(os.getenv("LIVEKIT_INTERRUPT_MIN_DURATION", "0.4")),
                "resume_false_interruption": True,
            },
            "preemptive_generation": {
                "enabled": False,
            },
        },
    )

    agent = Agent(
        instructions=SYSTEM_PROMPT,
        stt=LocalASRSTT(ctx.proc.userdata["asr_engine"]),
        llm=ctx.proc.userdata["gemini_llm"],
        tts=LocalTTS(ctx.proc.userdata["tts_engine"]),
    )

    await session.start(agent=agent, room=ctx.room)
    await shutdown_event.wait()


async def _notify_shutdown(event: asyncio.Event) -> None:
    event.set()


if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            prewarm_fnc=prewarm,
            agent_name="sna-livekit-chain",
        )
    )
