"""Configurable LLM clients for S2S assistant responses."""

import os
import re

from dotenv import load_dotenv
from google import genai
from google.genai import types
from openai import OpenAI

DEFAULT_MODEL = "gemini-3.1-flash-lite-preview"
DEFAULT_LIVE_MODEL = "gemini-3.1-flash-live-preview"
DEFAULT_LOCAL_MODEL = "tiny-aya-earth"
DEFAULT_LOCAL_BASE_URL = "http://127.0.0.1:1234/v1"

SYSTEM_PROMPT = (
    "You are a helpful assistant that must reply in very simple, modern Shona only. "
    "Use everyday Shona that sounds natural and easy to understand. "
    "Do not use deep, old-fashioned, literary, or overly formal Shona. "
    "The user's main language is Shona, so interpret their speech as Shona first whenever reasonable. "
    "Avoid English words, loanwords, and other foreign words whenever a simple Shona phrasing is possible. "
    "Keep names of people only when necessary, but otherwise prefer plain Shona wording. "
    "Avoid digits such as 1, 2, 3, or 2026. If a quantity must be mentioned, prefer simple Shona words, or avoid the number entirely if it is not important. "
    "Avoid English names for technical terms, objects, and places when a simpler Shona explanation is possible. "
    "Do not respond in English, Spanish, Portuguese, Japanese, or any other language unless the user explicitly requires it. "
    "Do not include timestamps, subtitle markers, bracketed timings, or text like [0m10s]. "
    "Ignore background speech, TV or radio audio, and any echo or playback of your own voice. "
    "If the transcript has mistakes, infer the intended Shona carefully. "
    "If the speech is truly unclear, ask the user to repeat. "
    "Keep answers short, complete, and easy for a Shona TTS model to pronounce clearly."
)
INTRO_GREETING = "Mhoro shamwari. Unogona kutaura zvino, uye ndichakupindura nechiShona chakareruka."


def load_llm_env() -> None:
    """Load .env once before reading LLM-related configuration."""
    load_dotenv()


def create_gemini_client() -> genai.Client:
    """Create an authenticated Gemini client from environment variables."""
    load_llm_env()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not set.")

    return genai.Client(
        api_key=api_key,
        http_options={"api_version": "v1beta"},
    )


def normalize_live_model(model: str) -> str:
    """Use the explicit Live model path form expected by the SDK examples."""
    if model.startswith("models/"):
        return model
    return f"models/{model}"


def get_llm_backend() -> str:
    """Return the configured LLM backend name."""
    load_llm_env()
    return os.getenv("LLM_BACKEND", "gemini").strip().lower()


def get_text_model_name() -> str:
    """Return the configured text model for the active backend."""
    backend = get_llm_backend()
    if backend == "local":
        return os.getenv("LOCAL_LLM_MODEL", DEFAULT_LOCAL_MODEL).strip()
    return os.getenv("GEMINI_TEXT_MODEL", DEFAULT_MODEL).strip()


def get_llm_backend_label() -> str:
    """Human-friendly label for startup logs."""
    backend = get_llm_backend()
    model = get_text_model_name()

    if backend == "local":
        base_url = os.getenv("LOCAL_LLM_BASE_URL", DEFAULT_LOCAL_BASE_URL).strip()
        return f"local OpenAI-compatible backend ({model} @ {base_url})"

    return f"Gemini ({model})"


def create_live_connect_config() -> types.LiveConnectConfig:
    """Build the Live API config used by the realtime S2S bridge."""
    return types.LiveConnectConfig(
        response_modalities=[types.Modality.AUDIO],
        system_instruction=types.Content(
            parts=[types.Part(text=SYSTEM_PROMPT)]
        ),
        temperature=0.6,
        max_output_tokens=256,
        input_audio_transcription=types.AudioTranscriptionConfig(),
        output_audio_transcription=types.AudioTranscriptionConfig(),
        realtime_input_config=types.RealtimeInputConfig(
            automatic_activity_detection=types.AutomaticActivityDetection(
                disabled=False,
                start_of_speech_sensitivity=types.StartSensitivity.START_SENSITIVITY_HIGH,
                end_of_speech_sensitivity=types.EndSensitivity.END_SENSITIVITY_LOW,
                prefix_padding_ms=220,
                silence_duration_ms=700,
            ),
            activity_handling=types.ActivityHandling.START_OF_ACTIVITY_INTERRUPTS,
            turn_coverage=types.TurnCoverage.TURN_INCLUDES_ONLY_ACTIVITY,
        ),
    )


class LLMClient:
    """Calls the configured text model and keeps lightweight conversation history."""

    def __init__(self, model: str | None = None) -> None:
        self._backend = get_llm_backend()
        self._model = model or get_text_model_name()
        self._history: list[types.Content] = []
        self._local_history: list[dict[str, str]] = []

        if self._backend == "local":
            self._client = self._create_local_client()
        else:
            self._client = create_gemini_client()

    def _create_local_client(self) -> OpenAI:
        base_url = os.getenv("LOCAL_LLM_BASE_URL", DEFAULT_LOCAL_BASE_URL).strip()
        api_key = os.getenv("LOCAL_LLM_API_KEY", "not-needed").strip() or "not-needed"
        return OpenAI(base_url=base_url, api_key=api_key)

    def respond(self, text: str, *, maintain_context: bool = True) -> str:
        """Send a Shona utterance and return a short Shona reply."""
        if self._backend == "local":
            return self._respond_local(text, maintain_context=maintain_context)

        user_content = types.Content(
            role="user",
            parts=[types.Part.from_text(text=text)],
        )

        contents: list[types.Content] = []
        if maintain_context:
            contents.extend(self._history)
        contents.append(user_content)

        config = types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            temperature=0.6,
            max_output_tokens=256,
        )

        response = self._client.models.generate_content(
            model=self._model,
            contents=contents,
            config=config,
        )

        reply = (response.text or "").strip()
        if not reply:
            raise RuntimeError("No text in Gemini response.")

        # Safety cleanup: never return digits.
        reply = re.sub(r"\d+", "", reply)
        reply = re.sub(r"\s{2,}", " ", reply).strip()

        if maintain_context:
            self._history.append(user_content)
            self._history.append(
                types.Content(
                    role="model",
                    parts=[types.Part.from_text(text=reply)],
                )
            )

        return reply

    def _respond_local(self, text: str, *, maintain_context: bool = True) -> str:
        messages: list[dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]

        if maintain_context:
            messages.extend(self._local_history)
        messages.append({"role": "user", "content": text})

        response = self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            temperature=0.6,
            max_tokens=256,
        )

        reply = response.choices[0].message.content or ""
        reply = reply.strip()
        if not reply:
            raise RuntimeError("No text in local LLM response.")

        reply = re.sub(r"\d+", "", reply)
        reply = re.sub(r"\s{2,}", " ", reply).strip()

        if maintain_context:
            self._local_history.append({"role": "user", "content": text})
            self._local_history.append({"role": "assistant", "content": reply})

        return reply

    def reset_context(self) -> None:
        """Clear conversation history (start a fresh S2S session)."""
        self._history = []
        self._local_history = []
