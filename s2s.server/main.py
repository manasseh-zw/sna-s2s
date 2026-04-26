"""SNA S2S – FastAPI server exposing ASR and TTS endpoints."""

import asyncio
from base64 import b64encode
from contextlib import asynccontextmanager
import datetime as dt
from io import BytesIO
import os
from pathlib import Path
from typing import Any
import uuid

from fastapi import FastAPI, File, HTTPException, UploadFile, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from livekit import api as livekit_api
from pydantic import BaseModel
from asr import ASREngine, WhisperEngine
from live_s2s import run_live_s2s_session
from llm import LLMClient, get_llm_backend_label
from tts import TTSEngine


# ---------------------------------------------------------------------------
# App state – engines are loaded once at startup
# ---------------------------------------------------------------------------


class _AppState:
    asr: Any
    llm: LLMClient
    tts: TTSEngine


state = _AppState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    asr_backend = os.getenv("ASR_BACKEND", "w2v").strip().lower()

    whisper_path_env = os.getenv("ASR_WHISPER_PATH")
    w2v_path_env = os.getenv("ASR_W2V_PATH")

    if asr_backend in {"whisper", "sna-whisper", "sna-whisper-asr"}:
        print("Loading ASR engine (Whisper)…")
        state.asr = WhisperEngine(whisper_path=Path(whisper_path_env)) if whisper_path_env else WhisperEngine()
    else:
        print("Loading ASR engine (Wav2Vec2-BERT)…")
        state.asr = ASREngine(w2v_path=Path(w2v_path_env)) if w2v_path_env else ASREngine()

    print("ASR engine ready.")

    print("Loading TTS engine…")
    state.tts = TTSEngine()
    print("TTS engine ready.")

    print(f"Connecting to {get_llm_backend_label()}…")
    state.llm = LLMClient()
    print("LLM client ready.")

    yield  # Server is running

    print("Shutting down.")


app = FastAPI(title="SNA Speech API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
    expose_headers=["X-Transcript", "X-Reply"],
)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


class TTSRequest(BaseModel):
    text: str


class LiveKitSessionRequest(BaseModel):
    room_name: str | None = None


class LiveKitSessionResponse(BaseModel):
    token: str
    url: str
    room_name: str
    participant_identity: str


def _create_livekit_session(room_name: str | None = None) -> LiveKitSessionResponse:
    livekit_url = os.getenv("LIVEKIT_URL", "ws://127.0.0.1:7880").strip()
    agent_name = os.getenv("LIVEKIT_AGENT_NAME", "sna-livekit-chain").strip()

    room_name = room_name or f"sna-live-{uuid.uuid4().hex[:8]}"
    participant_identity = f"web-{uuid.uuid4().hex[:8]}"

    room_config = livekit_api.RoomConfiguration(name=room_name)
    room_config.agents.append(
        livekit_api.RoomAgentDispatch(
            agent_name=agent_name,
            metadata='{"pipeline":"chain","language":"sn"}',
        )
    )

    token = (
        livekit_api.AccessToken()
        .with_identity(participant_identity)
        .with_name("Web Voice User")
        .with_grants(
            livekit_api.VideoGrants(
                room_join=True,
                room=room_name,
                can_publish=True,
                can_subscribe=True,
                can_publish_data=True,
            )
        )
        .with_room_config(room_config)
        .with_ttl(dt.timedelta(hours=1))
        .to_jwt()
    )

    return LiveKitSessionResponse(
        token=token,
        url=livekit_url,
        room_name=room_name,
        participant_identity=participant_identity,
    )


@app.post("/asr")
async def asr_endpoint(file: UploadFile = File(...)):
    """Transcribe with the configured ASR backend (Wav2Vec2-BERT or Whisper)."""
    audio_bytes = await file.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="Empty audio file.")

    try:
        text = await asyncio.to_thread(state.asr.transcribe, audio_bytes)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return {"text": text}


@app.post("/tts")
async def tts_endpoint(body: TTSRequest):
    """Synthesize Shona text and stream back a WAV audio file."""
    if not body.text.strip():
        raise HTTPException(status_code=400, detail="Text must not be empty.")

    try:
        wav_bytes = await asyncio.to_thread(state.tts.synthesize, body.text)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return StreamingResponse(
        BytesIO(wav_bytes),
        media_type="audio/wav",
        headers={"Content-Disposition": "inline; filename=tts_output.wav"},
    )


@app.post("/s2s")
async def s2s_endpoint(file: UploadFile = File(...)):
    """Full speech-to-speech: ASR → LLM → TTS in one request."""
    audio_bytes = await file.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="Empty audio file.")

    try:
        transcript = await asyncio.to_thread(state.asr.transcribe, audio_bytes)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"ASR failed: {exc}") from exc

    if not transcript:
        raise HTTPException(status_code=422, detail="No speech detected.")

    try:
        reply = await asyncio.to_thread(state.llm.respond, transcript)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"LLM failed: {exc}") from exc

    try:
        wav_bytes = await asyncio.to_thread(state.tts.synthesize, reply)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"TTS failed: {exc}") from exc

    # Return JSON instead of custom headers to avoid header encoding/size issues.
    # (LLM output may contain characters that are not valid for HTTP header values.)
    wav_base64 = b64encode(wav_bytes).decode("ascii")
    return JSONResponse(
        {
            "wav_base64": wav_base64,
            "transcript": transcript,
            "reply": reply,
        }
    )


@app.post("/s2s/reset")
async def s2s_reset_endpoint():
    """Clear the LLM conversation context to start a fresh session."""
    state.llm.reset_context()
    return {"status": "ok"}


@app.post("/livekit/token")
async def livekit_token_endpoint(body: LiveKitSessionRequest | None = None):
    """Mint a room token for the browser and dispatch the LiveKit agent."""
    try:
        session = _create_livekit_session(body.room_name if body else None)
    except ValueError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return session.model_dump()


@app.websocket("/s2s/live")
async def s2s_live_websocket(websocket: WebSocket):
    """Realtime speech-to-speech over Gemini Live with local TTS playback."""
    await run_live_s2s_session(websocket, state.tts)
