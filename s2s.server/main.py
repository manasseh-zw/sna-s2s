"""SNA S2S – FastAPI server exposing ASR and TTS endpoints."""

import asyncio
from base64 import b64encode
from contextlib import asynccontextmanager
from io import BytesIO
import os
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from asr import ASREngine, WhisperEngine
from llm import LLMClient
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
    asr_backend = os.getenv("ASR_BACKEND", "whisper").strip().lower()

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

    print("Connecting to LM Studio (tiny-aya-earth)…")
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
