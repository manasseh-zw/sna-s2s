"""SNA S2S – FastAPI server exposing ASR and TTS endpoints."""

from contextlib import asynccontextmanager
from io import BytesIO

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from asr import ASREngine
from tts import TTSEngine


# ---------------------------------------------------------------------------
# App state – engines are loaded once at startup
# ---------------------------------------------------------------------------

class _AppState:
    asr: ASREngine
    tts: TTSEngine


state = _AppState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Loading ASR engines (Whisper + Wav2Vec2-BERT)…")
    state.asr = ASREngine()
    print("ASR engines ready.")

    print("Loading TTS engine…")
    state.tts = TTSEngine()
    print("TTS engine ready.")

    yield  # Server is running

    print("Shutting down.")


app = FastAPI(title="SNA Speech API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["POST"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

class TTSRequest(BaseModel):
    text: str


@app.post("/asr")
async def asr_endpoint(file: UploadFile = File(...)):
    """Transcribe with Wav2Vec2-BERT CTC (fast, Shona fine-tuned)."""
    audio_bytes = await file.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="Empty audio file.")

    try:
        text = state.asr.transcribe_w2v(audio_bytes)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return {"text": text}


@app.post("/tts")
async def tts_endpoint(body: TTSRequest):
    """Synthesize Shona text and stream back a WAV audio file."""
    if not body.text.strip():
        raise HTTPException(status_code=400, detail="Text must not be empty.")

    try:
        wav_bytes = state.tts.synthesize(body.text)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return StreamingResponse(
        BytesIO(wav_bytes),
        media_type="audio/wav",
        headers={"Content-Disposition": "inline; filename=tts_output.wav"},
    )
