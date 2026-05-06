"""Speech backend loaders and remote HTTP adapters."""

from __future__ import annotations

from pathlib import Path
import os
from typing import Any

import httpx

from asr import ASREngine, WhisperEngine
from tts import TTSEngine


DEFAULT_TIMEOUT_S = float(os.getenv("SNA_SPEECH_TIMEOUT_S", "60"))
DEFAULT_TTS_SAMPLE_RATE = 24_000
REMOTE_TTS_VOICE = "Manasseh"
LOCAL_TTS_VOICE = "Tatenda"


def _speech_backend() -> str:
    return os.getenv("SNA_SPEECH_BACKEND", "local").strip().lower()


def _remote_base_url() -> str:
    return os.getenv("SNA_SPEECH_BASE_URL", "").strip().rstrip("/")


def _remote_endpoint(path: str, explicit_env: str) -> str:
    explicit = os.getenv(explicit_env, "").strip()
    if explicit:
        return explicit
    base_url = _remote_base_url()
    if not base_url:
        raise ValueError(
            f"{explicit_env} is not set and SNA_SPEECH_BASE_URL is empty. "
            "Set SNA_SPEECH_BASE_URL or explicit remote endpoint URLs."
        )
    return f"{base_url}{path}"


def using_remote_speech() -> bool:
    return _speech_backend() in {"remote", "modal", "http"}


def remote_tts_enabled() -> bool:
    explicit = os.getenv("SNA_SPEECH_TTS_URL", "").strip()
    return bool(explicit or using_remote_speech())


def _warm_on_startup_enabled() -> bool:
    return os.getenv("SNA_SPEECH_WARM_ON_STARTUP", "true").strip().lower() not in {
        "0",
        "false",
        "no",
    }


def remote_health_url() -> str:
    explicit = os.getenv("SNA_SPEECH_HEALTH_URL", "").strip()
    if explicit:
        return explicit
    return _remote_endpoint("/healthz", "SNA_SPEECH_HEALTH_URL")


def warm_remote_speech_service() -> None:
    """Ping the remote speech service so Modal spins up the warm container."""
    if not remote_tts_enabled() or not _warm_on_startup_enabled():
        return

    response = httpx.get(remote_health_url(), timeout=DEFAULT_TIMEOUT_S)
    response.raise_for_status()


class RemoteASREngine:
    """Drop-in ASR engine that calls the deployed Modal HTTP endpoint."""

    def __init__(self, url: str, timeout_s: float = DEFAULT_TIMEOUT_S) -> None:
        self._url = url
        self._timeout_s = timeout_s

    def transcribe(self, audio_bytes: bytes) -> str:
        response = httpx.post(
            self._url,
            files={"file": ("audio.wav", audio_bytes, "audio/wav")},
            timeout=self._timeout_s,
        )
        response.raise_for_status()
        payload = response.json()
        return str(payload.get("text", "")).strip()


class RemoteTTSEngine:
    """Drop-in TTS engine that calls the deployed Modal HTTP endpoint."""

    def __init__(
        self,
        url: str,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        sample_rate: int = DEFAULT_TTS_SAMPLE_RATE,
    ) -> None:
        self._url = url
        self._timeout_s = timeout_s
        self.sample_rate = sample_rate

    def synthesize(self, text: str) -> bytes:
        response = httpx.post(
            self._url,
            json={"text": text},
            timeout=self._timeout_s,
        )
        response.raise_for_status()
        return response.content


def load_asr_engine() -> ASREngine | WhisperEngine | RemoteASREngine:
    if using_remote_speech():
        return RemoteASREngine(
            url=_remote_endpoint("/asr", "SNA_SPEECH_ASR_URL"),
            timeout_s=DEFAULT_TIMEOUT_S,
        )

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


def load_tts_engine() -> TTSEngine | RemoteTTSEngine:
    if using_remote_speech():
        return RemoteTTSEngine(
            url=_remote_endpoint("/tts", "SNA_SPEECH_TTS_URL"),
            timeout_s=DEFAULT_TIMEOUT_S,
            sample_rate=int(os.getenv("SNA_SPEECH_TTS_SAMPLE_RATE", str(DEFAULT_TTS_SAMPLE_RATE))),
        )
    return TTSEngine()


def load_remote_tts_engine() -> RemoteTTSEngine | None:
    if not remote_tts_enabled():
        return None

    return RemoteTTSEngine(
        url=_remote_endpoint("/tts", "SNA_SPEECH_TTS_URL"),
        timeout_s=DEFAULT_TIMEOUT_S,
        sample_rate=int(os.getenv("SNA_SPEECH_TTS_SAMPLE_RATE", str(DEFAULT_TTS_SAMPLE_RATE))),
    )


def resolve_tts_sample_rate(engine: Any) -> int:
    sample_rate = getattr(engine, "sample_rate", None)
    if isinstance(sample_rate, int) and sample_rate > 0:
        return sample_rate

    model = getattr(engine, "_model", None)
    config = getattr(model, "config", None)
    config_sample_rate = getattr(config, "sampling_rate", None)
    if isinstance(config_sample_rate, int) and config_sample_rate > 0:
        return config_sample_rate

    return DEFAULT_TTS_SAMPLE_RATE
