"""ASR engine – Wav2Vec2-BERT (CTC) for Shona."""

import subprocess
from pathlib import Path

import numpy as np
import torch
from transformers import Wav2Vec2BertForCTC, Wav2Vec2BertProcessor

DEFAULT_W2V_PATH = Path("/Users/manasseh/models/shona/w2v-bert-sna")

_SAMPLE_RATE = 16_000  # model expects 16 kHz
_CHUNK_SECONDS = 25  # process audio in 25-second chunks to avoid OOM
_CHUNK_SAMPLES = _CHUNK_SECONDS * _SAMPLE_RATE


def _decode_audio(audio_bytes: bytes, target_sr: int = _SAMPLE_RATE) -> np.ndarray:
    """Convert any audio format (webm, wav, ogg…) to float32 mono PCM via ffmpeg."""
    result = subprocess.run(
        [
            "ffmpeg",
            "-hide_banner", "-loglevel", "error",
            "-i", "pipe:0",
            "-ac", "1",
            "-ar", str(target_sr),
            "-f", "f32le",
            "pipe:1",
        ],
        input=audio_bytes,
        capture_output=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {result.stderr.decode(errors='replace')}")
    return np.frombuffer(result.stdout, dtype=np.float32)


def _is_silent(audio: np.ndarray, threshold: float = 1e-4) -> bool:
    """Return True if the clip is effectively silence."""
    return float(np.sqrt(np.mean(audio ** 2))) < threshold


class ASREngine:
    """Wav2Vec2-BERT CTC engine for Shona speech recognition."""

    def __init__(self, w2v_path: Path = DEFAULT_W2V_PATH) -> None:
        if torch.cuda.is_available():
            self._device = "cuda:0"
            self._dtype = torch.float16
        elif torch.backends.mps.is_available():
            self._device = "mps"
            self._dtype = torch.float16
        else:
            self._device = "cpu"
            self._dtype = torch.float32

        print(f"  Loading Wav2Vec2-BERT from {w2v_path} …")
        if not w2v_path.exists():
            raise FileNotFoundError(f"Wav2Vec2-BERT model not found: {w2v_path}")

        self._processor = Wav2Vec2BertProcessor.from_pretrained(str(w2v_path))
        self._model = (
            Wav2Vec2BertForCTC.from_pretrained(
                str(w2v_path), torch_dtype=self._dtype,
            )
            .to(self._device)
            .eval()
        )

    def _transcribe_chunk(self, audio: np.ndarray) -> str:
        """Transcribe a single audio chunk that fits in memory."""
        inputs = self._processor(
            audio,
            sampling_rate=_SAMPLE_RATE,
            return_tensors="pt",
            padding=True,
        )
        inputs = {
            k: v.to(device=self._device, dtype=self._dtype if v.is_floating_point() else None)
            for k, v in inputs.items()
        }

        with torch.inference_mode():
            logits = self._model(**inputs).logits

        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = self._processor.batch_decode(predicted_ids)
        return transcription[0].strip()

    def transcribe(self, audio_bytes: bytes) -> str:
        """Transcribe with Wav2Vec2-BERT CTC (Shona fine-tuned).

        Long audio is split into chunks to avoid OOM on the attention matrices.
        """
        audio = _decode_audio(audio_bytes)
        if _is_silent(audio):
            return ""

        if len(audio) <= _CHUNK_SAMPLES:
            return self._transcribe_chunk(audio)

        parts: list[str] = []
        for start in range(0, len(audio), _CHUNK_SAMPLES):
            chunk = audio[start : start + _CHUNK_SAMPLES]
            if _is_silent(chunk):
                continue
            text = self._transcribe_chunk(chunk)
            if text:
                parts.append(text)

        return " ".join(parts)
