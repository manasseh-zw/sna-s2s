"""ASR engine – Whisper Large v3 + Wav2Vec2-BERT (CTC) for Shona."""

import subprocess
from pathlib import Path

import numpy as np
import torch
from transformers import (
    AutoModelForSpeechSeq2Seq,
    AutoProcessor,
    Wav2Vec2BertForCTC,
    Wav2Vec2BertProcessor,
    pipeline,
)

DEFAULT_WHISPER_PATH = Path("/Users/manasseh/models/shona/noirlab-whisper-shona")
DEFAULT_W2V_PATH = Path("/Users/manasseh/models/shona/w2v-bert-sna")

_SAMPLE_RATE = 16_000  # both models expect 16 kHz


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
    """Return True if the clip is effectively silence (avoids Whisper hallucinations)."""
    return float(np.sqrt(np.mean(audio ** 2))) < threshold


class ASREngine:
    """Hosts both Whisper and Wav2Vec2-BERT engines. Call the one you want."""

    def __init__(
        self,
        whisper_path: Path = DEFAULT_WHISPER_PATH,
        w2v_path: Path = DEFAULT_W2V_PATH,
    ) -> None:
        # ── Device selection ────────────────────────────────────────────────
        if torch.cuda.is_available():
            self._device = "cuda:0"
            self._dtype = torch.float16
        elif torch.backends.mps.is_available():
            self._device = "mps"
            self._dtype = torch.float16
        else:
            self._device = "cpu"
            self._dtype = torch.float32

        # ── Whisper Large v3 ────────────────────────────────────────────────
        print(f"  Loading Whisper from {whisper_path} …")
        if not whisper_path.exists():
            raise FileNotFoundError(f"Whisper model not found: {whisper_path}")

        _w_model = AutoModelForSpeechSeq2Seq.from_pretrained(
            str(whisper_path),
            dtype=self._dtype,
            low_cpu_mem_usage=True,
            attn_implementation="eager",
        ).to(self._device).eval()

        self._w_processor = AutoProcessor.from_pretrained(str(whisper_path))

        self._whisper_pipe = pipeline(
            "automatic-speech-recognition",
            model=_w_model,
            tokenizer=self._w_processor.tokenizer,
            feature_extractor=self._w_processor.feature_extractor,
            dtype=self._dtype,
            device=self._device,
        )

        # ── Wav2Vec2-BERT (CTC) ─────────────────────────────────────────────
        print(f"  Loading Wav2Vec2-BERT from {w2v_path} …")
        if not w2v_path.exists():
            raise FileNotFoundError(f"Wav2Vec2-BERT model not found: {w2v_path}")

        self._w2v_processor = Wav2Vec2BertProcessor.from_pretrained(str(w2v_path))
        self._w2v_model = (
            Wav2Vec2BertForCTC.from_pretrained(str(w2v_path))
            .to(self._device)
            .eval()
        )

    # ── Public API ──────────────────────────────────────────────────────────

    def transcribe(
        self,
        audio_bytes: bytes,
        language: str = "shona",
        task: str = "transcribe",
    ) -> str:
        """Transcribe with Whisper Large v3 (highest accuracy)."""
        audio = _decode_audio(audio_bytes)
        if _is_silent(audio):
            return ""

        result = self._whisper_pipe(
            {"raw": audio, "sampling_rate": _SAMPLE_RATE},
            generate_kwargs={
                "language": language,
                "task": task,
                "max_new_tokens": 224,
            },
        )
        return result["text"].strip()

    def transcribe_w2v(self, audio_bytes: bytes) -> str:
        """Transcribe with Wav2Vec2-BERT CTC (faster, Shona fine-tuned)."""
        audio = _decode_audio(audio_bytes)
        if _is_silent(audio):
            return ""

        inputs = self._w2v_processor(
            audio,
            sampling_rate=_SAMPLE_RATE,
            return_tensors="pt",
            padding=True,
        )
        # Move all tensors to device
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        with torch.inference_mode():
            logits = self._w2v_model(**inputs).logits

        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = self._w2v_processor.batch_decode(predicted_ids)
        return transcription[0].strip()
