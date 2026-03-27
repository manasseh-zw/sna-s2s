"""ASR engine wrapping the noirlab-whisper-shona model."""

import io
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

DEFAULT_ASR_MODEL_PATH = Path("/Users/manasseh/models/shona/noirlab-whisper-shona")


class ASREngine:
    """Loads the Whisper-Shona model once and exposes a transcribe method."""

    def __init__(self, model_path: Path = DEFAULT_ASR_MODEL_PATH) -> None:
        self.model_path = model_path

        if not model_path.exists():
            raise FileNotFoundError(f"ASR model not found: {model_path}")

        if torch.cuda.is_available():
            self._device = "cuda:0"
            self._dtype = torch.float16
        elif torch.backends.mps.is_available():
            self._device = "mps"
            self._dtype = torch.float32
        else:
            self._device = "cpu"
            self._dtype = torch.float32

        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            str(model_path),
            torch_dtype=self._dtype,
            low_cpu_mem_usage=True,
        )
        self._processor = AutoProcessor.from_pretrained(str(model_path))

        self._pipe = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=self._processor.tokenizer,
            feature_extractor=self._processor.feature_extractor,
            torch_dtype=self._dtype,
            device=self._device,
        )

    def transcribe(
        self,
        audio_bytes: bytes,
        language: str = "shona",
        task: str = "transcribe",
    ) -> str:
        """Transcribe raw audio bytes and return the transcript string."""
        target_sr: int = self._processor.feature_extractor.sampling_rate

        # Write to a temp file so soundfile can detect the format
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name

        audio, sr = sf.read(tmp_path, dtype="float32")

        # Mix down to mono
        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        # Resample if needed
        if sr != target_sr:
            audio = resample_poly(audio, target_sr, sr).astype(np.float32)
            sr = target_sr

        result = self._pipe(
            {"raw": audio, "sampling_rate": sr},
            generate_kwargs={"language": language, "task": task},
        )
        return result["text"].strip()
