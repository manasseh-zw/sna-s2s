"""ASR engine wrapping the noirlab-whisper-shona model."""

import subprocess
from pathlib import Path

import numpy as np
import torch
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
        """Transcribe raw audio bytes (any browser format) and return the transcript."""
        target_sr: int = self._processor.feature_extractor.sampling_rate

        # Use ffmpeg to decode any incoming format (webm, ogg, mp4, wav…) into
        # raw 32-bit float PCM at the model's expected sample rate.
        ffmpeg = subprocess.run(
            [
                "ffmpeg",
                "-hide_banner", "-loglevel", "error",
                "-i", "pipe:0",                  # read from stdin
                "-ac", "1",                       # mono
                "-ar", str(target_sr),            # resample to target SR
                "-f", "f32le",                    # raw float32 little-endian
                "pipe:1",                         # write to stdout
            ],
            input=audio_bytes,
            capture_output=True,
        )

        if ffmpeg.returncode != 0:
            raise RuntimeError(
                f"ffmpeg failed: {ffmpeg.stderr.decode(errors='replace')}"
            )

        audio = np.frombuffer(ffmpeg.stdout, dtype=np.float32)

        result = self._pipe(
            {"raw": audio, "sampling_rate": target_sr},
            generate_kwargs={"language": language, "task": task},
        )
        return result["text"].strip()

