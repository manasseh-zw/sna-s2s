"""TTS engine wrapping the mms-tts-sna VITS model."""

import io
from pathlib import Path

import numpy as np
import torch
from scipy.io.wavfile import write as wav_write
from transformers import AutoTokenizer, VitsModel

DEFAULT_TTS_MODEL_PATH = Path("/Users/manasseh/models/shona/mms-tts-sna")


class TTSEngine:
    """Loads the MMS-TTS-SNA model once and exposes a synthesize method."""

    def __init__(self, model_path: Path = DEFAULT_TTS_MODEL_PATH) -> None:
        if not model_path.exists():
            raise FileNotFoundError(f"TTS model not found: {model_path}")

        if torch.cuda.is_available():
            self._device = "cuda:0"
            self._dtype = torch.float16
        elif torch.backends.mps.is_available():
            self._device = "mps"
            self._dtype = torch.float16
        else:
            self._device = "cpu"
            self._dtype = torch.float32

        self._tokenizer = AutoTokenizer.from_pretrained(str(model_path))
        self._model = (
            VitsModel.from_pretrained(str(model_path), torch_dtype=self._dtype)
            .to(self._device)
            .eval()
        )

    def synthesize(self, text: str) -> bytes:
        """Synthesize Shona text and return WAV audio as raw bytes."""
        inputs = self._tokenizer(text, return_tensors="pt")
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        with torch.inference_mode():
            waveform = self._model(**inputs).waveform.squeeze().cpu().float().numpy()

        sample_rate: int = self._model.config.sampling_rate
        pcm_waveform = np.clip(waveform, -1.0, 1.0)
        pcm_waveform = (pcm_waveform * 32767.0).astype(np.int16)

        buf = io.BytesIO()
        wav_write(buf, sample_rate, pcm_waveform)
        return buf.getvalue()
