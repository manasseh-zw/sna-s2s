"""TTS engine wrapping the mms-tts-sna VITS model."""

import io
from pathlib import Path

import torch
from scipy.io.wavfile import write as wav_write
from transformers import AutoTokenizer, VitsModel

DEFAULT_TTS_MODEL_PATH = Path("/Users/manasseh/models/shona/mms-tts-sna")


class TTSEngine:
    """Loads the MMS-TTS-SNA model once and exposes a synthesize method."""

    def __init__(self, model_path: Path = DEFAULT_TTS_MODEL_PATH) -> None:
        self.model_path = model_path

        if not model_path.exists():
            raise FileNotFoundError(f"TTS model not found: {model_path}")

        self._tokenizer = AutoTokenizer.from_pretrained(str(model_path))
        self._model = VitsModel.from_pretrained(str(model_path))

    def synthesize(self, text: str) -> bytes:
        """Synthesize Shona text and return WAV audio as raw bytes."""
        inputs = self._tokenizer(text, return_tensors="pt")

        with torch.no_grad():
            waveform = self._model(**inputs).waveform.squeeze().cpu().numpy()

        sample_rate: int = self._model.config.sampling_rate

        buf = io.BytesIO()
        wav_write(buf, sample_rate, waveform)
        return buf.getvalue()
