"""Generate the pre-recorded LiveKit intro audio for the frontend."""

from __future__ import annotations

from pathlib import Path

from llm import INTRO_GREETING
from tts import TTSEngine


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = ROOT / "s2s.client" / "public" / "livekit-intro.wav"


def main() -> None:
    engine = TTSEngine()
    wav_bytes = engine.synthesize(INTRO_GREETING)
    OUTPUT_PATH.write_bytes(wav_bytes)
    print(f"Wrote intro audio to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
