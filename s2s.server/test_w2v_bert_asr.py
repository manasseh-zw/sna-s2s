#!/usr/bin/env python3
"""CLI to transcribe a local audio file with the Wav2Vec2-BERT CTC ASR model."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from asr import ASREngine, DEFAULT_W2V_PATH


def main() -> int:
    p = argparse.ArgumentParser(description="Transcribe audio with Wav2Vec2-BERT (CTC).")
    p.add_argument("audio", type=Path, help="Path to an audio file (wav, mp3, webm, …)")
    p.add_argument(
        "-m",
        "--model",
        type=Path,
        default=DEFAULT_W2V_PATH,
        help=f"Directory with the fine-tuned model (default: {DEFAULT_W2V_PATH})",
    )
    args = p.parse_args()

    audio_path = args.audio.expanduser().resolve()
    if not audio_path.is_file():
        print(f"error: not a file: {audio_path}", file=sys.stderr)
        return 1

    model_path = args.model.expanduser().resolve()
    engine = ASREngine(w2v_path=model_path)
    audio_bytes = audio_path.read_bytes()
    text = engine.transcribe(audio_bytes)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
