#!/usr/bin/env python3
"""Transcribe WAV folders with Whisper large-v3 and write evaluation CSVs."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any


DEFAULT_FOLDERS = ("clean", "random_15db", "random_20db")
OUTPUT_COLUMNS = (
    "id",
    "category",
    "transcript",
    "ground truth transcript",
    "answer",
    "answer_type",
    "difficulty",
)


def load_metadata(reference_csv: Path) -> dict[str, dict[str, str]]:
    with reference_csv.open("r", encoding="utf-8", newline="") as csv_file:
        rows = csv.DictReader(csv_file)
        return {row["id"]: row for row in rows}


def choose_device(device_setting: str) -> tuple[int, Any]:
    import torch

    if device_setting == "cpu":
        return -1, torch.float32

    if device_setting.startswith("cuda"):
        device_id = 0 if device_setting == "cuda" else int(device_setting.split(":", 1)[1])
        return device_id, torch.float16

    if torch.cuda.is_available() and torch.cuda.get_device_capability(0) >= (7, 5):
        return 0, torch.float16

    return -1, torch.float32


def build_transcriber(model_name: str, batch_size: int, device_setting: str):
    from transformers import pipeline

    device, torch_dtype = choose_device(device_setting)
    return pipeline(
        task="automatic-speech-recognition",
        model=model_name,
        dtype=torch_dtype,
        device=device,
        batch_size=batch_size,
    )


def transcribe_file(transcriber, wav_path: Path, language: str) -> str:
    result = transcriber(
        str(wav_path),
        generate_kwargs={
            "language": language,
            "task": "transcribe",
        },
    )
    return result["text"].strip()


def metadata_for_audio(audio_id: str, metadata: dict[str, dict[str, str]]) -> dict[str, str]:
    row = metadata.get(audio_id, {})
    return {
        "id": audio_id,
        "category": row.get("category", audio_id.rsplit("_", 1)[0]),
        "ground truth transcript": row.get("question", ""),
        "answer": row.get("answer", ""),
        "answer_type": row.get("answer_type", ""),
        "difficulty": row.get("difficulty", ""),
    }


def write_folder_csv(
    transcriber,
    audio_dir: Path,
    output_csv: Path,
    metadata: dict[str, dict[str, str]],
    language: str,
) -> None:
    wav_paths = sorted(audio_dir.glob("*.wav"))
    if not wav_paths:
        raise FileNotFoundError(f"No .wav files found in {audio_dir}")

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()

        for index, wav_path in enumerate(wav_paths, start=1):
            audio_id = wav_path.stem
            row = metadata_for_audio(audio_id, metadata)
            row["transcript"] = transcribe_file(transcriber, wav_path, language)
            writer.writerow(row)
            print(f"[{audio_dir.name}] {index}/{len(wav_paths)} {wav_path.name}")

    print(f"Wrote {output_csv}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Transcribe clean/noisy WAV folders with openai/whisper-large-v3."
    )
    parser.add_argument(
        "--ready-dir",
        type=Path,
        default=Path("ready"),
        help="Directory containing clean, random_15db, and random_20db folders.",
    )
    parser.add_argument(
        "--reference-csv",
        type=Path,
        default=Path("ready/generated_transcript.csv"),
        help="CSV containing id/category/answer/answer_type/difficulty metadata.",
    )
    parser.add_argument(
        "--folders",
        nargs="+",
        default=list(DEFAULT_FOLDERS),
        help="Audio folder names inside --ready-dir.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("ready"),
        help="Directory where transcript CSV files will be written.",
    )
    parser.add_argument(
        "--model",
        default="openai/whisper-large-v3",
        help="Hugging Face model name or local model path.",
    )
    parser.add_argument(
        "--language",
        default="english",
        help="Whisper language hint.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="ASR pipeline batch size.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=("auto", "cpu", "cuda", "cuda:0", "cuda:1"),
        help="Device to use. Auto uses CUDA only when the installed PyTorch supports the GPU.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metadata = load_metadata(args.reference_csv)
    transcriber = build_transcriber(args.model, args.batch_size, args.device)

    for folder in args.folders:
        audio_dir = args.ready_dir / folder
        output_csv = args.output_dir / f"{folder}_whisper_large_v3.csv"
        write_folder_csv(
            transcriber=transcriber,
            audio_dir=audio_dir,
            output_csv=output_csv,
            metadata=metadata,
            language=args.language,
        )


if __name__ == "__main__":
    main()
