#!/usr/bin/env python3
"""Ask Qwen2-Audio questions about WAV files and save answer-only outputs."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any


DEFAULT_MODEL_PATH = Path(
    "~/qian_jiang/models/Qwen2-Audio-7B-Instruct/"
    "models--Qwen--Qwen2-Audio-7B-Instruct"
).expanduser()

OUTPUT_COLUMNS = (
    "id",
    "category",
    "question",
    "model_answer",
    "ground_truth_answer",
    "answer_type",
    "difficulty",
)


def resolve_model_path(model_path: Path) -> Path:
    snapshots_dir = model_path / "snapshots"
    if not snapshots_dir.exists():
        return model_path

    snapshots = sorted(path for path in snapshots_dir.iterdir() if path.is_dir())
    if not snapshots:
        raise FileNotFoundError(f"No snapshots found in {snapshots_dir}")

    return snapshots[-1]


def load_metadata(reference_csv: Path) -> dict[str, dict[str, str]]:
    metadata: dict[str, dict[str, str]] = {}

    with reference_csv.open("r", encoding="utf-8", newline="") as csv_file:
        reader = csv.reader(csv_file)
        next(reader, None)

        for row in reader:
            if len(row) < 7:
                raise ValueError(
                    f"{reference_csv} row has {len(row)} columns, expected 7: {row}"
                )

            audio_id, category, transcript, question, answer, answer_type, difficulty = row[:7]
            metadata[audio_id] = {
                "id": audio_id,
                "category": category,
                "ground_truth_transcript": transcript,
                "question": question,
                "answer": answer,
                "answer_type": answer_type,
                "difficulty": difficulty,
            }

    return metadata


def choose_torch_dtype(torch_module: Any, device: str) -> Any:
    if device == "cpu":
        return torch_module.float32
    return torch_module.float16


def build_max_memory(torch_module: Any, gpu_memory: str, cpu_memory: str) -> dict[Any, str]:
    max_memory: dict[Any, str] = {"cpu": cpu_memory}
    for gpu_index in range(torch_module.cuda.device_count()):
        max_memory[gpu_index] = gpu_memory
    return max_memory


def load_qwen2_audio(
    model_path: Path,
    device: str,
    gpu_memory: str,
    cpu_memory: str,
    offload_folder: Path,
):
    import torch
    from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration

    resolved_path = resolve_model_path(model_path)
    processor = AutoProcessor.from_pretrained(resolved_path, trust_remote_code=True)
    torch_dtype = choose_torch_dtype(torch, device)

    if device == "auto":
        offload_folder.mkdir(parents=True, exist_ok=True)
        model = Qwen2AudioForConditionalGeneration.from_pretrained(
            resolved_path,
            torch_dtype=torch_dtype,
            device_map="auto",
            max_memory=build_max_memory(torch, gpu_memory, cpu_memory),
            offload_folder=offload_folder,
            offload_state_dict=True,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
    else:
        model = Qwen2AudioForConditionalGeneration.from_pretrained(
            resolved_path,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        ).to(device)

    model.eval()
    return processor, model


def model_device(model) -> Any:
    return next(model.parameters()).device


def build_prompt(question: str) -> str:
    return (
        "Listen to the audio and answer the question.\n"
        "Return only the short answer. Do not write a sentence. Do not explain.\n"
    )


def answer_audio(processor, model, audio_path: Path, question: str, max_new_tokens: int) -> str:
    import librosa
    import torch
    print(f"Answering {audio_path.name} question: {question}")
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio_url": str(audio_path)},
                {"type": "text", "text": build_prompt(question)},
            ],
        }
    ]

    text = processor.apply_chat_template(
        conversation,
        add_generation_prompt=True,
        tokenize=False,
    )
    audio, _ = librosa.load(
        audio_path,
        sr=processor.feature_extractor.sampling_rate,
        mono=True,
    )
    inputs = processor(
        text=text,
        audio=[audio],
        sampling_rate=processor.feature_extractor.sampling_rate,
        return_tensors="pt",
        padding=True,
    )
    inputs = {key: value.to(model_device(model)) for key, value in inputs.items()}

    with torch.inference_mode():
        generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)

    generated_ids = generated_ids[:, inputs["input_ids"].shape[1] :]
    answer = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]

    return answer.strip().splitlines()[0].strip()


def write_answers(
    audio_dir: Path,
    output_csv: Path,
    metadata: dict[str, dict[str, str]],
    processor,
    model,
    max_new_tokens: int,
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
            row = metadata.get(audio_id)
            if row is None:
                raise KeyError(f"No metadata found for {audio_id} in reference CSV")

            model_answer = answer_audio(
                processor=processor,
                model=model,
                audio_path=wav_path,
                question=row["question"],
                max_new_tokens=max_new_tokens,
            )
            writer.writerow(
                {
                    "id": audio_id,
                    "category": row["category"],
                    "question": row["question"],
                    "model_answer": model_answer,
                    "ground_truth_answer": row["answer"],
                    "answer_type": row["answer_type"],
                    "difficulty": row["difficulty"],
                }
            )
            print(f"[{index}/{len(wav_paths)}] {wav_path.name}: {model_answer}")

    print(f"Wrote {output_csv}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Use local Qwen2-Audio-7B-Instruct to answer questions from WAV audio."
    )
    parser.add_argument(
        "audio_dir",
        type=Path,
        help="Folder containing .wav files, e.g. ready/random_-5db.",
    )
    parser.add_argument(
        "--reference-csv",
        type=Path,
        default=Path("ready/generated_transcript.csv"),
        help="CSV with id/category/transcript/question/answer metadata.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Output CSV path. Defaults to <audio_dir>_qwen2_audio_answers.csv in ready/.",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="Local Qwen2-Audio model path or Hugging Face cache model directory.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Device for the model: auto, cpu, cuda, cuda:0, etc.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=16,
        help="Maximum generated tokens for the short answer.",
    )
    parser.add_argument(
        "--gpu-memory",
        default="8GiB",
        help="Per-GPU memory limit for --device auto, e.g. 8GiB or 9GiB.",
    )
    parser.add_argument(
        "--cpu-memory",
        default="48GiB",
        help="CPU RAM limit for --device auto offloading.",
    )
    parser.add_argument(
        "--offload-folder",
        type=Path,
        default=Path(".cache/qwen2_audio_offload"),
        help="Folder for model weights offloaded by accelerate.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_csv = args.output_csv
    if output_csv is None:
        output_csv = Path("ready") / f"{args.audio_dir.name}_qwen2_audio_answers.csv"

    metadata = load_metadata(args.reference_csv)
    processor, model = load_qwen2_audio(
        model_path=args.model_path.expanduser(),
        device=args.device,
        gpu_memory=args.gpu_memory,
        cpu_memory=args.cpu_memory,
        offload_folder=args.offload_folder,
    )
    write_answers(
        audio_dir=args.audio_dir,
        output_csv=output_csv,
        metadata=metadata,
        processor=processor,
        model=model,
        max_new_tokens=args.max_new_tokens,
    )


if __name__ == "__main__":
    main()
