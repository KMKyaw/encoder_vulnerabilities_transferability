#!/usr/bin/env python3
"""Create PGD adversarial WAVs that increase Whisper large-v3 transcription loss.

This is an untargeted evaluation attack: it keeps each waveform inside an
L-infinity perturbation bound while maximizing Whisper's loss on the known
ground-truth transcript.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import numpy as np


SAMPLE_RATE = 16000
N_FFT = 400
HOP_LENGTH = 160
CHUNK_LENGTH_SECONDS = 30


def load_ground_truth(reference_csv: Path) -> dict[str, str]:
    ground_truth: dict[str, str] = {}

    with reference_csv.open("r", encoding="utf-8", newline="") as csv_file:
        reader = csv.reader(csv_file)
        next(reader, None)
        for row in reader:
            if len(row) < 3:
                continue
            audio_id = row[0]
            spoken_transcript = row[2]
            ground_truth[audio_id] = spoken_transcript

    return ground_truth


def choose_device(device_setting: str):
    import torch

    if device_setting == "auto":
        if torch.cuda.is_available() and torch.cuda.get_device_capability(0) >= (7, 5):
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(device_setting)


def resolve_model_dtype(torch_module: Any, device) -> Any:
    if device.type == "cuda":
        return torch_module.float16
    return torch_module.float32


def load_whisper(model_name: str, device_setting: str):
    import torch
    from transformers import AutoProcessor, WhisperForConditionalGeneration

    device = choose_device(device_setting)
    dtype = resolve_model_dtype(torch, device)

    processor = AutoProcessor.from_pretrained(model_name)
    processor.tokenizer.set_prefix_tokens(language="english", task="transcribe")

    model = WhisperForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=dtype,
    ).to(device)
    model.train()
    model.config.use_cache = False

    return processor, model, device, dtype


def read_wav(path: Path) -> tuple[np.ndarray, int, str]:
    import soundfile as sf

    audio, sample_rate = sf.read(path, dtype="float32", always_2d=False)
    info = sf.info(path)

    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    if sample_rate != SAMPLE_RATE:
        import librosa

        audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=SAMPLE_RATE)
        sample_rate = SAMPLE_RATE

    return audio.astype(np.float32), sample_rate, info.subtype


def write_wav(path: Path, audio: np.ndarray, sample_rate: int, subtype: str) -> None:
    import soundfile as sf

    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, audio, sample_rate, subtype=subtype)


def pad_or_trim(audio, target_length: int):
    import torch

    if audio.shape[-1] > target_length:
        return audio[..., :target_length]

    if audio.shape[-1] < target_length:
        return torch.nn.functional.pad(audio, (0, target_length - audio.shape[-1]))

    return audio


def build_mel_filter(processor, device):
    import librosa
    import torch

    feature_extractor = processor.feature_extractor
    n_mels = feature_extractor.feature_size
    mel = librosa.filters.mel(
        sr=SAMPLE_RATE,
        n_fft=N_FFT,
        n_mels=n_mels,
        fmin=0,
        fmax=8000,
        norm="slaney",
        htk=False,
    )
    return torch.tensor(mel, device=device, dtype=torch.float32)


def differentiable_log_mel(audio, mel_filter):
    import torch

    audio = audio.float()
    window = torch.hann_window(N_FFT, device=audio.device, dtype=torch.float32)
    stft = torch.stft(
        audio,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        win_length=N_FFT,
        window=window,
        center=True,
        pad_mode="reflect",
        return_complex=True,
    )
    magnitudes = stft[..., :-1].abs().pow(2)
    mel_spec = torch.matmul(mel_filter, magnitudes)
    log_spec = torch.clamp(mel_spec, min=1e-10).log10()
    log_spec = torch.maximum(log_spec, log_spec.amax(dim=(-2, -1), keepdim=True) - 8.0)
    log_spec = (log_spec + 4.0) / 4.0

    return log_spec


def labels_for_text(processor, text: str, device):
    labels = processor.tokenizer(
        text,
        return_tensors="pt",
        add_special_tokens=True,
    ).input_ids
    return labels.to(device)


def pgd_attack(
    audio_np: np.ndarray,
    transcript: str,
    processor,
    model,
    device,
    dtype,
    epsilon: float,
    step_size: float,
    steps: int,
) -> np.ndarray:
    import torch

    original_length = min(len(audio_np), SAMPLE_RATE * CHUNK_LENGTH_SECONDS)
    target_length = SAMPLE_RATE * CHUNK_LENGTH_SECONDS
    # Keep waveform optimization and STFT in float32. CUDA cuFFT does not support
    # half precision for n_fft=400, which is Whisper's required window size.
    original = torch.tensor(audio_np, device=device, dtype=torch.float32)
    original = pad_or_trim(original, target_length)

    mask = torch.zeros_like(original)
    mask[:original_length] = 1.0

    delta = torch.zeros_like(original, requires_grad=True)
    labels = labels_for_text(processor, transcript, device)
    mel_filter = build_mel_filter(processor, device)

    for step in range(1, steps + 1):
        adv_audio = torch.clamp(original + delta * mask, -1.0, 1.0)
        input_features = differentiable_log_mel(adv_audio, mel_filter).unsqueeze(0).to(dtype)
        outputs = model(input_features=input_features, labels=labels)
        loss = outputs.loss

        if delta.grad is not None:
            delta.grad.zero_()
        loss.backward()

        with torch.no_grad():
            delta += step_size * delta.grad.sign() * mask
            delta.clamp_(min=-epsilon, max=epsilon)
            delta.data = torch.clamp(original + delta, -1.0, 1.0) - original

        print(f"    step {step}/{steps} loss={loss.item():.4f}")

    adversarial = torch.clamp(original + delta.detach() * mask, -1.0, 1.0)
    return adversarial[:original_length].float().cpu().numpy()


def attack_folder(
    clean_dir: Path,
    output_dir: Path,
    reference_csv: Path,
    model_name: str,
    device_setting: str,
    epsilon: float,
    step_size: float,
    steps: int,
    limit: int | None,
    start_index: int,
    skip_existing: bool,
) -> None:
    ground_truth = load_ground_truth(reference_csv)
    processor, model, device, dtype = load_whisper(model_name, device_setting)

    wav_paths = sorted(clean_dir.glob("*.wav"))
    if start_index < 1:
        raise ValueError("--start-index must be 1 or greater")
    if start_index > 1:
        wav_paths = wav_paths[start_index - 1 :]
    if limit is not None:
        wav_paths = wav_paths[:limit]
    if not wav_paths:
        raise FileNotFoundError(f"No .wav files found in {clean_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    for index, wav_path in enumerate(wav_paths, start=start_index):
        output_path = output_dir / wav_path.name
        if skip_existing and output_path.exists():
            print(f"[{index}/{start_index + len(wav_paths) - 1}] Skipping existing {wav_path.name}")
            continue

        audio_id = wav_path.stem
        transcript = ground_truth.get(audio_id)
        if transcript is None:
            raise KeyError(f"No ground-truth transcript found for {audio_id}")

        print(f"[{index}/{start_index + len(wav_paths) - 1}] Attacking {wav_path.name}: {transcript}")
        audio_np, sample_rate, subtype = read_wav(wav_path)
        adversarial = pgd_attack(
            audio_np=audio_np,
            transcript=transcript,
            processor=processor,
            model=model,
            device=device,
            dtype=dtype,
            epsilon=epsilon,
            step_size=step_size,
            steps=steps,
        )
        write_wav(output_path, adversarial, sample_rate, subtype)

    print(f"Wrote adversarial WAVs to {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Use PGD to create Whisper large-v3 adversarial WAVs from clean audio."
    )
    parser.add_argument(
        "--clean-dir",
        type=Path,
        default=Path("ready/clean"),
        help="Folder containing clean .wav files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("ready/pgd_whisper_large_v3"),
        help="Folder where adversarial .wav files will be written.",
    )
    parser.add_argument(
        "--reference-csv",
        type=Path,
        default=Path("ready/generated_transcript.csv"),
        help="CSV containing ground-truth spoken transcripts.",
    )
    parser.add_argument(
        "--model",
        default="openai/whisper-large-v3",
        help="Whisper model name or local model path.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Device: auto, cpu, cuda, cuda:0, etc.",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=0.01,
        help="L-infinity perturbation limit in normalized waveform amplitude.",
    )
    parser.add_argument(
        "--step-size",
        type=float,
        default=0.002,
        help="PGD step size in normalized waveform amplitude.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=10,
        help="Number of PGD steps per file.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional number of files to attack for testing.",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=1,
        help="1-based position in the sorted input .wav list to start from.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip files whose output .wav already exists.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    attack_folder(
        clean_dir=args.clean_dir,
        output_dir=args.output_dir,
        reference_csv=args.reference_csv,
        model_name=args.model,
        device_setting=args.device,
        epsilon=args.epsilon,
        step_size=args.step_size,
        steps=args.steps,
        limit=args.limit,
        start_index=args.start_index,
        skip_existing=args.skip_existing,
    )


if __name__ == "__main__":
    main()
