#!/usr/bin/env python3
"""Create Gaussian-noise WAV copies at fixed SNR levels.

This script intentionally uses only the Python standard library, so it can run
even before the project's optional audio/science dependencies are installed.
"""

from __future__ import annotations

import argparse
import random
import wave
from pathlib import Path


DEFAULT_SNRS = (-5.0, 1.0)


def pcm_bytes_to_floats(frames: bytes, sample_width: int) -> list[float]:
    if sample_width == 1:
        return [(sample - 128) / 128.0 for sample in frames]

    if sample_width == 2:
        return [
            int.from_bytes(frames[i : i + 2], byteorder="little", signed=True) / 32768.0
            for i in range(0, len(frames), 2)
        ]

    if sample_width == 3:
        samples = []
        for i in range(0, len(frames), 3):
            sample_bytes = frames[i : i + 3]
            sign_byte = b"\xff" if sample_bytes[2] & 0x80 else b"\x00"
            sample = int.from_bytes(sample_bytes + sign_byte, byteorder="little", signed=True)
            samples.append(sample / 8388608.0)
        return samples

    if sample_width == 4:
        return [
            int.from_bytes(frames[i : i + 4], byteorder="little", signed=True) / 2147483648.0
            for i in range(0, len(frames), 4)
        ]

    raise ValueError(f"Unsupported WAV sample width: {sample_width} bytes")


def floats_to_pcm_bytes(samples: list[float], sample_width: int) -> bytes:
    clipped = [max(-1.0, min(1.0, sample)) for sample in samples]

    if sample_width == 1:
        return bytes(max(0, min(255, round(sample * 128.0 + 128))) for sample in clipped)

    if sample_width == 2:
        return b"".join(
            max(-32768, min(32767, round(sample * 32768.0))).to_bytes(
                2, byteorder="little", signed=True
            )
            for sample in clipped
        )

    if sample_width == 3:
        output = bytearray()
        for sample in clipped:
            value = max(-8388608, min(8388607, round(sample * 8388608.0)))
            output.extend(value.to_bytes(4, byteorder="little", signed=True)[:3])
        return bytes(output)

    if sample_width == 4:
        return b"".join(
            max(-2147483648, min(2147483647, round(sample * 2147483648.0))).to_bytes(
                4, byteorder="little", signed=True
            )
            for sample in clipped
        )

    raise ValueError(f"Unsupported WAV sample width: {sample_width} bytes")


def add_gaussian_noise(audio: list[float], snr_db: float, rng: random.Random) -> list[float]:
    """Return audio with zero-mean Gaussian noise at the requested SNR."""
    signal_power = sum(sample * sample for sample in audio) / len(audio)
    if signal_power == 0.0:
        return list(audio)

    noise_power = signal_power / (10.0 ** (snr_db / 10.0))
    noise_std = noise_power**0.5

    return [sample + rng.gauss(0.0, noise_std) for sample in audio]


def read_wav(path: Path) -> tuple[wave._wave_params, bytes]:
    with wave.open(str(path), "rb") as wav_file:
        return wav_file.getparams(), wav_file.readframes(wav_file.getnframes())


def write_wav(path: Path, params: wave._wave_params, frames: bytes) -> None:
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setparams(params)
        wav_file.writeframes(frames)


def make_noisy_dataset(
    clean_dir: Path,
    output_root: Path,
    snrs: tuple[float, ...],
    seed: int | None,
) -> None:
    rng = random.Random(seed)
    wav_paths = sorted(clean_dir.glob("*.wav"))

    if not wav_paths:
        raise FileNotFoundError(f"No .wav files found in {clean_dir}")

    for snr_db in snrs:
        output_dir = output_root / f"random_{int(snr_db)}db"
        output_dir.mkdir(parents=True, exist_ok=True)

        for wav_path in wav_paths:
            params, frames = read_wav(wav_path)
            audio = pcm_bytes_to_floats(frames, params.sampwidth)
            noisy_audio = add_gaussian_noise(audio, snr_db, rng)
            noisy_frames = floats_to_pcm_bytes(noisy_audio, params.sampwidth)

            output_path = output_dir / wav_path.name
            write_wav(output_path, params, noisy_frames)

        print(f"Wrote {len(wav_paths)} files to {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Add zero-mean Gaussian noise to clean WAV files at fixed SNR levels."
    )
    parser.add_argument(
        "--clean-dir",
        type=Path,
        default=Path("ready/clean"),
        help="Folder containing clean .wav files.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("ready"),
        help="Folder where noisy dataset folders will be created.",
    )
    parser.add_argument(
        "--snr",
        type=float,
        nargs="+",
        default=list(DEFAULT_SNRS),
        help="SNR values in dB. Defaults to 20 15.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional random seed for reproducible noise.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    make_noisy_dataset(
        clean_dir=args.clean_dir,
        output_root=args.output_root,
        snrs=tuple(args.snr),
        seed=args.seed,
    )


if __name__ == "__main__":
    main()