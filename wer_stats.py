#!/usr/bin/env python3
"""Compute average word error rate and standard deviation from transcript CSVs."""

from __future__ import annotations

import argparse
import csv
import re
import statistics
import string
from pathlib import Path


def normalize_text(text: str) -> list[str]:
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text)
    return text.strip().split()


def edit_distance(reference: list[str], hypothesis: list[str]) -> int:
    previous = list(range(len(hypothesis) + 1))

    for ref_index, ref_word in enumerate(reference, start=1):
        current = [ref_index]
        for hyp_index, hyp_word in enumerate(hypothesis, start=1):
            substitution_cost = 0 if ref_word == hyp_word else 1
            current.append(
                min(
                    previous[hyp_index] + 1,
                    current[hyp_index - 1] + 1,
                    previous[hyp_index - 1] + substitution_cost,
                )
            )
        previous = current

    return previous[-1]


def word_error_rate(reference_text: str, hypothesis_text: str) -> float:
    reference = normalize_text(reference_text)
    hypothesis = normalize_text(hypothesis_text)

    if not reference:
        return 0.0 if not hypothesis else 1.0

    return edit_distance(reference, hypothesis) / len(reference)


def read_wers(csv_path: Path) -> list[float]:
    wers = []

    with csv_path.open("r", encoding="utf-8", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        required_columns = {"transcript", "ground truth transcript"}
        missing_columns = required_columns - set(reader.fieldnames or [])
        if missing_columns:
            missing = ", ".join(sorted(missing_columns))
            raise ValueError(f"{csv_path} is missing required column(s): {missing}")

        for row in reader:
            wers.append(
                word_error_rate(
                    reference_text=row["ground truth transcript"],
                    hypothesis_text=row["transcript"],
                )
            )

    return wers


def print_stats(csv_path: Path) -> None:
    wers = read_wers(csv_path)
    if not wers:
        print(f"{csv_path}: no rows")
        return

    average = statistics.mean(wers)
    stdev = statistics.stdev(wers) if len(wers) > 1 else 0.0

    print(f"{csv_path}")
    print(f"  rows: {len(wers)}")
    print(f"  average WER: {average:.4f} ({average * 100:.2f}%)")
    print(f"  stdev WER:   {stdev:.4f} ({stdev * 100:.2f}%)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Show average word error rate and standard deviation for transcript CSVs."
    )
    parser.add_argument(
        "csv_path",
        type=Path,
        help="CSV file to evaluate.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print_stats(args.csv_path)


if __name__ == "__main__":
    main()
