#!/usr/bin/env python3
"""Measure WER between transcript columns in a CSV file."""

from __future__ import annotations

import argparse
import csv
import re
import statistics
import sys
from pathlib import Path


DEFAULT_HYP_COL = "transcript"
DEFAULT_REF_COL = "ground truth transcript"


class WerError(RuntimeError):
    pass


def normalize(text: str, keep_case: bool = False, keep_punct: bool = False) -> str:
    text = text.strip()
    if not keep_case:
        text = text.casefold()
    if not keep_punct:
        text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def edit_distance(reference: list[str], hypothesis: list[str]) -> int:
    previous = list(range(len(hypothesis) + 1))
    for i, ref_word in enumerate(reference, start=1):
        current = [i]
        for j, hyp_word in enumerate(hypothesis, start=1):
            if ref_word == hyp_word:
                current.append(previous[j - 1])
            else:
                current.append(1 + min(previous[j], current[j - 1], previous[j - 1]))
        previous = current
    return previous[-1]


def wer(reference_text: str, hypothesis_text: str) -> tuple[float, int, int]:
    reference = reference_text.split()
    hypothesis = hypothesis_text.split()
    edits = edit_distance(reference, hypothesis)
    words = len(reference)
    if words == 0:
        return (0.0 if len(hypothesis) == 0 else 1.0, edits, words)
    return edits / words, edits, words


def read_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise WerError(f"{path} has no CSV header")
        rows = [{key: (value or "").strip() for key, value in row.items()} for row in reader]
        return reader.fieldnames, rows


def write_rows(path: Path, rows: list[dict[str, object]], header: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=header, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def process(args: argparse.Namespace) -> int:
    header, rows = read_rows(args.csv_path)
    if args.reference_col not in header:
        raise WerError(f"Reference column '{args.reference_col}' not found. Available columns: {', '.join(header)}")
    if args.transcript_col not in header:
        raise WerError(f"Transcript column '{args.transcript_col}' not found. Available columns: {', '.join(header)}")

    output_rows: list[dict[str, object]] = []
    total_edits = 0
    total_words = 0
    row_wers = []

    for index, row in enumerate(rows, start=1):
        row_id = row.get(args.id_col, str(index)) if args.id_col else str(index)
        reference = normalize(row[args.reference_col], args.keep_case, args.keep_punct)
        hypothesis = normalize(row[args.transcript_col], args.keep_case, args.keep_punct)
        row_wer, edits, words = wer(reference, hypothesis)
        total_edits += edits
        total_words += words
        row_wers.append(row_wer)
        output_rows.append({
            "id": row_id,
            "reference": reference,
            "transcript": hypothesis,
            "reference_words": words,
            "edit_distance": edits,
            "wer": f"{row_wer:.6f}",
        })

    corpus_wer = total_edits / total_words if total_words else 0.0
    mean_wer = sum(row_wers) / len(row_wers) if row_wers else 0.0
    population_stdev = statistics.pstdev(row_wers) if row_wers else 0.0
    sample_stdev = statistics.stdev(row_wers) if len(row_wers) > 1 else 0.0

    for row in output_rows:
        row_wer = float(row["wer"])
        row["wer_deviation_from_mean"] = f"{row_wer - mean_wer:.6f}"

    print(f"rows: {len(rows)}")
    print(f"reference words: {total_words}")
    print(f"total edits: {total_edits}")
    print(f"corpus WER: {corpus_wer:.6f}")
    print(f"mean row WER: {mean_wer:.6f}")
    print(f"row WER stdev population: {population_stdev:.6f}")
    print(f"row WER stdev sample: {sample_stdev:.6f}")

    if args.output:
        write_rows(
            args.output,
            output_rows,
            ["id", "reference", "transcript", "reference_words", "edit_distance", "wer", "wer_deviation_from_mean"],
        )
        print(f"wrote per-row WER: {args.output}")

    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Measure WER between transcript and ground-truth transcript columns.")
    parser.add_argument("csv_path", type=Path, help="Input CSV path.")
    parser.add_argument("--transcript-col", default=DEFAULT_HYP_COL, help=f"Hypothesis column. Default: {DEFAULT_HYP_COL}")
    parser.add_argument("--reference-col", default=DEFAULT_REF_COL, help=f"Reference column. Default: {DEFAULT_REF_COL}")
    parser.add_argument("--id-col", default="id", help="Optional row id column. Default: id")
    parser.add_argument("--output", type=Path, help="Optional CSV path for per-row WER output.")
    parser.add_argument("--keep-case", action="store_true", help="Do not lowercase before scoring.")
    parser.add_argument("--keep-punct", action="store_true", help="Do not remove punctuation before scoring.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return process(args)
    except WerError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
