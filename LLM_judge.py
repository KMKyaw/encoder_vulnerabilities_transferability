#!/usr/bin/env python3

import argparse
import json
import re
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

def parse_args():
    parser = argparse.ArgumentParser(
        description="Use an LLM to judge whether model_answer and ground_truth_answer have the same meaning."
    )

    parser.add_argument(
        "csv_path",
        type=str,
        help="Path to the input CSV file.",
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save the judged CSV. Default: <input_stem>_llm_judged.csv",
    )

    parser.add_argument(
        "--model-id",
        type=str,
        default="Qwen/Qwen2.5-14B-Instruct",
        help="Hugging Face model ID.",
    )

    parser.add_argument(
        "--question-col",
        type=str,
        default="question",
        help="Name of the question column.",
    )

    parser.add_argument(
        "--model-answer-col",
        type=str,
        default="model_answer",
        help="Name of the model answer column.",
    )

    parser.add_argument(
        "--ground-truth-col",
        type=str,
        default="ground_truth_answer",
        help="Name of the ground truth answer column.",
    )

    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=160,
        help="Maximum number of new tokens to generate for each judgment.",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional number of rows to process, useful for testing.",
    )

    return parser.parse_args()


def build_output_path(csv_path: str, output: str | None) -> str:
    if output is not None:
        return output

    path = Path(csv_path)
    return str(path.with_name(f"{path.stem}_llm_judged.csv"))


def safe_str(value) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def extract_json(text: str) -> dict:
    """
    Extract the first JSON object from the model response.
    """
    text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*?\}", text, re.DOTALL)
    if match:
        candidate = match.group(0)
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

    return {
        "same_meaning": None,
        "reason": f"Could not parse JSON from response: {text[:300]}",
    }


def build_prompt(question: str, model_answer: str, ground_truth: str) -> str:
    return f"""
You are an evaluator. Decide whether two answers should be considered the same in meaning.

Use the question context when available.

Mark SAME when:
- One answer is written as digits and the other is written in words, but they represent the same value.
- Punctuation, capitalization, or harmless formatting differs.
- One answer includes extra surrounding words, but the actual answer is still clearly the same.
- Time formats are equivalent, such as "3:30" and "three thirty", when the question context supports it.
- Dates or months are expressed differently but answer the same requested value.

Mark DIFFERENT when:
- One answer is only partially correct or less specific than required.
- The actual person, place, date, number, code, or time differs.
- Ordinal and cardinal numbers differ in meaning, such as "seventh" vs "seven".
- The answers are related but not actually equivalent.

Return ONLY valid JSON in exactly this structure:
{{
  "same_meaning": true,
  "reason": "brief explanation"
}}

Question:
{question}

Model answer:
{model_answer}

Ground truth answer:
{ground_truth}
""".strip()


LOCAL_MODEL_PATH = "./models/Qwen2.5-14B-Instruct"


def load_model(model_id: str):
    model_path = LOCAL_MODEL_PATH if Path(LOCAL_MODEL_PATH).exists() else model_id

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        local_files_only=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype="auto",
        local_files_only=True,
    )

    return tokenizer, model


def judge_equivalence(
    tokenizer,
    model,
    question: str,
    model_answer: str,
    ground_truth: str,
    max_new_tokens: int,
) -> dict:
    prompt = build_prompt(question, model_answer, ground_truth)

    messages = [
        {"role": "user", "content": prompt}
    ]

    formatted_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(
        formatted_prompt,
        return_tensors="pt",
    )

    inputs = {
        key: value.to(model.device)
        for key, value in inputs.items()
    }

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated_ids = output_ids[0][inputs["input_ids"].shape[1]:]
    response = tokenizer.decode(
        generated_ids,
        skip_special_tokens=True,
    ).strip()

    parsed = extract_json(response)

    same_meaning = parsed.get("same_meaning")
    reason = parsed.get("reason", "")

    if isinstance(same_meaning, str):
        lowered = same_meaning.lower().strip()
        if lowered == "true":
            same_meaning = True
        elif lowered == "false":
            same_meaning = False
        else:
            same_meaning = None

    if same_meaning not in [True, False, None]:
        same_meaning = None

    return {
        "same_meaning": same_meaning,
        "reason": str(reason),
        "raw_llm_output": response,
    }


def main():
    args = parse_args()

    csv_path = args.csv_path
    output_path = build_output_path(csv_path, args.output)

    df = pd.read_csv(csv_path)

    required_cols = [
        args.model_answer_col,
        args.ground_truth_col,
    ]

    for col in required_cols:
        if col not in df.columns:
            raise ValueError(
                f"Required column '{col}' not found. "
                f"Available columns: {list(df.columns)}"
            )

    if args.limit is not None:
        df = df.head(args.limit).copy()

    tokenizer, model = load_model(args.model_id)

    llm_same_meaning = []
    llm_reason = []
    llm_raw_output = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Judging answers"):
        question = (
            safe_str(row[args.question_col])
            if args.question_col in df.columns
            else ""
        )

        model_answer = safe_str(row[args.model_answer_col])
        ground_truth = safe_str(row[args.ground_truth_col])

        result = judge_equivalence(
            tokenizer=tokenizer,
            model=model,
            question=question,
            model_answer=model_answer,
            ground_truth=ground_truth,
            max_new_tokens=args.max_new_tokens,
        )

        llm_same_meaning.append(result["same_meaning"])
        llm_reason.append(result["reason"])
        llm_raw_output.append(result["raw_llm_output"])

    df["llm_same_meaning"] = llm_same_meaning
    df["llm_reason"] = llm_reason
    df["llm_raw_output"] = llm_raw_output

    df.to_csv(output_path, index=False)

    same_count = df["llm_same_meaning"].eq(True).sum()
    false_count = df["llm_same_meaning"].eq(False).sum()
    unknown_count = df["llm_same_meaning"].isna().sum()
    total_count = len(df)

    print()
    print("Done.")
    print(f"Equivalent answers: {same_count} / {total_count}")
    print(f"Different answers:  {false_count} / {total_count}")
    print(f"Unparsed/unknown:   {unknown_count} / {total_count}")
    print(f"Saved output to:    {output_path}")


if __name__ == "__main__":
    main()