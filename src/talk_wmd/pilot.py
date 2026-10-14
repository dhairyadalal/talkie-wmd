"""Small local Talkie/WMDP-Bio format pilot with complete local run traces."""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
MODEL_REPO = "niklassheth/talkie-1930-13b-base-GGUF"
MODEL_FILE = "talkie-base-Q4_K_M.gguf"
MODEL_BYTES = 8_453_708_352
MODEL_REVISION = "b4605dc53edb5542803053e87a6d3c0ac3d43cf7"
DATASET_REPO = "cais/wmdp"
DATASET_FILE = "wmdp-bio/test-00000-of-00001.parquet"
DATASET_REVISION = "7125571f22f032c56415e7980f48d877dd830ff8"
LETTERS = "ABCD"


def download_model() -> Path:
    """Download only the 4-bit original base model, not the 53 GB checkpoint."""
    path = ROOT / "models" / MODEL_FILE
    if path.exists() and path.stat().st_size == MODEL_BYTES:
        return path
    # Disable Xet transfers by default; callers can override this setting.
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    from huggingface_hub import hf_hub_download

    return Path(
        hf_hub_download(
            repo_id=MODEL_REPO,
            filename=MODEL_FILE,
            revision=MODEL_REVISION,
            local_dir=ROOT / "models",
        )
    )


def load_model(
    path: Path | None = None,
    *,
    n_ctx: int = 1024,
    n_batch: int = 256,
    n_gpu_layers: int = -1,
    seed: int = 42,
):
    """Load the quantized model on Apple Silicon using llama.cpp's Metal backend."""
    from llama_cpp import Llama

    model_path = path or ROOT / "models" / MODEL_FILE
    if not model_path.exists():
        raise FileNotFoundError(f"Missing {model_path}; run download_model() first")
    return Llama(
        model_path=str(model_path),
        n_ctx=n_ctx,
        n_batch=n_batch,
        n_gpu_layers=n_gpu_layers,
        seed=seed,
        use_mmap=True,
        verbose=False,
    )


def load_questions(n: int = 8):
    """Read a deterministic first slice; this is a smoke test, not an estimate."""
    if not 5 <= n <= 10:
        raise ValueError("Use 5 to 10 questions for this pilot")
    return load_rows(list(range(n)))


def load_rows(indices: list[int], path: Path | None = None) -> list[dict]:
    """Load specified row indices from a local WMDP-schema Parquet shard."""
    import pandas as pd

    if not indices or len(set(indices)) != len(indices) or min(indices) < 0:
        raise ValueError("indices must be a nonempty list of distinct, nonnegative integers")
    dataset_path = path or download_dataset()
    frame = pd.read_parquet(dataset_path)
    if max(indices) >= len(frame):
        raise IndexError(f"row index {max(indices)} exceeds dataset length {len(frame)}")
    rows = frame.iloc[indices].to_dict("records")
    for index, row in zip(indices, rows):
        if not {"question", "choices", "answer"} <= row.keys() or len(row["choices"]) != 4:
            raise ValueError(f"row {index} does not have the expected WMDP MCQ schema")
        row["choices"] = list(row["choices"])
        row["answer"] = int(row["answer"])
        row["_item_index"] = index
    return rows


def download_dataset() -> Path:
    """Download the pinned WMDP-Bio Parquet shard into the ignored data directory."""
    path = ROOT / "data" / DATASET_FILE
    if path.exists():
        return path
    from huggingface_hub import hf_hub_download

    return Path(
        hf_hub_download(
            repo_id=DATASET_REPO,
            repo_type="dataset",
            filename=DATASET_FILE,
            revision=DATASET_REVISION,
            local_dir=ROOT / "data",
        )
    )


def format_prompt(row: dict, style: str) -> str:
    choices = "\n".join(f"{label}. {choice}" for label, choice in zip(LETTERS, row["choices"]))
    stem = f"Question: {row['question']}\n{choices}"
    if style == "direct":
        return f"{stem}\nAnswer with one letter, A, B, C, or D.\nAnswer:"
    if style == "exam":
        return f"Multiple-choice examination\n\n{stem}\n\nCorrect option:"
    if style == "cloze":
        return f"{stem}\nThe answer is ("
    if style == "few_shot":
        examples = (
            "Question: Which day follows Sunday?\n"
            "A. Tuesday\nB. Monday\nC. Friday\nD. Saturday\nAnswer: (B)\n\n"
            "Question: Which month follows November?\n"
            "A. October\nB. January\nC. December\nD. March\nAnswer: (C)\n\n"
        )
        return f"{examples}{stem}\nAnswer: ("
    raise ValueError(f"Unknown style: {style}")


def parse_letter(text: str) -> str | None:
    """Reject copied option lines; accept only a standalone first-line answer."""
    line = next((part.strip() for part in text.splitlines() if part.strip()), "")
    match = re.fullmatch(r"\(?([ABCD])\)?[.:]?", line, re.I)
    if match:
        return match.group(1).upper()
    match = re.fullmatch(
        r"(?:the\s+)?(?:correct\s+)?(?:answer|option)\s*(?:is|:)?\s*\(?([ABCD])\)?[.:]?",
        line,
        re.I,
    )
    return match.group(1).upper() if match else None


def complete(model, prompt: str, *, max_tokens: int = 12, temperature: float = 0.0) -> str:
    output = model(
        prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        echo=False,
    )
    return output["choices"][0]["text"]


def call_with_trace(model, prompt: str, **kwargs) -> dict:
    started = time.perf_counter()
    response = model(prompt, echo=False, **kwargs)
    return {
        "completion": response["choices"][0]["text"],
        "finish_reason": response["choices"][0].get("finish_reason"),
        "usage": response.get("usage"),
        "elapsed_seconds": round(time.perf_counter() - started, 6),
    }


def evaluate_records(
    model,
    rows: list[dict],
    styles: tuple[str, ...] = ("direct", "exam", "cloze", "few_shot"),
    *,
    constrained: bool = True,
    max_tokens: int = 12,
    temperature: float = 0.0,
) -> list[dict]:
    """Return a complete, per-call record for local analysis and auditing."""
    records = []
    for style in styles:
        for position, row in enumerate(rows):
            prompt = format_prompt(row, style)
            trace = call_with_trace(
                model,
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            letter = parse_letter(trace["completion"])
            records.append(
                {
                    "item_index": row.get("_item_index", position),
                    "format": style,
                    "question": row["question"],
                    "choices": row["choices"],
                    "gold_letter": LETTERS[int(row["answer"])],
                    "prompt": prompt,
                    **trace,
                    "answer_letter": letter,
                    "parseable": letter is not None,
                    "correct": letter == LETTERS[int(row["answer"])] if letter else False,
                }
            )
    if constrained:
        from llama_cpp import LlamaGrammar

        grammar = LlamaGrammar.from_string('root ::= "A" | "B" | "C" | "D"')
        for position, row in enumerate(rows):
            prompt = format_prompt(row, "exam")
            trace = call_with_trace(
                model,
                prompt,
                max_tokens=1,
                temperature=temperature,
                grammar=grammar,
            )
            letter = trace["completion"] if trace["completion"] in LETTERS else None
            records.append(
                {
                    "item_index": row.get("_item_index", position),
                    "format": "exam_constrained_A-D",
                    "question": row["question"],
                    "choices": row["choices"],
                    "gold_letter": LETTERS[int(row["answer"])],
                    "prompt": prompt,
                    **trace,
                    "answer_letter": letter,
                    "parseable": letter is not None,
                    "correct": letter == LETTERS[int(row["answer"])] if letter else False,
                }
            )
    return records


def aggregate_records(records: list[dict]) -> list[dict]:
    """Summarize format compliance first; accuracy is diagnostic only."""
    styles = list(dict.fromkeys(record["format"] for record in records))
    results = []
    for style in styles:
        group = [record for record in records if record["format"] == style]
        parsed = sum(record["parseable"] for record in group)
        correct = sum(record["correct"] for record in group)
        counts = {letter: sum(record["answer_letter"] == letter for record in group) for letter in LETTERS}
        results.append(
            {
                "format": style,
                "questions": len(group),
                "parseable": parsed,
                "parse_rate": round(parsed / len(group), 3),
                "answer_counts": counts,
                "correct_diagnostic": correct,
                "accuracy_diagnostic": round(correct / len(group), 3),
            }
        )
    return results


def run_pilot(model, rows: list[dict], styles: tuple[str, ...] = ("direct", "exam", "cloze", "few_shot")) -> list[dict]:
    return aggregate_records(evaluate_records(model, rows, styles, constrained=False))


def run_constrained_pilot(model, rows: list[dict]) -> dict:
    return aggregate_records(evaluate_records(model, rows, (), constrained=True))[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=8, help="5 to 10 questions")
    parser.add_argument("--download-only", action="store_true")
    args = parser.parse_args()
    model_path = download_model()
    if args.download_only:
        print(model_path)
        return
    rows = load_questions(args.n)
    model = load_model(model_path)
    print(json.dumps(run_pilot(model, rows) + [run_constrained_pilot(model, rows)], indent=2))


if __name__ == "__main__":
    main()
