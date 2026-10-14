"""Small local Talkie/WMDP-Bio format pilot. No question text is saved."""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parent
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
    # The default Xet transfer stalled on this Mac; allow opt-in override.
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


def load_model(path: Path | None = None):
    """Load the quantized model on Apple Silicon using llama.cpp's Metal backend."""
    from llama_cpp import Llama

    model_path = path or ROOT / "models" / MODEL_FILE
    if not model_path.exists():
        raise FileNotFoundError(f"Missing {model_path}; run download_model() first")
    return Llama(
        model_path=str(model_path),
        n_ctx=1024,
        n_batch=256,
        n_gpu_layers=-1,
        use_mmap=True,
        verbose=False,
    )


def load_questions(n: int = 8):
    """Read a deterministic first slice; this is a smoke test, not an estimate."""
    if not 5 <= n <= 10:
        raise ValueError("Use 5 to 10 questions for this pilot")
    import pandas as pd

    path = download_dataset()
    return pd.read_parquet(path).head(n).to_dict("records")


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


def complete(model, prompt: str) -> str:
    output = model(
        prompt,
        max_tokens=12,
        temperature=0.0,
        echo=False,
    )
    return output["choices"][0]["text"]


def run_pilot(model, rows: list[dict], styles: tuple[str, ...] = ("direct", "exam", "cloze", "few_shot")) -> list[dict]:
    """Compare output parseability and descriptive accuracy; return aggregates only."""
    results = []
    for style in styles:
        parsed = correct = 0
        answer_counts = {letter: 0 for letter in LETTERS}
        for row in rows:
            letter = parse_letter(complete(model, format_prompt(row, style)))
            if letter is not None:
                parsed += 1
                answer_counts[letter] += 1
                correct += int(LETTERS[int(row["answer"])] == letter)
        results.append(
            {
                "format": style,
                "questions": len(rows),
                "parseable": parsed,
                "correct": correct,
                "parse_rate": round(parsed / len(rows), 3),
                "accuracy_all": round(correct / len(rows), 3),
                "answer_counts": answer_counts,
            }
        )
    return results


def run_constrained_pilot(model, rows: list[dict]) -> dict:
    """Force exactly one A–D token; this fixes formatting, not answer quality."""
    from llama_cpp import LlamaGrammar

    grammar = LlamaGrammar.from_string('root ::= "A" | "B" | "C" | "D"')
    correct = 0
    answer_counts = {letter: 0 for letter in LETTERS}
    for row in rows:
        output = model(
            format_prompt(row, "exam"),
            max_tokens=1,
            temperature=0.0,
            grammar=grammar,
            echo=False,
        )["choices"][0]["text"]
        answer_counts[output] += 1
        correct += int(output == LETTERS[int(row["answer"])])
    return {
        "format": "exam_constrained_A-D",
        "questions": len(rows),
        "parseable": len(rows),
        "correct": correct,
        "parse_rate": 1.0,
        "accuracy_all": round(correct / len(rows), 3),
        "answer_counts": answer_counts,
    }


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
