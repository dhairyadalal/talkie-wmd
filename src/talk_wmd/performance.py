"""Compute end-to-end completion-token throughput from a recorded run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def summarize_performance(records: list[dict]) -> dict:
    """Divide generated tokens by model-call wall time, including prompt processing."""
    if not records:
        raise ValueError("Run contains no model calls")

    def metrics(group: list[dict]) -> dict:
        try:
            completion_tokens = sum(record["usage"]["completion_tokens"] for record in group)
            prompt_tokens = sum(record["usage"]["prompt_tokens"] for record in group)
            elapsed_seconds = sum(record["elapsed_seconds"] for record in group)
        except (KeyError, TypeError) as error:
            raise ValueError("Every trace needs token usage and elapsed_seconds") from error
        if elapsed_seconds <= 0:
            raise ValueError("Recorded model-call time must be positive")
        return {
            "calls": len(group),
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "elapsed_seconds": round(elapsed_seconds, 3),
            "completion_tokens_per_second": round(completion_tokens / elapsed_seconds, 3),
        }

    formats = dict.fromkeys(record["format"] for record in records)
    return {
        "measurement": "completion tokens / sum of model-call wall time; includes prompt processing, excludes model loading",
        "overall": metrics(records),
        "by_format": {
            style: metrics([record for record in records if record["format"] == style])
            for style in formats
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path, help="Directory containing traces.jsonl")
    args = parser.parse_args()
    records = [json.loads(line) for line in (args.run_dir / "traces.jsonl").read_text().splitlines()]
    print(json.dumps(summarize_performance(records), indent=2))


if __name__ == "__main__":
    main()
