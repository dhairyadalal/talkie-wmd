import json

import pytest

from talk_wmd.experiment import build_manifest, validate_config, write_run
from talk_wmd.pilot import aggregate_records, evaluate_records
from talk_wmd.performance import summarize_performance


class FakeModel:
    def __call__(self, prompt, **kwargs):
        return {"choices": [{"text": "(B)"}]}


def test_full_trace_and_aggregate() -> None:
    row = {
        "_item_index": 7,
        "question": "Which day follows Sunday?",
        "choices": ["Tuesday", "Monday", "Friday", "Saturday"],
        "answer": 1,
    }
    records = evaluate_records(FakeModel(), [row], ("few_shot",), constrained=False)
    record = records[0]
    assert record["question"] == row["question"]
    assert record["choices"] == row["choices"]
    assert record["prompt"].endswith("Answer: (")
    assert record["completion"] == "(B)"
    assert record["elapsed_seconds"] >= 0
    assert record["usage"] is None
    assert record["gold_letter"] == "B"
    assert record["answer_letter"] == "B"
    assert record["correct"] is True
    assert aggregate_records(records)[0]["parse_rate"] == 1.0


def test_run_files_are_complete(tmp_path) -> None:
    config = {"name": "test", "schema_version": 1}
    manifest = {"run_id": "test-run"}
    record = {"prompt": "Question: test", "completion": "B", "gold_letter": "B"}
    summary = [{"format": "few_shot", "parse_rate": 1.0}]
    run_dir = tmp_path / "run"
    write_run(run_dir, config, manifest, [record], summary)
    assert json.loads((run_dir / "config.json").read_text()) == config
    assert json.loads((run_dir / "manifest.json").read_text()) == manifest
    assert json.loads((run_dir / "traces.jsonl").read_text()) == record


def test_invalid_duplicate_indices() -> None:
    with pytest.raises(ValueError, match="distinct"):
        validate_config({
            "schema_version": 1,
            "item_indices": [0, 0],
            "prompt_styles": ["direct"],
            "constrained_exam": False,
            "generation": {"max_tokens": 12, "temperature": 0.0},
        })


def test_manifest_hashes_package_sources(tmp_path) -> None:
    dataset = tmp_path / "sample.parquet"
    dataset.write_bytes(b"fixture")
    manifest = build_manifest({"name": "test"}, tmp_path / "unused-model.gguf", dataset, "test-run")
    assert "src/talk_wmd/pilot.py" in manifest["code_sha256"]
    assert "src/talk_wmd/experiment.py" in manifest["code_sha256"]
    assert "src/talk_wmd/assets.py" in manifest["code_sha256"]
    assert manifest["dataset"]["sha256"]


def test_completion_throughput_uses_total_call_time() -> None:
    records = [
        {"format": "direct", "usage": {"prompt_tokens": 10, "completion_tokens": 4}, "elapsed_seconds": 1.0},
        {"format": "direct", "usage": {"prompt_tokens": 20, "completion_tokens": 8}, "elapsed_seconds": 3.0},
    ]
    performance = summarize_performance(records)
    assert performance["overall"]["completion_tokens_per_second"] == 3.0
    assert performance["overall"]["prompt_tokens"] == 30
    assert performance["overall"]["calls"] == 2
