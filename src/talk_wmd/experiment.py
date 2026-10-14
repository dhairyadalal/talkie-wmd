"""Run a configured pilot and record complete experiment results."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

from talk_wmd.assets import MODEL_SHA256, verify_sha256
from talk_wmd.pilot import (
    DATASET_FILE,
    DATASET_REPO,
    DATASET_REVISION,
    MODEL_FILE,
    MODEL_REPO,
    MODEL_REVISION,
    ROOT,
    aggregate_records,
    evaluate_records,
    load_model,
    load_rows,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_value(*args: str) -> str | None:
    result = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else None


def validate_config(config: dict) -> None:
    if config.get("schema_version") != 1:
        raise ValueError("Unsupported experiment config schema_version")
    indices = config.get("item_indices")
    if not isinstance(indices, list) or not indices or any(type(i) is not int or i < 0 for i in indices):
        raise ValueError("item_indices must be a nonempty list of nonnegative integers")
    if len(indices) != len(set(indices)):
        raise ValueError("item_indices must be distinct")
    styles = config.get("prompt_styles")
    if not isinstance(styles, list) or not styles or not set(styles) <= {"direct", "exam", "cloze", "few_shot"}:
        raise ValueError("prompt_styles must contain supported prompt formats")
    if len(styles) != len(set(styles)):
        raise ValueError("prompt_styles must be distinct")
    if type(config.get("constrained_exam")) is not bool:
        raise ValueError("constrained_exam must be a boolean")
    generation = config.get("generation", {})
    if not isinstance(generation.get("max_tokens"), int) or generation["max_tokens"] < 1:
        raise ValueError("generation.max_tokens must be a positive integer")
    if generation.get("temperature") != 0.0:
        raise ValueError("This deterministic pilot requires temperature 0.0")


def build_manifest(config: dict, model_path: Path, dataset_path: Path, run_id: str) -> dict:
    packages = ("llama-cpp-python", "huggingface-hub", "pandas", "pyarrow", "mlflow")
    return {
        "run_id": run_id,
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "experiment": config["name"],
        "description": config.get("description", ""),
        "git_commit": git_value("rev-parse", "HEAD"),
        "git_dirty": bool(git_value("status", "--porcelain")),
        "python": sys.version,
        "platform": platform.platform(),
        "packages": {name: importlib.metadata.version(name) for name in packages},
        "code_sha256": {
            str(path.relative_to(ROOT)): sha256(path)
            for path in sorted((ROOT / "src" / "talk_wmd").glob("*.py"))
        },
        "model": {
            "repo": MODEL_REPO,
            "revision": MODEL_REVISION,
            "file": MODEL_FILE,
            "sha256": MODEL_SHA256,
            "quantization": "Q4_K_M",
        },
        "dataset": {
            "repo": DATASET_REPO,
            "revision": DATASET_REVISION,
            "file": DATASET_FILE,
            "sha256": sha256(dataset_path),
        },
        "config": config,
    }


def write_run(run_dir: Path, config: dict, manifest: dict, records: list[dict], summary: list[dict]) -> None:
    payloads = {
        "config.json": json.dumps(config, indent=2, ensure_ascii=False) + "\n",
        "manifest.json": json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        "summary.json": json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        "traces.jsonl": "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
    }
    run_dir.mkdir(parents=True, exist_ok=False)
    for name, content in payloads.items():
        (run_dir / name).write_text(content)


def index_with_mlflow(run_dir: Path, config: dict, summary: list[dict]) -> str:
    """Add a local comparison index; the files in results/ remain canonical."""
    import mlflow

    tracking = ROOT / ".tracking"
    tracking.mkdir(exist_ok=True)
    mlflow.set_tracking_uri(f"sqlite:///{tracking / 'mlflow.db'}")
    name = config["name"]
    experiment = mlflow.get_experiment_by_name(name)
    if experiment is None:
        experiment_id = mlflow.create_experiment(name, artifact_location=(tracking / "artifacts").as_uri())
    else:
        experiment_id = experiment.experiment_id
    with mlflow.start_run(experiment_id=experiment_id, run_name=run_dir.name) as run:
        mlflow.log_params({
            "config_sha256": sha256(run_dir / "config.json"),
            "model_revision": MODEL_REVISION,
            "dataset_revision": DATASET_REVISION,
            "item_count": len(config["item_indices"]),
        })
        for entry in summary:
            prefix = entry["format"]
            mlflow.log_metric(f"{prefix}.parse_rate", entry["parse_rate"])
            mlflow.log_metric(f"{prefix}.accuracy_diagnostic", entry["accuracy_diagnostic"])
        for path in run_dir.iterdir():
            mlflow.log_artifact(str(path), artifact_path="run_files")
        return run.info.run_id


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "experiments" / "format_pilot.json")
    parser.add_argument("--no-mlflow", action="store_true", help="Keep run files but skip the local comparison index")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    validate_config(config)
    model_path = ROOT / "models" / MODEL_FILE
    dataset_path = ROOT / "data" / DATASET_FILE
    if not model_path.exists() or not dataset_path.exists():
        raise FileNotFoundError("Assets missing. Run talk-wmd-download first.")
    verify_sha256(model_path, MODEL_SHA256)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    rows = load_rows(config["item_indices"], dataset_path)
    model = load_model(model_path, **config.get("model", {}))
    records = evaluate_records(
        model,
        rows,
        tuple(config["prompt_styles"]),
        constrained=config["constrained_exam"],
        **config["generation"],
    )
    summary = aggregate_records(records)
    run_dir = ROOT / "results" / run_id
    manifest = build_manifest(config, model_path, dataset_path, run_id)
    write_run(run_dir, config, manifest, records, summary)
    mlflow_id = None if args.no_mlflow else index_with_mlflow(run_dir, config, summary)
    print(json.dumps({"run_dir": str(run_dir), "mlflow_run_id": mlflow_id, "summary": summary}, indent=2))


if __name__ == "__main__":
    main()
