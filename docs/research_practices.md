# Reproducible research practices in the Talkie pilot

This pilot tests whether a quantized Talkie base model returns parseable answer letters on eight WMDP-Bio questions. Its value as a research example is the audit trail: another researcher can inspect the experimental choices, individual model calls, and calculations behind each reported result. The pilot is not a capability estimate.

## How the practices are implemented

| Practice | Implementation | What it enables |
| --- | --- | --- |
| Specify the experiment before running it | [`experiments/format_pilot.json`](../experiments/format_pilot.json) fixes the question indices, prompt formats, decoding parameters, and seed. | A new condition can be compared against an explicit baseline. |
| Pin software and assets | [`uv.lock`](../uv.lock) fixes Python dependencies. The runner uses pinned model and dataset revisions, verifies the model checksum, and records both asset checksums in each manifest. | A rerun can identify the same inputs without storing large weights or dataset shards in Git. |
| Preserve individual observations | Each run stores its config, manifest, [`traces.jsonl`](../results/20261014T061723Z-ec8a0715/traces.jsonl), and [`summary.json`](../results/20261014T061723Z-ec8a0715/summary.json). A trace contains the prompt, raw completion, answer key, parsed answer, timing, and token usage. | Summary statistics can be checked against the underlying calls, including parsing failures. |
| Keep analysis tied to its measure | Parse rate is the primary outcome. Accuracy is labeled diagnostic. [`talk-wmd-performance`](../src/talk_wmd/performance.py) computes completion tokens divided by recorded model-call time. | Results retain their operational meaning; the 7.01 tokens/s figure is not mistaken for decode-only speed. |
| Separate exploration from recorded runs | The [notebook](../notebooks/talkie_wmdp_poc.ipynb) is exploratory. The [experiment runner](../src/talk_wmd/experiment.py) creates a new run directory and a local MLflow entry. | Ad hoc checks do not silently become reported results. |
| Test the measurement code | [Automated tests](../tests/test_records.py) check trace contents, summary aggregation, manifest hashes, config validation, and throughput arithmetic. | Changes to the pipeline are less likely to alter a reported measure unnoticed. |

## Reproduce and inspect the pilot

From the repository root, follow the [setup instructions](../README.md#setup), then run:

```bash
./.venv/bin/talk-wmd-run
./.venv/bin/talk-wmd-performance results/20261014T061723Z-ec8a0715
./.venv/bin/python -m pytest
```

Use the new run directory printed by `talk-wmd-run` to calculate its throughput. The runner writes full records to `results/`, which can be versioned with the code. MLflow provides a local comparison view in `.tracking/`; that index is not needed to read or recompute the saved results. Each run's manifest records the platform, installed package versions, asset revisions and checksums, Git state, and source-file hashes.

## Limits of this example

The [recorded pilot](../results/20261014T061723Z-ec8a0715/summary.json) uses the first eight WMDP-Bio rows and one fixed option order. Its 8/8 few-shot parse rate is a format check on those items, not an estimate of general accuracy. The model is quantized, and inference may vary across hardware or llama.cpp builds.

That run's manifest reports a dirty Git worktree. The source hashes identify the code used, but its commit alone cannot reconstruct the run. For a shared study, commit the code, configuration, and run records together before drawing comparisons. A larger, preselected evaluation set and varied option order would be needed for claims beyond this smoke test.
