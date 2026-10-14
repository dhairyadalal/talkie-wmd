# Recorded pilot runs

Each directory contains one experiment run. Its manifest records the code and environment used to produce it. Generate a new run after changing code or configuration.

| Run | Notes |
| --- | --- |
| `20261014T060537Z-b996e2ec` | Initial instrumentation: prompts, completions, and answer diagnostics. |
| `20261014T060735Z-c1dfd22a` | Adds seed 42, per-call timing, token usage, and finish reason. |
| `20261014T061723Z-ec8a0715` | Metal-enabled rerun after the package reorganization; 7.01 end-to-end completion tokens/s. |

All runs sampled WMDP-Bio rows 0 through 7. Use the latest run as the baseline for new format experiments.
