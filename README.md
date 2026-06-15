# ZenteiQ MaxText Assignment

This repository is a reproducible submission workspace for the ZenteiQ ML Engineer assignment.

The assignment asks for MaxText training runs across CPU, GPU, and TPU using:

- Dense Qwen: `qwen3-0.6b` and a scaled ~1B variant
- DeepSeek MoE: a scaled-down variant under 1B total parameters
- Synthetic data for all training runs
- Complete MaxText logs and all reported metrics

## Principles

- Do not modify MaxText source code.
- Keep every model change in explicit config files.
- Keep raw logs, derived tables, and written interpretation separate.
- Make commits small enough that reviewers can audit the work.

## Layout

```text
configs/              Run matrix and model-shape overrides
docs/                 Written analysis and discussion prep
notebooks/            Colab entrypoint
results/raw_logs/     Raw stdout/stderr logs from MaxText runs
results/metrics/      Parsed CSV/Markdown metric tables
results/screenshots/  Optional captioned screenshots from Colab
scripts/              Minimal setup, run, parse, and estimate helpers
```

## Expected Run Matrix

There are 9 required training runs:

| Architecture | Model | CPU | GPU | TPU |
| --- | --- | --- | --- | --- |
| Dense | Qwen 0.6B base | required | required | required |
| Dense | Qwen scaled ~1B | required | required | required |
| MoE | DeepSeek under 1B | required | required | required |

Each run uses `steps=50` and `dataset_type=synthetic`.

## Evidence Standard

For each run, keep:

- Raw MaxText log in `results/raw_logs/`
- Parsed per-step metrics in `results/metrics/`
- Final comparison tables in `docs/final_analysis.md`
- Notes about failures, retries, and hardware/runtime constraints

The raw logs are the source of truth. Tables are derived from them.

