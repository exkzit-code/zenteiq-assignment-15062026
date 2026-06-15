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

## Local Planning Checks

Before running Colab, check the planned parameter sizes:

```bash
python3 scripts/estimate_params.py
```

Expected planning estimates:

| Model | Architecture | Estimated total | Estimated active |
| --- | --- | ---: | ---: |
| Qwen3 0.6B base | dense | ~596M | ~596M |
| Qwen3 scaled approximately 1B | dense | ~981M | ~981M |
| DeepSeek MoE scaled under 1B | moe | ~478M | ~336M |

MaxText's logged parameter count remains the final source of truth.

## Colab Execution

Use a fresh Colab runtime for each backend because CPU, GPU, and TPU installs can require different dependencies.

Recommended path:

1. Create a public GitHub repo.
2. Push this repository to it.
3. In Colab, choose **File > Open notebook > GitHub**.
4. Paste the GitHub repo URL and open `notebooks/zenteiq_maxtext_runner.ipynb`.
5. In the first notebook cell, set:
   - `BACKEND = "cpu"`, `"gpu"`, or `"tpu"`
   - `REPO_URL = "https://github.com/<your-user>/<your-repo>.git"`
6. Match the Colab runtime type to `BACKEND`.
7. Run all cells.
8. Repeat in fresh Colab runtimes for the other two backends.
9. Commit the generated logs and tables back to the repo.

You can upload only the notebook to Colab, but the notebook still needs the rest of this repo. Set `REPO_URL` in the first cell so it can clone the scripts and configs.

Equivalent terminal commands inside Colab:

```bash
bash scripts/install_maxtext_colab.sh gpu
python3 scripts/run_maxtext_colab.py --backend gpu
python3 scripts/parse_maxtext_logs.py
```

Repeat with `cpu` and `tpu`.

## After All Runs

Update `docs/final_analysis.md` using:

- `results/metrics/run_summary.md`
- `results/metrics/run_summary.csv`
- `results/metrics/step_metrics.csv`
- raw logs under `results/raw_logs/`

Then commit the completed results and analysis.
