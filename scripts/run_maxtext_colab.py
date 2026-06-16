#!/usr/bin/env python3
"""Run the ZenteiQ MaxText assignment matrix in Colab.

This script is intentionally thin. It builds MaxText CLI overrides from
configs/runs.json, runs MaxText, and stores unedited stdout/stderr logs.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MATRIX = REPO_ROOT / "configs" / "runs.json"
DEFAULT_LOG_DIR = REPO_ROOT / "results" / "raw_logs"
DEFAULT_COMMAND_LOG = REPO_ROOT / "results" / "metrics" / "run_commands.jsonl"
LOG_TAIL_LINES = 80


def load_matrix(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def cli_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "None"
    if isinstance(value, list):
        return "[" + ",".join(cli_value(item) for item in value) + "]"
    return str(value)


def build_command(matrix: dict[str, Any], run: dict[str, Any], base_output_directory: str | None) -> list[str]:
    model = matrix["models"][run["model_key"]]
    overrides: dict[str, Any] = {}
    overrides.update(matrix["common_overrides"])
    overrides.update(model["overrides"])
    overrides.update(run.get("overrides", {}))

    if base_output_directory:
        overrides["base_output_directory"] = base_output_directory

    overrides["model_name"] = model["maxtext_model_name"]
    overrides["run_name"] = run["run_id"]
    overrides["hardware"] = run["backend"]

    args = [sys.executable, "-m", "maxtext.trainers.pre_train.train"]
    args.extend(f"{key}={cli_value(value)}" for key, value in overrides.items())
    return args


def selected_runs(matrix: dict[str, Any], backend: str | None, run_id: str | None) -> list[dict[str, Any]]:
    runs = matrix["runs"]
    if backend:
        runs = [run for run in runs if run["backend"] == backend]
    if run_id:
        runs = [run for run in runs if run["run_id"] == run_id]
    if not runs:
        raise SystemExit("No runs matched the requested filters.")
    return runs


def run_env(backend: str, platform_env: str) -> dict[str, str]:
    env = os.environ.copy()
    env.setdefault("DECOUPLE_GCLOUD", "TRUE")

    if platform_env == "auto":
        env["JAX_PLATFORMS"] = "cuda" if backend == "gpu" else backend
    elif platform_env != "none":
        env["JAX_PLATFORMS"] = platform_env

    return env


def backend_is_available(backend: str, env: dict[str, str]) -> bool:
    if backend not in {"gpu", "tpu"}:
        return True

    probe = subprocess.run(
        [sys.executable, "-c", "import jax; print({d.platform for d in jax.devices()})"],
        env=env,
        capture_output=True,
        text=True,
    )
    if probe.returncode == 0 and backend in probe.stdout:
        return True

    runtime = "TPU" if backend == "tpu" else "GPU"
    print(
        f"BACKEND={backend!r}, but JAX cannot see a {runtime} device. "
        f"In Colab, use Runtime > Change runtime type > {runtime}, reconnect, then rerun from the first cell.",
        file=sys.stderr,
    )
    if probe.stderr:
        print(probe.stderr.strip().splitlines()[-1], file=sys.stderr)
    return False


def append_command_record(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, sort_keys=True) + "\n")


def run_command(command: list[str], log_path: Path, env: dict[str, str], dry_run: bool) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    printable = " ".join(command)
    print(f"\n=== {log_path.stem} ===")
    print(printable)

    if dry_run:
        return 0

    with log_path.open("w", encoding="utf-8") as log_file:
        log_file.write(printable + "\n\n")
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log_file.write(line)
        return process.wait()


def print_log_tail(log_path: Path) -> None:
    if not log_path.exists():
        return
    lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    print(f"\n--- tail: {log_path} ---", file=sys.stderr)
    for line in lines[-LOG_TAIL_LINES:]:
        print(line, file=sys.stderr)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, default=DEFAULT_MATRIX)
    parser.add_argument("--backend", choices=["cpu", "gpu", "tpu"], help="Run only one backend.")
    parser.add_argument("--run-id", help="Run only one run_id from configs/runs.json.")
    parser.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR)
    parser.add_argument("--command-log", type=Path, default=DEFAULT_COMMAND_LOG)
    parser.add_argument("--base-output-directory", help="Override MaxText base_output_directory.")
    parser.add_argument(
        "--platform-env",
        default="auto",
        help="JAX_PLATFORMS behavior: auto, none, or an explicit value such as cpu/cuda/tpu.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running MaxText.")
    parser.add_argument("--continue-on-error", action="store_true", help="Continue after a failed run.")
    args = parser.parse_args()

    matrix = load_matrix(args.matrix)
    runs = selected_runs(matrix, args.backend, args.run_id)
    started_at = dt.datetime.now(dt.timezone.utc).isoformat()
    failures: list[str] = []

    for run in runs:
        command = build_command(matrix, run, args.base_output_directory)
        log_path = args.log_dir / f"{run['run_id']}.log"
        env = run_env(run["backend"], args.platform_env)
        if not args.dry_run and not backend_is_available(run["backend"], env):
            failures.append(run["run_id"])
            if not args.continue_on_error:
                return 1
            continue

        if not args.dry_run:
            append_command_record(
                args.command_log,
                {
                    "run_id": run["run_id"],
                    "backend": run["backend"],
                    "model_key": run["model_key"],
                    "started_at_utc": started_at,
                    "command": command,
                    "jax_platforms": env.get("JAX_PLATFORMS", ""),
                    "decouple_gcloud": env.get("DECOUPLE_GCLOUD", ""),
                    "dry_run": args.dry_run,
                },
            )

        return_code = run_command(command, log_path, env, args.dry_run)
        if return_code != 0:
            failures.append(run["run_id"])
            print_log_tail(log_path)
            print(f"Run failed with exit code {return_code}: {run['run_id']}", file=sys.stderr)
            if not args.continue_on_error:
                return return_code

    if failures:
        print("Failed runs: " + ", ".join(failures), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
