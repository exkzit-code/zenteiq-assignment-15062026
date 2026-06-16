#!/usr/bin/env python3
"""Parse MaxText raw logs into CSV and Markdown metric tables."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import re
import statistics
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MATRIX = REPO_ROOT / "configs" / "runs.json"
DEFAULT_LOG_DIR = REPO_ROOT / "results" / "raw_logs"
DEFAULT_METRICS_DIR = REPO_ROOT / "results" / "metrics"

STEP_RE = re.compile(r"completed step:\s*(?P<step>\d+),\s*(?P<metrics>.*)", re.IGNORECASE)
METRIC_RE = re.compile(r"(?P<key>[^:,]+):\s*(?P<value>[^,]+)")
PARAM_RE = re.compile(r"number parameters:\s*(?P<value>[-+0-9.eE]+)\s*(?P<unit>billion|million|trillion)?", re.I)
MEMORY_RE = re.compile(
    r"Total memory size:\s*(?P<total_memory_gb>[-+0-9.eE]+)\s*GB,\s*"
    r"Output size:\s*(?P<output_memory_gb>[-+0-9.eE]+)\s*GB,\s*"
    r"Temp size:\s*(?P<temp_memory_gb>[-+0-9.eE]+)\s*GB,\s*"
    r"Argument size:\s*(?P<argument_memory_gb>[-+0-9.eE]+)\s*GB,\s*"
    r"Host temp size:\s*(?P<host_temp_memory_gb>[-+0-9.eE]+)\s*GB",
    re.I,
)
CONFIG_RE = re.compile(r"Config param (?P<key>[^:]+):\s*(?P<value>.*)")
SYSTEM_RE = re.compile(r"System Information:\s*(?P<key>[^:]+):\s*(?P<value>.*)")
NUM_DEVICES_RE = re.compile(r"Num_devices:\s*(?P<num_devices>\d+),\s*shape\s*(?P<device_shape>.*)")
STEP_METADATA_FIELDS = {"run_id", "backend", "architecture", "model_key", "display_name", "step"}
METRIC_KEY_OVERRIDES = {
    "TFLOP/s/device": "tflops_per_device",
    "Tokens/s/device": "tokens_per_device",
}


def load_matrix(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def matrix_lookup(matrix: dict[str, Any]) -> dict[str, dict[str, str]]:
    models = matrix["models"]
    lookup = {}
    for run in matrix["runs"]:
        model = models[run["model_key"]]
        lookup[run["run_id"]] = {
            "backend": run["backend"],
            "model_key": run["model_key"],
            "architecture": model["architecture"],
            "display_name": model["display_name"],
        }
    return lookup


def parse_number(value: str) -> float:
    return float(value)


def parse_metric_value(value: str) -> Any:
    text = value.strip()
    try:
        return float(text)
    except ValueError:
        return text


def normalize_metric_key(key: str) -> str:
    text = key.strip()
    if text in METRIC_KEY_OVERRIDES:
        return METRIC_KEY_OVERRIDES[text]
    return re.sub(r"[^0-9a-zA-Z]+", "_", text).strip("_").lower()


def parse_step_metrics(metrics: str) -> dict[str, Any]:
    return {
        normalize_metric_key(match.group("key")): parse_metric_value(match.group("value"))
        for match in METRIC_RE.finditer(metrics)
    }


def is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not math.isnan(value)


def parse_param_count(match: re.Match[str]) -> float:
    value = float(match.group("value"))
    unit = (match.group("unit") or "").lower()
    if unit == "trillion":
        return value * 1_000_000_000_000
    if unit == "billion":
        return value * 1_000_000_000
    if unit == "million":
        return value * 1_000_000
    return value


def parse_log(path: Path, metadata: dict[str, str]) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, str]]:
    text = path.read_text(encoding="utf-8", errors="replace")
    run_id = path.stem
    steps: list[dict[str, Any]] = []
    config: dict[str, str] = {}
    summary: dict[str, Any] = {
        "run_id": run_id,
        **metadata,
        "log_file": str(path),
        "status": "no_steps",
        "total_parameters_reported": "",
        "jax_version": "",
        "jaxlib_version": "",
        "jax_backend": "",
        "num_devices": "",
        "device_shape": "",
        "total_memory_gb": "",
        "output_memory_gb": "",
        "temp_memory_gb": "",
        "argument_memory_gb": "",
        "host_temp_memory_gb": "",
    }

    for line in text.splitlines():
        step_match = STEP_RE.search(line)
        if step_match:
            row: dict[str, Any] = {
                "run_id": run_id,
                **metadata,
                "step": int(step_match.group("step")),
            }
            row.update(parse_step_metrics(step_match.group("metrics")))
            steps.append(row)
            continue

        param_match = PARAM_RE.search(line)
        if param_match:
            summary["total_parameters_reported"] = int(parse_param_count(param_match))
            continue

        memory_match = MEMORY_RE.search(line)
        if memory_match:
            for key, value in memory_match.groupdict().items():
                summary[key] = parse_number(value)
            continue

        config_match = CONFIG_RE.search(line)
        if config_match:
            config[config_match.group("key")] = config_match.group("value")
            continue

        system_match = SYSTEM_RE.search(line)
        if system_match:
            key = system_match.group("key").strip().lower().replace(" ", "_")
            value = system_match.group("value").strip()
            if key == "jax_version":
                summary["jax_version"] = value
            elif key == "jaxlib_version":
                summary["jaxlib_version"] = value
            elif key == "jax_backend":
                summary["jax_backend"] = value
            continue

        device_match = NUM_DEVICES_RE.search(line)
        if device_match:
            summary["num_devices"] = int(device_match.group("num_devices"))
            summary["device_shape"] = device_match.group("device_shape")

    if steps:
        steps_sorted = sorted(steps, key=lambda row: row["step"])
        last = steps_sorted[-1]
        steady = steps_sorted[-10:] if len(steps_sorted) >= 10 else steps_sorted
        step_metric_keys = sorted({key for row in steps_sorted for key in row} - STEP_METADATA_FIELDS)
        summary.update(
            {
                "status": "complete" if len(steps_sorted) >= 50 else "partial",
                "steps_completed": len(steps_sorted),
                "first_step": steps_sorted[0]["step"],
                "last_step": last["step"],
            }
        )
        for key in step_metric_keys:
            values = [row[key] for row in steps_sorted if is_number(row.get(key))]
            steady_values = [row[key] for row in steady if is_number(row.get(key))]
            summary[f"last_{key}"] = last.get(key, "")
            if values:
                summary[f"min_{key}"] = min(values)
                summary[f"max_{key}"] = max(values)
            if steady_values:
                summary[f"steady_avg_{key}"] = statistics.fmean(steady_values)
    else:
        summary.update(
            {
                "steps_completed": 0,
                "first_step": "",
                "last_step": "",
                "last_seconds": "",
                "last_tflops_per_device": "",
                "last_tokens_per_device": "",
                "last_total_weights": "",
                "last_loss": "",
                "min_loss": "",
                "max_loss": "",
                "steady_avg_seconds": "",
                "steady_avg_tflops_per_device": "",
                "steady_avg_tokens_per_device": "",
            }
        )

    return steps, summary, config


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    lines = []
    lines.append("| " + " | ".join(columns) + " |")
    lines.append("| " + " | ".join("---" for _ in columns) + " |")
    for row in rows:
        values = [str(row.get(column, "")) for column in columns]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines) + "\n"


def ordered_fields(rows: list[dict[str, Any]], preferred: list[str]) -> list[str]:
    seen = {field for row in rows for field in row}
    return preferred + sorted(seen - set(preferred))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, default=DEFAULT_MATRIX)
    parser.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR)
    parser.add_argument("--metrics-dir", type=Path, default=DEFAULT_METRICS_DIR)
    args = parser.parse_args()

    matrix = load_matrix(args.matrix)
    lookup = matrix_lookup(matrix)
    log_files = sorted(args.log_dir.glob("*.log"))
    if not log_files:
        raise SystemExit(f"No .log files found in {args.log_dir}")

    all_steps: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    configs: dict[str, dict[str, str]] = {}

    for log_file in log_files:
        metadata = lookup.get(
            log_file.stem,
            {"backend": "", "model_key": "", "architecture": "", "display_name": ""},
        )
        steps, summary, config = parse_log(log_file, metadata)
        all_steps.extend(steps)
        summaries.append(summary)
        configs[log_file.stem] = config

    preferred_step_fields = [
        "run_id",
        "backend",
        "architecture",
        "model_key",
        "display_name",
        "step",
        "seconds",
        "tflops_per_device",
        "tokens_per_device",
        "total_weights",
        "loss",
        "lm_loss",
        "perplexity",
        "moe_lb_loss",
    ]
    preferred_summary_fields = [
        "run_id",
        "backend",
        "architecture",
        "model_key",
        "display_name",
        "status",
        "steps_completed",
        "first_step",
        "last_step",
        "last_seconds",
        "last_tflops_per_device",
        "last_tokens_per_device",
        "last_total_weights",
        "last_loss",
        "last_lm_loss",
        "last_perplexity",
        "last_moe_lb_loss",
        "min_loss",
        "max_loss",
        "steady_avg_seconds",
        "steady_avg_tflops_per_device",
        "steady_avg_tokens_per_device",
        "total_parameters_reported",
        "jax_version",
        "jaxlib_version",
        "jax_backend",
        "num_devices",
        "device_shape",
        "total_memory_gb",
        "output_memory_gb",
        "temp_memory_gb",
        "argument_memory_gb",
        "host_temp_memory_gb",
        "log_file",
    ]
    step_fields = ordered_fields(all_steps, preferred_step_fields)
    summary_fields = ordered_fields(summaries, preferred_summary_fields)

    args.metrics_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.metrics_dir / "step_metrics.csv", all_steps, step_fields)
    write_csv(args.metrics_dir / "run_summary.csv", summaries, summary_fields)
    (args.metrics_dir / "config_params.json").write_text(json.dumps(configs, indent=2, sort_keys=True), encoding="utf-8")
    markdown_columns = [
        "run_id",
        "backend",
        "architecture",
        "status",
        "steps_completed",
        "last_seconds",
        "last_tflops_per_device",
        "last_tokens_per_device",
        "last_loss",
        "last_lm_loss",
        "last_perplexity",
        "last_moe_lb_loss",
        "total_parameters_reported",
    ]
    (args.metrics_dir / "run_summary.md").write_text(
        markdown_table(
            summaries,
            [column for column in markdown_columns if any(row.get(column, "") != "" for row in summaries)],
        ),
        encoding="utf-8",
    )
    print(f"Wrote {len(all_steps)} step rows and {len(summaries)} run summaries to {args.metrics_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
