#!/usr/bin/env python3
"""Estimate model parameter counts from the assignment run matrix.

MaxText's own "number parameters" log line is the final source of truth. This
script is only for pre-run planning and for explaining why the chosen configs
fit the assignment.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MATRIX = REPO_ROOT / "configs" / "runs.json"


BUILTIN_MODEL_CONFIGS: dict[str, dict[str, Any]] = {
    "qwen3-0.6b": {
        "base_emb_dim": 1024,
        "base_num_query_heads": 16,
        "base_num_kv_heads": 8,
        "base_mlp_dim": 3072,
        "base_num_decoder_layers": 28,
        "head_dim": 128,
        "vocab_size": 151936,
        "logits_via_embedding": True,
    },
    "deepseek2-16b": {
        "base_emb_dim": 2048,
        "base_num_query_heads": 16,
        "base_num_kv_heads": 16,
        "base_mlp_dim": 10944,
        "base_moe_mlp_dim": 1408,
        "base_num_decoder_layers": 27,
        "first_num_dense_layers": 1,
        "vocab_size": 102400,
        "logits_via_embedding": False,
        "num_experts": 64,
        "num_experts_per_tok": 6,
        "shared_experts": 2,
        "q_lora_rank": 0,
        "kv_lora_rank": 512,
        "qk_nope_head_dim": 128,
        "qk_rope_head_dim": 64,
        "v_head_dim": 128,
    },
}


def load_matrix(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def merged_model_config(model: dict[str, Any]) -> dict[str, Any]:
    base = BUILTIN_MODEL_CONFIGS[model["maxtext_model_name"]].copy()
    base.update(model["overrides"])
    return base


def estimate_dense_qwen(cfg: dict[str, Any]) -> dict[str, int]:
    dim = int(cfg["base_emb_dim"])
    query_heads = int(cfg["base_num_query_heads"])
    kv_heads = int(cfg["base_num_kv_heads"])
    mlp_dim = int(cfg["base_mlp_dim"])
    layers = int(cfg["base_num_decoder_layers"])
    head_dim = int(cfg["head_dim"])
    vocab_size = int(cfg["vocab_size"])

    q_dim = query_heads * head_dim
    kv_dim = kv_heads * head_dim
    embedding = vocab_size * dim
    unembedding = 0 if cfg.get("logits_via_embedding", True) else vocab_size * dim
    attention_per_layer = (dim * q_dim) + (2 * dim * kv_dim) + (q_dim * dim)
    mlp_per_layer = 3 * dim * mlp_dim
    norms = (2 * layers + 1) * dim
    total = embedding + unembedding + layers * (attention_per_layer + mlp_per_layer) + norms
    return {"total_parameters_est": total, "active_parameters_est": total}


def estimate_deepseek_moe(cfg: dict[str, Any]) -> dict[str, int]:
    dim = int(cfg["base_emb_dim"])
    query_heads = int(cfg["base_num_query_heads"])
    kv_heads = int(cfg["base_num_kv_heads"])
    dense_mlp_dim = int(cfg["base_mlp_dim"])
    moe_mlp_dim = int(cfg["base_moe_mlp_dim"])
    layers = int(cfg["base_num_decoder_layers"])
    dense_layers = int(cfg["first_num_dense_layers"])
    moe_layers = max(layers - dense_layers, 0)
    vocab_size = int(cfg["vocab_size"])
    num_experts = int(cfg["num_experts"])
    top_k = int(cfg["num_experts_per_tok"])
    shared_experts = int(cfg["shared_experts"])
    q_lora_rank = int(cfg["q_lora_rank"])
    kv_lora_rank = int(cfg["kv_lora_rank"])
    qk_nope = int(cfg["qk_nope_head_dim"])
    qk_rope = int(cfg["qk_rope_head_dim"])
    v_head_dim = int(cfg["v_head_dim"])

    embedding = vocab_size * dim
    unembedding = 0 if cfg.get("logits_via_embedding", False) else vocab_size * dim

    q_out_dim = query_heads * (qk_nope + qk_rope)
    if q_lora_rank > 0:
        q_projection = (dim * q_lora_rank) + (q_lora_rank * q_out_dim)
    else:
        q_projection = dim * q_out_dim

    kv_out_dim = kv_heads * (qk_nope + qk_rope + v_head_dim)
    if kv_lora_rank > 0:
        kv_projection = (dim * kv_lora_rank) + (kv_lora_rank * kv_out_dim)
    else:
        kv_projection = dim * kv_out_dim

    output_projection = query_heads * v_head_dim * dim
    attention_per_layer = q_projection + kv_projection + output_projection

    dense_mlp_per_layer = 3 * dim * dense_mlp_dim
    expert_mlp = 3 * dim * moe_mlp_dim
    total_moe_mlp_per_layer = (num_experts + shared_experts) * expert_mlp
    active_moe_mlp_per_layer = (top_k + shared_experts) * expert_mlp
    norms = (2 * layers + 1) * dim

    total = (
        embedding
        + unembedding
        + layers * attention_per_layer
        + dense_layers * dense_mlp_per_layer
        + moe_layers * total_moe_mlp_per_layer
        + norms
    )
    active = (
        embedding
        + unembedding
        + layers * attention_per_layer
        + dense_layers * dense_mlp_per_layer
        + moe_layers * active_moe_mlp_per_layer
        + norms
    )
    return {"total_parameters_est": total, "active_parameters_est": active}


def human_count(value: int) -> str:
    if value >= 1_000_000_000:
        return f"{value / 1_000_000_000:.3f}B"
    if value >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"
    return str(value)


def estimate_model(model_key: str, model: dict[str, Any]) -> dict[str, Any]:
    cfg = merged_model_config(model)
    if model["architecture"] == "dense":
        estimate = estimate_dense_qwen(cfg)
    elif model["architecture"] == "moe":
        estimate = estimate_deepseek_moe(cfg)
    else:
        raise ValueError(f"Unsupported architecture: {model['architecture']}")
    return {
        "model_key": model_key,
        "display_name": model["display_name"],
        "architecture": model["architecture"],
        **estimate,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, default=DEFAULT_MATRIX)
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of a table.")
    args = parser.parse_args()

    matrix = load_matrix(args.matrix)
    rows = [estimate_model(key, model) for key, model in matrix["models"].items()]

    if args.json:
        print(json.dumps(rows, indent=2, sort_keys=True))
        return 0

    print("| Model | Architecture | Estimated total | Estimated active |")
    print("| --- | --- | ---: | ---: |")
    for row in rows:
        print(
            "| {display_name} | {architecture} | {total} | {active} |".format(
                display_name=row["display_name"],
                architecture=row["architecture"],
                total=human_count(row["total_parameters_est"]),
                active=human_count(row["active_parameters_est"]),
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

