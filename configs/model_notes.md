# Model Configuration Notes

## Dense Qwen Base

The base dense run uses MaxText's built-in `qwen3-0.6b` model config unchanged.

Important built-in dimensions:

| Field | Value |
| --- | ---: |
| `base_emb_dim` | 1024 |
| `base_num_query_heads` | 16 |
| `base_num_kv_heads` | 8 |
| `base_mlp_dim` | 3072 |
| `base_num_decoder_layers` | 28 |
| `head_dim` | 128 |
| `vocab_size` | 151936 |

## Dense Qwen Scaled Target

The scaled dense run keeps the Qwen3 architecture and vocabulary but increases the major size drivers:

| Field | Base | Scaled |
| --- | ---: | ---: |
| `base_emb_dim` | 1024 | 1280 |
| `base_num_query_heads` | 16 | 20 |
| `base_num_kv_heads` | 8 | 10 |
| `base_mlp_dim` | 3072 | 3840 |
| `base_num_decoder_layers` | 28 | 32 |
| `head_dim` | 128 | 128 |

Rationale:

- Keep `head_dim=128`, matching the base Qwen config.
- Scale query and KV heads with hidden size so attention projection shapes remain coherent.
- Increase MLP dimension proportionally with hidden size.
- Add a small number of layers to land near the requested 1B target without jumping to MaxText's built-in 1.7B config.

## DeepSeek MoE Under 1B

The MoE run starts from `deepseek2-16b` and uses `override_model_config=true` for a controlled scale-down.

Important changes:

| Field | Built-in 16B | Scaled-down |
| --- | ---: | ---: |
| `base_emb_dim` | 2048 | 1024 |
| `base_num_query_heads` | 16 | 8 |
| `base_num_kv_heads` | 16 | 8 |
| `base_mlp_dim` | 10944 | 4096 |
| `base_moe_mlp_dim` | 1408 | 768 |
| `base_num_decoder_layers` | 27 | 12 |
| `first_num_dense_layers` | 1 | 2 |
| `num_experts` | 64 | 8 |
| `num_experts_per_tok` | 6 | 2 |
| `shared_experts` | 2 | 1 |

Rationale:

- Reduce total parameters by cutting expert count, expert width, hidden size, and layer count.
- Keep MoE behavior visible with routed experts, top-k routing, and one shared expert.
- Use `num_experts_per_tok=2` so active parameters are materially lower than total parameters.
- Keep two dense layers before MoE layers to make dense-vs-MoE comparisons easier to explain.

The exact parameter count from MaxText logs is the source of truth. The estimator script is only a pre-run planning aid.

