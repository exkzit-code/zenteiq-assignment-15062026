# Final Analysis

Status: complete. CPU, GPU, and TPU each ran the same three MaxText jobs for 50 steps on synthetic data.

## Run Matrix

| Architecture | Model | CPU log | GPU log | TPU log |
| --- | --- | --- | --- | --- |
| Dense | Qwen3 0.6B base | `results/raw_logs/dense_qwen_0_6b_cpu.log` | `results/raw_logs/dense_qwen_0_6b_gpu.log` | `results/raw_logs/dense_qwen_0_6b_tpu.log` |
| Dense | Qwen3 scaled ~1B | `results/raw_logs/dense_qwen_scaled_1b_cpu.log` | `results/raw_logs/dense_qwen_scaled_1b_gpu.log` | `results/raw_logs/dense_qwen_scaled_1b_tpu.log` |
| MoE | DeepSeek MoE under 1B | `results/raw_logs/moe_deepseek_under_1b_cpu.log` | `results/raw_logs/moe_deepseek_under_1b_gpu.log` | `results/raw_logs/moe_deepseek_under_1b_tpu.log` |

## Metric Summary

Source: `results/metrics/run_summary.md`

| run_id | backend | architecture | status | steps_completed | last_seconds | last_tflops_per_device | last_tokens_per_device | last_loss | last_lm_loss | last_perplexity | last_moe_lb_loss | total_parameters_reported |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| dense_qwen_0_6b_cpu | cpu | dense | complete | 50 | 7.464 | 0.015 | 4.287 | 0.0 | 0.0 | 1.0 |  | 596000000 |
| dense_qwen_0_6b_gpu | gpu | dense | complete | 50 | 0.023 | 4.952 | 1380.56 | 0.0 | 0.0 | 1.0 |  | 596000000 |
| dense_qwen_0_6b_tpu | tpu | dense | complete | 50 | 0.014 | 8.3 | 2313.978 | 0.0 | 0.0 | 1.0 |  | 596000000 |
| dense_qwen_scaled_1b_cpu | cpu | dense | complete | 50 | 11.801 | 0.016 | 2.712 | 0.0 | 0.0 | 1.0 |  | 981000000 |
| dense_qwen_scaled_1b_gpu | gpu | dense | complete | 50 | 0.033 | 5.728 | 970.638 | 0.0 | 0.0 | 1.0 |  | 981000000 |
| dense_qwen_scaled_1b_tpu | tpu | dense | complete | 50 | 0.025 | 7.619 | 1291.051 | 0.0 | 0.0 | 1.0 |  | 981000000 |
| moe_deepseek_under_1b_cpu | cpu | moe | complete | 50 | 4.672 | 0.01 | 6.849 | 0.264 | 0.264 | 1.303 | 0.0 | 478000000 |
| moe_deepseek_under_1b_gpu | gpu | moe | complete | 50 | 0.015 | 2.969 | 2137.466 | 0.262 | 0.262 | 1.299 | 0.0 | 478000000 |
| moe_deepseek_under_1b_tpu | tpu | moe | complete | 50 | 0.01 | 4.328 | 3116.175 | 0.268 | 0.268 | 1.307 | 0.0 | 478000000 |

## Backend Comparison

| Model | GPU tokens/sec vs CPU | TPU tokens/sec vs CPU | TPU tokens/sec vs GPU | CPU/GPU/TPU last seconds |
| --- | ---: | ---: | ---: | --- |
| Qwen3 0.6B | 322.0x | 539.8x | 1.68x | 7.464 / 0.023 / 0.014 |
| Qwen3 scaled ~1B | 357.9x | 476.1x | 1.33x | 11.801 / 0.033 / 0.025 |
| DeepSeek MoE under 1B | 312.1x | 455.0x | 1.46x | 4.672 / 0.015 / 0.010 |

The hardware ordering is consistent: CPU is slowest, GPU is much faster, and TPU is fastest on these Colab runs. The TPU advantage over GPU is largest for the smaller dense Qwen run and still present for the scaled dense and MoE runs.

Memory also scales as expected. Qwen3 0.6B used 7.2 GB on CPU, 5.0 GB on GPU, and 4.5 GB on TPU. The scaled Qwen run increased to 11.7 GB on CPU, 8.5 GB on GPU, and 8.4 GB on TPU. The MoE run used 5.2 GB on CPU and 3.7 GB on both GPU and TPU.

## Model Comparison

The scaled Qwen configuration increased the reported parameter count from 596M to 981M by widening the embedding dimension, increasing attention heads, increasing MLP width, and adding decoder layers. That produced the expected slowdown across all backends: CPU step time rose from 7.464s to 11.801s, GPU from 0.023s to 0.033s, and TPU from 0.014s to 0.025s.

The DeepSeek MoE run reports fewer total parameters, 478M, but it is not directly comparable to dense by parameter count alone because MoE activates only selected experts per token. At this small sequence length and batch size, MoE is very fast on accelerators but still carries routing behavior and MoE-specific metrics. Its loss is nonzero across all backends, with last perplexity around 1.30.

## Unexpected Findings

- Dense Qwen loss was logged as 0.0 and perplexity as 1.0 on all backends. Because this is consistent across CPU, GPU, and TPU, it does not affect the hardware throughput comparison, but it should not be interpreted as meaningful model convergence.
- GPU setup initially failed because Colab's cuDNN stack conflicted with optional Transformer Engine. The runner now removes Transformer Engine for GPU Colab runs.
- TPU setup required a real TPU runtime; selecting `BACKEND = "tpu"` alone was not enough.
- TPU MaxText initially aborted with a TensorFlow/JAX LLVM option collision. For synthetic data, TensorFlow is unnecessary, so the setup patches the installed MaxText 0.2.2 package to avoid TensorFlow imports on the TPU synthetic path.

## Evidence Files

- Raw logs: `results/raw_logs/`
- Step metrics: `results/metrics/step_metrics.csv`
- Run summaries: `results/metrics/run_summary.csv` and `results/metrics/run_summary.md`
- Config snapshots: `results/metrics/config_params.json`
- Commands: `results/metrics/run_commands.jsonl`
