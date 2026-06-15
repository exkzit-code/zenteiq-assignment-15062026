# Task 1 - MaxText Data Formats

## Non-Technical Summary

MaxText can train from several data input sources. For this assignment, the actual training runs use synthetic data because the goal is to compare model and hardware behavior, not dataset quality. Synthetic data removes network and preprocessing bottlenecks, so CPU, GPU, and TPU comparisons are cleaner.

For real training, the choice of data format affects repeatability, throughput, shuffle quality, and operational complexity.

## Technical Summary

MaxText's `dataset_type` controls which input pipeline is used.

| `dataset_type` | What it is | Strengths | Tradeoffs |
| --- | --- | --- | --- |
| `synthetic` | Generated token batches | Best for hardware/model benchmarking; removes data loading variance | Does not test real-data preprocessing or learning quality |
| `hf` | Hugging Face datasets pipeline | Easy to start; convenient for public datasets and experiments | Can depend on network/cache behavior; less ideal for large deterministic production runs |
| `grain` | Grain data pipeline | Strong option for large-scale training; supports ArrayRecord and Parquet; ArrayRecord supports global shuffle | More setup work; requires preparing data files correctly |
| `tfds` | TensorFlow Datasets / TFRecord-style pipeline | Stable ecosystem; useful when data already exists in TFDS-compatible form | Less flexible for some LLM workflows; tokenizer/data prep constraints can matter |

## Assignment Choice

For Tasks 2 and 3, use:

```text
dataset_type=synthetic
```

This is explicitly required by the assignment and is also the correct benchmarking choice because it isolates model architecture and backend differences.

## Observations To Mention In Discussion

- Synthetic data makes throughput comparisons cleaner but does not validate model quality.
- Grain is the pipeline to understand for serious large-scale runs because data shuffling, resumability, and throughput matter at scale.
- Hugging Face is good for fast iteration and readability.
- TFDS is useful when data is already in that ecosystem.
- The assignment asks for data-format understanding first because later runs depend on setting `dataset_type` correctly.

