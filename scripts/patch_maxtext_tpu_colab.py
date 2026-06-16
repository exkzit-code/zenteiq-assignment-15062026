#!/usr/bin/env python3
"""Patch installed MaxText for Colab TPU synthetic runs.

MaxText 0.2.2 imports TensorFlow on the pre-train path even for synthetic data.
On Colab TPU, TensorFlow and JAX can both register the same LLVM options and
abort before training starts. This keeps the synthetic path on JAX only.
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path


def maxtext_dir() -> Path:
    override = os.environ.get("MAXTEXT_PACKAGE_DIR")
    if override:
        return Path(override)

    spec = importlib.util.find_spec("maxtext")
    if not spec or not spec.submodule_search_locations:
        raise SystemExit("Could not locate installed maxtext package")
    return Path(next(iter(spec.submodule_search_locations)))


def replace_once(path: Path, old: str, new: str) -> None:
    text = path.read_text(encoding="utf-8")
    if new and new in text:
        return
    if old in text:
        path.write_text(text.replace(old, new, 1), encoding="utf-8")
        return
    if not new:
        return
    raise SystemExit(f"Expected patch target not found in {path}")


def main() -> None:
    root = maxtext_dir()

    tf_stub = """# ZenteiQ Colab TPU synthetic patch: avoid TensorFlow/JAX LLVM collision.
class _TensorFlowStub:
  class config:
    @staticmethod
    def set_visible_devices(*_args, **_kwargs):
      return None


tf = _TensorFlowStub()
"""

    replace_once(
        root / "trainers" / "pre_train" / "train.py",
        "import tensorflow as tf\n\nimport jax\n",
        f"{tf_stub}\nimport jax\n",
    )

    replace_once(
        root / "input_pipeline" / "multihost_dataloading.py",
        "import tensorflow as tf  # pylint: disable=g-import-not-at-top\n\nimport numpy as np\n",
        """# ZenteiQ Colab TPU synthetic patch: only real-data pipelines need TensorFlow.
class _TensorFlowStub:
  class data:
    class Dataset:
      pass


tf = _TensorFlowStub()

import numpy as np
""",
    )

    interface = root / "input_pipeline" / "input_pipeline_interface.py"
    real_data_imports = """from maxtext.input_pipeline.grain_data_processing import make_grain_train_iterator
from maxtext.input_pipeline.grain_data_processing import make_grain_eval_iterator
from maxtext.input_pipeline.hf_data_processing import make_hf_train_iterator
from maxtext.input_pipeline.hf_data_processing import make_hf_eval_iterator
from maxtext.input_pipeline.olmo_grain_data_processing import make_olmo_grain_train_iterator
from maxtext.input_pipeline.olmo_grain_data_processing import make_olmo_grain_eval_iterator
from maxtext.input_pipeline.tfds_data_processing import make_tfds_train_iterator
from maxtext.input_pipeline.tfds_data_processing import make_tfds_eval_iterator
from maxtext.input_pipeline.tfds_data_processing_c4_mlperf import make_c4_mlperf_train_iterator
from maxtext.input_pipeline.tfds_data_processing_c4_mlperf import make_c4_mlperf_eval_iterator
"""
    replace_once(interface, real_data_imports, "")

    lazy_imports = """  # ZenteiQ Colab TPU synthetic patch: avoid TensorFlow imports unless real data is used.
  from maxtext.input_pipeline.grain_data_processing import make_grain_train_iterator
  from maxtext.input_pipeline.grain_data_processing import make_grain_eval_iterator
  from maxtext.input_pipeline.hf_data_processing import make_hf_train_iterator
  from maxtext.input_pipeline.hf_data_processing import make_hf_eval_iterator
  from maxtext.input_pipeline.olmo_grain_data_processing import make_olmo_grain_train_iterator
  from maxtext.input_pipeline.olmo_grain_data_processing import make_olmo_grain_eval_iterator
  from maxtext.input_pipeline.tfds_data_processing import make_tfds_train_iterator
  from maxtext.input_pipeline.tfds_data_processing import make_tfds_eval_iterator
  from maxtext.input_pipeline.tfds_data_processing_c4_mlperf import make_c4_mlperf_train_iterator
  from maxtext.input_pipeline.tfds_data_processing_c4_mlperf import make_c4_mlperf_eval_iterator

"""
    replace_once(interface, "  dataset_type_to_train_eval_iterator = {\n", lazy_imports + "  dataset_type_to_train_eval_iterator = {\n")

    print(f"Patched MaxText TPU synthetic imports in {root}")


if __name__ == "__main__":
    main()
