#!/usr/bin/env bash
set -euo pipefail
trap 'echo "Setup failed at line ${LINENO}: ${BASH_COMMAND}" >&2' ERR

backend="${1:-}"
version="${MAXTEXT_VERSION:-0.2.2}"

if [[ "${backend}" != "cpu" && "${backend}" != "gpu" && "${backend}" != "tpu" ]]; then
  echo "Usage: $0 {cpu|gpu|tpu}" >&2
  exit 2
fi

python3 -m pip install -q uv

case "${backend}" in
  gpu)
    uv pip install --system "maxtext[cuda12]==${version}" --resolution=lowest
    install_cuda12_pre_train_extra_deps || true
    # ponytail: Colab's cuDNN can mismatch optional Transformer Engine; drop TE unless FP8 runs need it.
    python3 -m pip uninstall -y -q transformer-engine transformer-engine-cu12 transformer-engine-jax || true
    ;;
  cpu)
    uv pip install --system "maxtext[tpu]==${version}" --resolution=lowest
    ;;
  tpu)
    uv pip install --system "maxtext[tpu]==${version}" --resolution=lowest
    install_tpu_pre_train_extra_deps || true
    python3 scripts/patch_maxtext_tpu_colab.py
    ;;
esac

if [[ "${backend}" == "tpu" ]]; then
  python3 -c "import importlib.util; assert importlib.util.find_spec('maxtext'); print('MaxText package found')"
else
  python3 -c "import maxtext; import maxtext.trainers.pre_train.train; print('MaxText import OK')"
fi
echo "MaxText ${version} setup complete for ${backend}."
