#!/usr/bin/env bash
set -euo pipefail

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
    ;;
  cpu)
    uv pip install --system "maxtext[tpu]==${version}" --resolution=lowest
    ;;
  tpu)
    uv pip install --system "maxtext[tpu]==${version}" --resolution=lowest
    install_tpu_pre_train_extra_deps || true
    ;;
esac

python3 -c "import maxtext; import maxtext.trainers.pre_train.train; print('MaxText import OK')"
echo "MaxText ${version} setup complete for ${backend}."
