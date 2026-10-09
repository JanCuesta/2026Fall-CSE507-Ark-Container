#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="/workspace/.gaudi_pydeps:${PYTHONPATH:-}"
IMAGE="/scratch/jcuesta/BME/Practice3_UniMiSSPlus/Gaudi_Demo/gaudi-apptainer.sif"
[[ -f "$IMAGE" ]] || { echo "Gaudi image not found: $IMAGE" >&2; exit 1; }
HABANA_LOG_DIR="${SCRATCH:-$HOME}/.habana_logs"
mkdir -p "$HABANA_LOG_DIR"
exec apptainer exec --bind /dev/shm:/dev/shm --bind /scratch:/scratch:rw --bind /data:/data:rw --bind "$ROOT:/workspace:rw" --bind "$HABANA_LOG_DIR:/var/log/habana_logs" --pwd /workspace/base --env PYTHONPATH=/workspace/.gaudi_pydeps --env PT_HPU_LAZY_MODE="${PT_HPU_LAZY_MODE:-1}" "$IMAGE" "$@"
