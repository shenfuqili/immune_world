#!/usr/bin/env bash
# Convenience wrapper around torchrun for the paper's 4× A100 pretraining run.
# Usage:
#   scripts/launch_train.sh experiment=main
#   scripts/launch_train.sh experiment=perturbation_norman
set -euo pipefail

EXPERIMENT="${1:-main}"
NPROC="${NPROC_PER_NODE:-4}"

exec torchrun --nproc_per_node="${NPROC}" \
    -m immune_world.cli.train "experiment=${EXPERIMENT}"
