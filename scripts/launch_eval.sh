#!/usr/bin/env bash
# Single-GPU evaluation launcher.
# Usage: scripts/launch_eval.sh experiment=main checkpoint=path/to/ckpt
set -euo pipefail

EXPERIMENT="${1:-main}"
shift || true

exec python -m immune_world.cli.eval "experiment=${EXPERIMENT}" "$@"
