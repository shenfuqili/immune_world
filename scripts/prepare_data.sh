#!/usr/bin/env bash
# Fetch + preprocess every dataset in `docs/project-context.md §6`. SHA-256 manifests written
# to ${IMMUNE_WORLD_DATA_DIR:-data/processed}/manifests/<dataset>.sha256.
set -euo pipefail

DATASETS=("$@")
if [[ "${#DATASETS[@]}" -eq 0 ]]; then
    DATASETS=(all)
fi

exec python -m immune_world.cli.prepare_data "${DATASETS[@]}"
