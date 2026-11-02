#!/usr/bin/env bash
set -euo pipefail
# This is the documented Nemotron 3 Nano SFT recipe. Data must first be prepared/imported
# as the recipe's W&B Packed-Parquet SFTDataArtifact; raw JSONL is not accepted directly.
# TODO: configure valid W&B artifacts, cluster env.toml, Slurm resources, and CUDA container.
REPO_DIR="${NEMOTRON_REPO:-/workspace/Nemotron}"
cd "$REPO_DIR"
uv sync
# Repo README documents: uv run nemotron nano3 data prep sft --run PROFILE
# and: uv run nemotron nano3 sft --run PROFILE. It documents no local EDGAR JSONL input
# flag. Configure/import the artifact and profile before enabling these lines.
if [[ "${RUN_NEMOTRON_SFT:-0}" != "1" ]]; then
  echo "Not launching: set up W&B Packed-Parquet SFT artifact and env.toml profile; inspect configs/lora_sft.yaml." >&2
  exit 2
fi
: "${NEMOTRON_RUN_PROFILE:?Set NEMOTRON_RUN_PROFILE to a configured NeMo-Run cluster profile}"
uv run nemotron nano3 sft --run "$NEMOTRON_RUN_PROFILE"
