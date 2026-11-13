#!/usr/bin/env bash
# FinScale RunPod bootstrap. Run INSIDE the pod, from any directory:
#   bash /workspace/FinscaleNvidiaPlatform/runpod/setup_pod.sh
#
# Expected pod image: NGC NeMo AutoModel container (x86_64), e.g.
#   nvcr.io/nvidia/nemo-automodel:26.10.00
# It ships torch, transformers, mamba-ssm, causal-conv1d and AutoModel at /opt/Automodel,
# so none of the GB10 ARM64 source builds are needed. If you used a plain PyTorch
# image instead, set INSTALL_AUTOMODEL=1 (slower, builds CUDA extensions).
set -euo pipefail

WS="${WS:-/workspace}"
REPO_URL="${REPO_URL:-https://github.com/mkchoithani/finscale.git}"
BRANCH="${BRANCH:-nemotron-runpod-setup}"
REPO="$WS/FinscaleNvidiaPlatform"
MODEL_ID="${MODEL_ID:-nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16}"
MODEL_DIR="${MODEL_DIR:-$WS/models/nemotron-30b}"
export HF_HOME="${HF_HOME:-$WS/.cache/huggingface}"   # keep caches on the persistent volume
EVID="$REPO/results/runpod"

log() { printf '\n==== %s\n' "$*"; }

log "1. Hardware / volume checks"
test -w "$WS" || { echo "$WS not writable: attach a network/persistent volume"; exit 1; }
uname -m
nvidia-smi --query-gpu=index,name,memory.total,memory.free,compute_cap --format=csv
free_gb=$(df -BG --output=avail "$WS" | tail -1 | tr -dc '0-9')
echo "Free on $WS: ${free_gb} GB"
[ "$free_gb" -ge 120 ] || { echo "Need >=120 GB free (59 GiB model + caches + checkpoints)"; exit 1; }
min_mem=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | sort -n | head -1)
[ "$min_mem" -ge 79000 ] || echo "WARNING: <80 GB per GPU. The single-GPU BF16 30B LoRA recipe targets 80 GB+ (H100/H200/RTX PRO 6000)."

log "2. Repo"
if [ -d "$REPO/.git" ]; then
  git -C "$REPO" pull --ff-only || echo "pull skipped (local changes or different branch)"
else
  git clone --branch "$BRANCH" "$REPO_URL" "$REPO"
fi
cd "$REPO"
mkdir -p "$EVID"

log "3. Python environment"
if [ "${INSTALL_AUTOMODEL:-0}" = "1" ]; then
  pip install --upgrade pip
  pip install "nemo-automodel" || { echo "pip install nemo-automodel failed; use the NGC container"; exit 1; }
  pip install --no-build-isolation causal-conv1d mamba-ssm
fi
# Small extras used by FinScale scripts (do NOT reinstall torch).
pip install --quiet "pyyaml>=6" "matplotlib>=3.8" "requests>=2.31" "openai>=1.0" "huggingface_hub>=0.24"
python - <<'PY'
import torch, transformers, importlib
print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available())
print("transformers", transformers.__version__)
for m in ("nemo_automodel", "mamba_ssm", "causal_conv1d"):
    try:
        mod = importlib.import_module(m); print(m, getattr(mod, "__version__", "ok"))
    except Exception as e:
        raise SystemExit(f"MISSING {m}: {e}")
PY
command -v automodel >/dev/null || { echo "automodel CLI not on PATH"; exit 1; }
for d in /opt/Automodel /workspace/Automodel; do
  [ -f "$d/examples/llm_finetune/nemotron/nemotron_nano_v3_singlegpu_lora.yaml" ] && export AUTOMODEL_HOME="$d"
done
echo "AUTOMODEL_HOME=${AUTOMODEL_HOME:-NOT FOUND (pass --recipe to prepare_gb10_config.py)}"

log "4. Model download -> $MODEL_DIR (~59 GiB, resumable)"
python - "$MODEL_ID" "$MODEL_DIR" <<'PY'
import sys, os
from huggingface_hub import snapshot_download
repo, dest = sys.argv[1:3]
snapshot_download(repo_id=repo, local_dir=dest, token=os.environ.get("HF_TOKEN"), max_workers=8)
PY
shards=$(ls "$MODEL_DIR"/model-*.safetensors 2>/dev/null | wc -l)
echo "safetensors shards: $shards"; du -sh "$MODEL_DIR"
[ "$shards" -ge 1 ] || { echo "Model download incomplete"; exit 1; }

log "5. Data"
if [ -s data/processed/train.jsonl ]; then
  wc -l data/processed/train.jsonl data/processed/val.jsonl
  sha256sum data/processed/*.jsonl | tee "$EVID/dataset.sha256"
else
  echo "No data/processed/train.jsonl. Copy finscale-edgar-sft-backup.tar.gz to $WS and run:"
  echo "  tar -C $REPO -xzf $WS/finscale-edgar-sft-backup.tar.gz"
fi

log "6. Manifest"
{
  echo "date_utc=$(date -u +%FT%TZ)"
  echo "host=$(hostname)"
  echo "git_sha=$(git rev-parse HEAD)"
  echo "model_id=$MODEL_ID"
  nvidia-smi -L
  python -c "import torch,nemo_automodel as n;print('torch',torch.__version__,'automodel',getattr(n,'__version__','?'))"
} | tee "$EVID/manifest.txt"

log "DONE. Next: runpod/RUNPOD_QUICKSTART.md step 3 (fit data to seq length)"
