#!/usr/bin/env bash
# Rehearse the whole GB10 plan on a RunPod pod with 2 GPUs, BEFORE the event.
# Two "nodes" = two independent torchrun launches on one host (one GPU each), with NCCL forced to use
# sockets instead of shared memory / NVLink, so the multi-node code path is exercised end to end.
# Throughput on H100/H200 is NOT a GB10 number; this proves the code, not the performance.
#
# Pod: 2x H100 80GB (or 2x H200), image nvcr.io/nvidia/nemo-automodel:26.10.00, volume >= 200 GB at /workspace.
#   cd /workspace && git clone --branch finscale-multinode https://github.com/mkchoithani/finscale.git FinscaleNvidiaPlatform
#   cd FinscaleNvidiaPlatform && export HF_TOKEN=... NVIDIA_API_KEY=...
#   bash runpod/rehearse.sh all 2>&1 | tee results/rehearsal.log
# Or run one stage at a time: setup gate smoke nccl single multi modes eval charts
set -euo pipefail
cd "$(dirname "$0")/.."
export MODEL_DIR=${MODEL_DIR:-/workspace/models/nemotron-3-nano-30b}
PORT=${PORT:-29610}
STEPS=${STEPS:-8}            # short runs: enough steps for a steady-state tokens/s median
GBS=${GBS:-16}
SEQ=${SEQ:-3072}
STAGE=${1:-all}
mkdir -p results

say() { printf '\n========== %s ==========\n' "$*"; }
want() { [[ "$STAGE" == all || "$STAGE" == "$1" ]]; }

# Launch the same command as two "nodes" on this host, GPU 0 and GPU 1, network transport only.
two_nodes() {  # usage: two_nodes <cmd...> ; NODE_RANK / HEAD_ADDR / PORT are set per process
  local pids=()
  for r in 0 1; do
    CUDA_VISIBLE_DEVICES=$r NNODES=2 NODE_RANK=$r HEAD_ADDR=127.0.0.1 PORT=$PORT \
    NCCL_P2P_DISABLE=1 NCCL_SHM_DISABLE=1 NCCL_IB_DISABLE=1 NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-lo} \
      "$@" > "results/two_nodes_rank$r.out" 2>&1 &
    pids+=($!)
  done
  local rc=0
  for p in "${pids[@]}"; do wait "$p" || rc=$?; done
  tail -n 5 results/two_nodes_rank0.out results/two_nodes_rank1.out
  return $rc
}

if want setup; then
  say "SETUP: python deps, model download (~59 GiB), preflight"
  nvidia-smi -L
  python3 -m pip install --quiet openai matplotlib "peft>=0.19.1" 2>&1 | tail -2 || true
  python3 scripts/download_model.py --dest "$MODEL_DIR"
  python3 scripts/00_preflight.py --need-gpu --model-dir "$MODEL_DIR"
fi

if want gate; then
  say "GATE: every answer survives the real AutoModel ChatDataset at seq $SEQ"
  [[ -s data/sft/train.jsonl ]] || { echo "data/sft/train.jsonl missing: do Phase A and push the data first"; exit 1; }
  python3 scripts/05_check_tokens.py --model "$MODEL_DIR" --seq-length "$SEQ"
fi

if want smoke; then
  say "SMOKE: 1 optimizer step on 1 GPU"
  python3 scripts/06_make_config.py --mode single --run rh-smoke-1n --steps 1 --no-save --val '' --model "$MODEL_DIR"
  CUDA_VISIBLE_DEVICES=0 bash scripts/launch_train.sh configs/rh-smoke-1n.yaml
fi

if want nccl; then
  say "NCCL: 2 'nodes' over sockets"
  # single quotes: NODE_RANK / HEAD_ADDR / PORT expand inside each process
  two_nodes bash -c 'torchrun --nnodes=2 --nproc-per-node=1 --node-rank=$NODE_RANK --master-addr=$HEAD_ADDR \
    --master-port=$PORT scripts/nccl_bench.py --out results/rehearsal-nccl'
  python3 -c "import json;r=json.load(open('results/rehearsal-nccl/nccl_report.json'));print(r['status'],r['allreduce_sum'])"
fi

if want single; then
  say "SINGLE: 1-node baseline, $STEPS steps, saves adapter"
  python3 scripts/06_make_config.py --mode single --run rh-single-1n --steps "$STEPS" --gbs "$GBS" --seq-length "$SEQ" \
    --model "$MODEL_DIR"
  CUDA_VISIBLE_DEVICES=0 bash scripts/launch_train.sh configs/rh-single-1n.yaml
  ls results/runs/rh-single-1n/ckpt/LATEST/model/
fi

if want multi; then
  say "MULTI: replicate mode on 2 'nodes', same GBS/seq"
  python3 scripts/06_make_config.py --mode replicate --nodes 2 --run rh-replicate-2n --steps "$STEPS" --gbs "$GBS" \
    --seq-length "$SEQ" --no-save --model "$MODEL_DIR"
  two_nodes bash scripts/launch_train.sh configs/rh-replicate-2n.yaml
  python3 scripts/parse_train_log.py --run-dir results/runs/rh-replicate-2n
fi

if want modes; then
  say "MODES: fallbacks ddp / shard / ep (each must at least reach step 1)"
  for m in ddp shard ep; do
    python3 scripts/06_make_config.py --mode "$m" --nodes 2 --run "rh-$m-2n" --steps 2 --gbs "$GBS" --seq-length "$SEQ" \
      --no-save --val '' --model "$MODEL_DIR"
    if two_nodes bash scripts/launch_train.sh "configs/rh-$m-2n.yaml"; then echo "MODE $m: OK"; else echo "MODE $m: FAILED"; fi
  done
fi

if want eval; then
  say "EVAL: same 30B weights without vs with the LoRA adapter, on GPU (20 test rows)"
  CUDA_VISIBLE_DEVICES=0 python3 scripts/07_predict.py --name base --backend hf --limit 20 --model-path "$MODEL_DIR"
  CUDA_VISIBLE_DEVICES=0 python3 scripts/07_predict.py --name finetuned --backend hf --limit 20 \
    --model-path "$MODEL_DIR" --adapter results/runs/rh-single-1n/ckpt/LATEST/model
  python3 scripts/08_score.py results/eval/pred-base.jsonl results/eval/pred-finetuned.jsonl --out results/eval/rehearsal_scores.json
fi

if want charts; then
  say "CHARTS"
  python3 scripts/plot_results.py --runs results/runs/rh-single-1n results/runs/rh-replicate-2n \
    --scores results/eval/rehearsal_scores.json --nccl results/rehearsal-nccl/nccl_report.json --out results/rehearsal-charts
fi
say "DONE ($STAGE)"
