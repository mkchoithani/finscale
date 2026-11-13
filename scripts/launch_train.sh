#!/usr/bin/env bash
# Launch one FinScale training run on THIS node. For multi-node, run the same command on every node.
#
# Single node:
#   bash scripts/launch_train.sh configs/single-1n.yaml
# Two nodes (run on both, at the same time):
#   NNODES=2 NODE_RANK=0 HEAD_ADDR=<node0 IP> PORT=<allowed port> bash scripts/launch_train.sh configs/replicate-2n.yaml
#   NNODES=2 NODE_RANK=1 HEAD_ADDR=<node0 IP> PORT=<allowed port> bash scripts/launch_train.sh configs/replicate-2n.yaml
# Leave NODE_RANK unset to let torchrun's c10d rendezvous assign ranks.
#
# Optional env: NPROC_PER_NODE (default 1, GB10 has one GPU), NCCL_SOCKET_IFNAME, NCCL_IB_HCA,
#               RESULTS_DIR (default results/runs), NO_TELEMETRY=1
set -euo pipefail

CONFIG=${1:?usage: launch_train.sh <config.yaml>  (path relative to the repo root)}
cd "$(dirname "$0")/.."
[[ -f "$CONFIG" ]] || { echo "Config not found: $CONFIG (from $(pwd))" >&2; exit 2; }

NNODES=${NNODES:-1}
NPROC=${NPROC_PER_NODE:-1}
PORT=${PORT:-29400}
RUN=$(basename "$CONFIG" .yaml)
RUN_DIR=${RESULTS_DIR:-results/runs}/$RUN
HOST=$(hostname -s)${NODE_RANK:+-n$NODE_RANK}   # rank suffix keeps files apart when 2 "nodes" share a host (rehearsal)
mkdir -p "$RUN_DIR/logs" "$RUN_DIR/nodes"

export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8} MKL_NUM_THREADS=${MKL_NUM_THREADS:-8}
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false

python3 scripts/node_manifest.py --out "$RUN_DIR/nodes/$HOST.json" --config "$CONFIG" || true

TPID=""
if [[ -z "${NO_TELEMETRY:-}" ]]; then
  python3 scripts/telemetry.py --output "$RUN_DIR/telemetry-$HOST.csv" --interval 2 >/dev/null 2>&1 &
  TPID=$!
fi
cleanup() { [[ -n "$TPID" ]] && kill "$TPID" 2>/dev/null || true; }
trap cleanup EXIT

if [[ "$NNODES" == "1" ]]; then
  LAUNCH=(torchrun --standalone --nnodes=1 --nproc-per-node="$NPROC")
else
  HEAD_ADDR=${HEAD_ADDR:?set HEAD_ADDR to node 0 IP for multi-node}
  export NCCL_DEBUG=${NCCL_DEBUG:-INFO} NCCL_DEBUG_SUBSYS=${NCCL_DEBUG_SUBSYS:-INIT,NET}
  if [[ -n "${NODE_RANK:-}" ]]; then
    LAUNCH=(torchrun --nnodes="$NNODES" --nproc-per-node="$NPROC" --node-rank="$NODE_RANK"
            --master-addr="$HEAD_ADDR" --master-port="$PORT")
  else
    LAUNCH=(torchrun --nnodes="$NNODES" --nproc-per-node="$NPROC" --rdzv-backend=c10d
            --rdzv-endpoint="$HEAD_ADDR:$PORT" --rdzv-id="$RUN")
  fi
fi

echo "[finscale] host=$HOST run=$RUN nnodes=$NNODES node_rank=${NODE_RANK:-auto} head=${HEAD_ADDR:-local} port=$PORT"
echo "[finscale] ${LAUNCH[*]} -m nemo_automodel.cli.app $CONFIG"
START=$(date +%s)
set +e
"${LAUNCH[@]}" -m nemo_automodel.cli.app "$CONFIG" 2>&1 | tee "$RUN_DIR/logs/$HOST.log"
RC=${PIPESTATUS[0]}
set -e
END=$(date +%s)
printf '{"host": "%s", "exit_code": %d, "wall_seconds": %d, "start_epoch": %d, "end_epoch": %d}\n' \
  "$HOST" "$RC" "$((END - START))" "$START" "$END" > "$RUN_DIR/nodes/$HOST.exit.json"
echo "[finscale] exit=$RC wall=$((END - START))s log=$RUN_DIR/logs/$HOST.log"

# The rank-0 node writes training.jsonl; summarize wherever it exists.
if [[ -f "$RUN_DIR/ckpt/training.jsonl" ]]; then
  python3 scripts/parse_train_log.py --run-dir "$RUN_DIR" || true
fi
exit "$RC"
