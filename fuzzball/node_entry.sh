#!/usr/bin/env bash
# Entry point that Fuzzball runs on EACH node of a multinode job.
# usage: node_entry.sh <node_rank> <head_ip> <task: nccl|train> [config.yaml]
# Cluster settings (port, NCCL interface) come from fuzzball/cluster.env, written by fuzzball/render.sh.
set -euo pipefail
NODE_RANK=$1
HEAD_ADDR=$2
TASK=$3
CONFIG=${4:-}
cd "$(dirname "$0")/.."
# shellcheck disable=SC1091
[[ -f fuzzball/cluster.env ]] && source fuzzball/cluster.env
export NODE_RANK HEAD_ADDR NNODES=${NNODES:-2} PORT=${PORT:-29400}
export NCCL_DEBUG=${NCCL_DEBUG:-INFO} NCCL_DEBUG_SUBSYS=${NCCL_DEBUG_SUBSYS:-INIT,NET}
echo "[node_entry] host=$(hostname) node_rank=$NODE_RANK head=$HEAD_ADDR port=$PORT task=$TASK" \
     "iface=${NCCL_SOCKET_IFNAME:-auto}"

case "$TASK" in
  nccl)
    exec torchrun --nnodes="$NNODES" --nproc-per-node=1 --node-rank="$NODE_RANK" \
      --master-addr="$HEAD_ADDR" --master-port="$PORT" scripts/nccl_bench.py --out results/nccl ;;
  train)
    [[ -n "$CONFIG" ]] || { echo "train needs a config path" >&2; exit 2; }
    exec bash scripts/launch_train.sh "$CONFIG" ;;
  *)
    echo "unknown task $TASK" >&2; exit 2 ;;
esac
