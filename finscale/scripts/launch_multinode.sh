#!/bin/bash
# Expects: NNODES, NODE_RANK, MASTER_ADDR, MASTER_PORT (set by Fuzzball or manually)
# TODO(fuzzball): confirm which env vars Fuzzball exposes for rank / node count / head address
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
# export NCCL_SOCKET_IFNAME=<iface>   # TODO: set if auto-detect picks the wrong NIC
torchrun --nnodes=${NNODES:-2} --node_rank=${NODE_RANK:-0} --nproc_per_node=1 \
  --master_addr=${MASTER_ADDR:?} --master_port=${MASTER_PORT:-29500} train_gb10.py "$@"
