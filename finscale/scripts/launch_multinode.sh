#!/bin/bash
set -euo pipefail
# Fuzzball-documented environment: MULTINODE_NODE_IP,
# MULTINODE_HOSTLIST_NOSLOTS, MULTINODE_SSH_WRAPPER.
# TODO: confirm exact wrapper invocation and host-list format for the event version.
: "${MULTINODE_NODE_IP:?MULTINODE_NODE_IP is required}"
: "${MULTINODE_HOSTLIST_NOSLOTS:?MULTINODE_HOSTLIST_NOSLOTS is required}"
: "${MULTINODE_SSH_WRAPPER:?MULTINODE_SSH_WRAPPER is required}"
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
# export NCCL_SOCKET_IFNAME=<iface>   # Set only if auto-detection selects the wrong NIC.

# HOSTLIST_NOSLOTS is treated as a comma- or whitespace-separated list of hosts.
mapfile -t HOSTS < <(printf '%s' "$MULTINODE_HOSTLIST_NOSLOTS" | tr ', ' '\n' | sed '/^$/d')
NNODES=${#HOSTS[@]}
if (( NNODES != 2 )); then
  echo "Expected 2 hosts; found $NNODES: ${HOSTS[*]}" >&2
  exit 2
fi

pids=()
for rank in "${!HOSTS[@]}"; do
  host=${HOSTS[$rank]}
  cmd=(torchrun "--nnodes=${NNODES}" "--node_rank=${rank}" --nproc_per_node=1
       --master-addr "${MULTINODE_NODE_IP}" --master-port "${MASTER_PORT:-29500}"
       train_gb10.py "$@")
  if (( rank == 0 )); then
    "${cmd[@]}" &
  else
    # TODO: confirm helper/version argument order; expected pattern is wrapper HOST COMMAND.
    remote_cmd=$(printf '%q ' "${cmd[@]}")
    "${MULTINODE_SSH_WRAPPER}" "$host" bash -lc "$remote_cmd" &
  fi
  pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do
  wait "$pid" || status=$?
done
exit "$status"
