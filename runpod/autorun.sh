#!/usr/bin/env bash
# Unattended rehearsal: the pod runs this at start-up, so nobody needs a terminal on the pod.
# Each stage's output goes to the pod log (readable from the RunPod console or the MCP) and to
# results/rh-<stage>.log. Stops at the first failing stage. The pod then stays up (sleep infinity
# in the start command) so results can be inspected; delete the pod when done.
#
# Pod start command (container image nvcr.io/nvidia/nemo-automodel:26.10.00):
#   bash -c "cd /workspace && (git -C FinscaleNvidiaPlatform pull || git clone --branch finscale-multinode
#            https://github.com/mkchoithani/finscale.git FinscaleNvidiaPlatform)
#            && bash FinscaleNvidiaPlatform/runpod/autorun.sh; sleep infinity"
# Optional env: REHEARSAL_STAGES (default "setup gate smoke single eval"; add "nccl multi" on 2 GPUs)
set -uo pipefail
cd "$(dirname "$0")/.."
STAGES=${REHEARSAL_STAGES:-"setup gate smoke single eval"}
mkdir -p results
echo "##### AUTORUN START $(date -u +%FT%TZ) commit=$(git rev-parse --short HEAD) stages: $STAGES"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true
status=OK
for s in $STAGES; do
  echo "##### STAGE $s START $(date -u +%T)"
  if bash runpod/rehearse.sh "$s" 2>&1 | tee "results/rh-$s.log"; then
    echo "##### STAGE $s OK $(date -u +%T)"
  else
    echo "##### STAGE $s FAILED $(date -u +%T): last lines above; full log results/rh-$s.log"
    status="FAILED at $s"
    break
  fi
done
echo "##### RESULTS"
for f in results/preflight/*.json results/runs/*/summary.json results/eval/rehearsal_scores.md \
         results/rehearsal-nccl/nccl_report.json; do
  [[ -f "$f" ]] && { echo "--- $f"; head -c 4000 "$f"; echo; }
done
echo "##### AUTORUN DONE $(date -u +%FT%TZ) status=$status"
