#!/usr/bin/env python3
"""Multi-node NCCL communication proof. Not proof of model training."""
import json, os, socket, time
from pathlib import Path
import torch
import torch.distributed as dist

def main():
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    hostname = socket.gethostname()
    device = torch.device("cuda", local_rank)
    # Warm-up: the first collective includes NCCL communicator setup; don't time it.
    warm = torch.ones(1, device=device)
    dist.all_reduce(warm)
    x = torch.tensor([float(rank + 1)], device=device)
    torch.cuda.synchronize()
    start = time.perf_counter()
    dist.all_reduce(x, op=dist.ReduceOp.SUM)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    expected = world * (world + 1) / 2
    assert x.item() == expected, (x.item(), expected)
    result = {
        "rank": rank, "local_rank": local_rank, "world_size": world,
        "hostname": hostname, "gpu": torch.cuda.get_device_name(local_rank),
        "all_reduce_sum": x.item(), "expected_sum": expected,
        "all_reduce_seconds": elapsed, "status": "PASS",
    }
    out = Path(os.environ.get("FINSCALE_EVIDENCE_DIR", "results/distributed"))
    out.mkdir(parents=True, exist_ok=True)
    (out / f"rank-{rank}.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)
    dist.barrier()
    dist.destroy_process_group()

if __name__ == "__main__":
    main()
