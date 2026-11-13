#!/usr/bin/env python3
"""Two-node NCCL proof + bandwidth: correctness check, warm-up, then all-reduce timings by size.

Run under torchrun on every node (one process per GPU):
  torchrun --nnodes=2 --nproc-per-node=1 --node-rank=<0|1> --master-addr=<node0 IP> --master-port=<port> \
      scripts/nccl_bench.py --out results/nccl
Proves cross-node communication only, not distributed training.
"""
import argparse
import json
import os
import socket
import time
from pathlib import Path

import torch
import torch.distributed as dist


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/nccl")
    ap.add_argument("--sizes-mb", type=float, nargs="+", default=[1, 8, 64, 256, 1024])
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=5)
    a = ap.parse_args()

    rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    local = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local)
    t_init = time.perf_counter()
    dist.init_process_group("nccl", device_id=torch.device("cuda", local))
    init_s = time.perf_counter() - t_init

    # 1. Correctness: sum of (rank+1) over all ranks.
    x = torch.tensor([float(rank + 1)], device="cuda")
    dist.all_reduce(x)
    torch.cuda.synchronize()
    expected = world * (world + 1) / 2
    ok = abs(x.item() - expected) < 1e-6

    # 2. Who is where (proves ranks are on different hosts).
    me = {"rank": rank, "host": socket.gethostname(), "gpu": torch.cuda.get_device_name(local)}
    everyone = [None] * world
    dist.all_gather_object(everyone, me)

    # 3. Bandwidth sweep. busbw = algbw * 2(n-1)/n (standard nccl-tests definition for all-reduce).
    results = []
    for mb in a.sizes_mb:
        n = int(mb * 1024 * 1024 / 2)  # bf16 elements
        buf = torch.ones(n, dtype=torch.bfloat16, device="cuda")
        for _ in range(a.warmup):
            dist.all_reduce(buf)
        torch.cuda.synchronize()
        dist.barrier()
        t = time.perf_counter()
        for _ in range(a.iters):
            dist.all_reduce(buf)
        torch.cuda.synchronize()
        sec = (time.perf_counter() - t) / a.iters
        algbw = (n * 2) / sec / 1e9
        results.append({"size_mb": mb, "time_ms": round(sec * 1e3, 3), "algbw_GBps": round(algbw, 3),
                        "busbw_GBps": round(algbw * 2 * (world - 1) / world, 3)})
        del buf

    if rank == 0:
        hosts = sorted({e["host"] for e in everyone})
        report = {"status": "PASS" if ok and len(hosts) == world else ("PASS_SAME_HOST" if ok else "FAIL"),
                  "world_size": world, "distinct_hosts": hosts, "ranks": everyone, "allreduce_sum": x.item(),
                  "expected_sum": expected, "init_seconds": round(init_s, 2), "bandwidth": results,
                  "nccl_env": {k: v for k, v in os.environ.items() if k.startswith("NCCL_")},
                  "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        out = Path(a.out)
        out.mkdir(parents=True, exist_ok=True)
        (out / "nccl_report.json").write_text(json.dumps(report, indent=2))
        print(json.dumps({k: report[k] for k in ("status", "distinct_hosts", "allreduce_sum")}))
        for r in results:
            print(f"  {r['size_mb']:>7} MB  {r['time_ms']:>9} ms  busbw {r['busbw_GBps']:>7} GB/s")
        print(f"saved {out / 'nccl_report.json'}")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
