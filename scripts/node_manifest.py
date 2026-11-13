#!/usr/bin/env python3
"""Record what ran on this node: host, IPs, GPU, software versions, code and data hashes.
Called by launch_train.sh; also useful on its own as a preflight on each node."""
import argparse
import hashlib
import json
import os
import platform
import shutil
import socket
import subprocess
import time
from pathlib import Path


def sh(cmd):
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception:
        return ""


def sha(path):
    try:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    except OSError:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--config", default=None)
    a = ap.parse_args()

    info = {
        "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "hostname": socket.gethostname(), "fqdn": socket.getfqdn(),
        "arch": platform.machine(), "python": platform.python_version(),
        "ip_addrs": sh("ip -br addr 2>/dev/null || ifconfig 2>/dev/null | grep 'inet '"),
        "nvidia_smi": sh("nvidia-smi --query-gpu=name,driver_version --format=csv,noheader"),
        "mem_total_kib": sh("grep MemTotal /proc/meminfo | awk '{print $2}'"),
        "git_commit": sh("git rev-parse HEAD"), "git_dirty": bool(sh("git status --porcelain")),
        "config": a.config, "config_sha256": sha(a.config) if a.config else None,
        "env": {k: v for k, v in os.environ.items()
                if k.startswith(("NCCL_", "MASTER_", "NNODES", "NODE_RANK", "HEAD_ADDR", "PORT", "MULTINODE_",
                                 "CUDA_VISIBLE", "OMP_NUM"))},
        "disk_free_gb": round(shutil.disk_usage(".").free / 1e9, 1),
    }
    manifest = Path("data/sft/manifest.json")
    if manifest.exists():
        info["data_sha256"] = json.loads(manifest.read_text()).get("sha256")
    try:
        import torch
        info["torch"] = torch.__version__
        info["cuda"] = torch.version.cuda
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
            info["gpu_capability"] = ".".join(map(str, torch.cuda.get_device_capability(0)))
            info["nccl"] = ".".join(map(str, torch.cuda.nccl.version()))
    except Exception as e:
        info["torch_error"] = str(e)
    for mod in ("nemo_automodel", "transformers", "mamba_ssm", "causal_conv1d", "peft"):
        try:
            m = __import__(mod)
            info[f"{mod}_version"] = getattr(m, "__version__", "installed")
        except Exception as e:
            info[f"{mod}_version"] = f"MISSING ({type(e).__name__})"
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(info, indent=2))
    print(json.dumps({k: info.get(k) for k in ("hostname", "arch", "gpu", "torch", "cuda", "nccl",
                                                "nemo_automodel_version", "git_commit")}))


if __name__ == "__main__":
    main()
