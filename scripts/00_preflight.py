#!/usr/bin/env python3
"""Node readiness check. Run on EACH GB10 node (and on your laptop for the CPU steps).

    python scripts/00_preflight.py --model-dir /shared/models/nemotron-3-nano-30b
Prints PASS/WARN/FAIL per item and saves results/preflight/<host>.json.
"""
import argparse
import importlib
import json
import os
import platform
import shutil
import socket
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default=os.environ.get("MODEL_DIR", ""))
    ap.add_argument("--need-gpu", action="store_true", default=None)
    a = ap.parse_args()
    checks = []

    def add(name, status, detail=""):
        checks.append({"check": name, "status": status, "detail": str(detail)})
        print(f"[{status:4}] {name}: {detail}")

    add("host", "INFO", f"{socket.gethostname()} {platform.machine()} python {platform.python_version()}")
    try:
        import torch
        if torch.cuda.is_available():
            add("cuda", "PASS", f"{torch.cuda.get_device_name(0)} | torch {torch.__version__} cuda {torch.version.cuda} "
                                f"| nccl {'.'.join(map(str, torch.cuda.nccl.version()))}")
            x = torch.randn(1024, 1024, device="cuda", dtype=torch.bfloat16)
            add("cuda_matmul", "PASS", f"bf16 matmul ok ({float((x @ x).float().abs().mean()):.2f})")
        else:
            add("cuda", "FAIL" if a.need_gpu else "WARN", "torch.cuda.is_available() is False")
    except ImportError:
        add("torch", "FAIL" if a.need_gpu else "WARN", "torch not installed (fine on a laptop)")

    for mod, why in [("nemo_automodel", "training"), ("mamba_ssm", "Nemotron-H Mamba layers"),
                     ("causal_conv1d", "Mamba conv kernel"), ("transformers", "tokenizer"),
                     ("openai", "labeling / API eval"), ("yaml", "configs"), ("matplotlib", "charts")]:
        try:
            m = importlib.import_module(mod)
            add(mod, "PASS", getattr(m, "__version__", "installed"))
        except Exception as e:
            add(mod, "WARN", f"missing ({why}): {type(e).__name__}")

    if a.model_dir:
        p = Path(a.model_dir)
        shards = list(p.glob("*.safetensors"))
        if (p / "config.json").exists() and shards:
            size = sum(s.stat().st_size for s in shards) / 2**30
            add("model", "PASS", f"{p}: {len(shards)} shards, {size:.1f} GiB")
        else:
            add("model", "FAIL", f"{p}: config.json or *.safetensors missing (run scripts/download_model.py)")
    for var in ("NVIDIA_API_KEY", "HF_TOKEN", "SEC_USER_AGENT"):
        add(var, "PASS" if os.environ.get(var) else "WARN", "set" if os.environ.get(var) else "not set")
    free = shutil.disk_usage(".").free / 1e9
    add("disk", "PASS" if free > 100 else "WARN", f"{free:.0f} GB free in {Path.cwd()}")
    try:
        mem = {l.split(":")[0]: int(l.split()[1]) for l in Path("/proc/meminfo").read_text().splitlines()
               if l.startswith(("MemTotal", "MemAvailable"))}
        add("memory", "INFO", f"{mem['MemAvailable'] / 2**20:.0f} GiB available of {mem['MemTotal'] / 2**20:.0f} GiB")
    except OSError:
        pass

    out = Path("results/preflight") / f"{socket.gethostname()}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(checks, indent=2))
    fails = [c for c in checks if c["status"] == "FAIL"]
    print(f"\n{'FAIL' if fails else 'OK'}: {len(fails)} failure(s). Saved {out}")
    raise SystemExit(1 if fails else 0)


if __name__ == "__main__":
    main()
