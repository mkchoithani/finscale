#!/usr/bin/env python3
"""Download Nemotron 3 Nano 30B-A3B (BF16, ~59 GiB) to shared storage once.

    export HF_TOKEN=hf_...
    python scripts/download_model.py --dest /shared/models/nemotron-3-nano-30b
Re-running resumes; it also verifies the shard count listed in the index file.
"""
import argparse
import json
import os
import time
from pathlib import Path

REPO = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=REPO)
    ap.add_argument("--dest", required=True)
    ap.add_argument("--revision", default=None, help="pin a commit for reproducibility")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    from huggingface_hub import snapshot_download
    t = time.time()
    path = snapshot_download(repo_id=a.repo, local_dir=a.dest, revision=a.revision, max_workers=a.workers,
                             token=os.environ.get("HF_TOKEN") or None)
    idx = Path(path) / "model.safetensors.index.json"
    if idx.exists():
        expected = set(json.loads(idx.read_text())["weight_map"].values())
        missing = [f for f in expected if not (Path(path) / f).exists()]
        print(f"shards: {len(expected) - len(missing)}/{len(expected)} present")
        if missing:
            raise SystemExit(f"missing shards: {missing[:5]}")
    print(f"downloaded {a.repo} -> {path} in {(time.time() - t) / 60:.1f} min")


if __name__ == "__main__":
    main()
