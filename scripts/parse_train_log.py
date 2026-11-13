#!/usr/bin/env python3
"""Turn AutoModel's per-step training log into steps.csv + summary.json for one run.

Reads <run>/ckpt/training.jsonl (written by rank 0); falls back to the
"step N | ... | tps X" console lines in <run>/logs/*.log.
    python scripts/parse_train_log.py --run-dir results/runs/replicate-2n
"""
import argparse
import csv
import json
import re
import statistics
from datetime import datetime
from pathlib import Path

LINE_RE = re.compile(r"step (\d+) \| epoch (\d+) \| loss ([\d.eE+-]+) .*?\| mem ([\d.]+) GiB \| tps ([\d.]+)"
                     r"\(([\d.]+)/gpu\) \| num_label_tokens (\d+) \| num_tokens_per_step (\d+)")


def load_jsonl(path):
    rows = {}
    for line in path.read_text().splitlines():
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        rows[int(r["step"])] = r  # a restarted run overwrites earlier entries for the same step
    return [rows[k] for k in sorted(rows)]


def load_console(run_dir):
    rows = {}
    for log in sorted((run_dir / "logs").glob("*.log")):
        for m in LINE_RE.finditer(log.read_text(errors="ignore")):
            s = int(m.group(1))
            rows[s] = {"step": s, "epoch": int(m.group(2)), "loss": float(m.group(3)), "mem": float(m.group(4)),
                       "tps": float(m.group(5)), "tps_per_gpu": float(m.group(6)),
                       "num_label_tokens": int(m.group(7)), "num_tokens_per_step": int(m.group(8))}
    return [rows[k] for k in sorted(rows)]


def ts(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp() if s else None


def num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--warmup-steps", type=int, default=3, help="excluded from throughput medians")
    a = ap.parse_args()
    rd = a.run_dir
    src = rd / "ckpt" / "training.jsonl"
    rows = load_jsonl(src) if src.exists() else load_console(rd)
    if not rows:
        raise SystemExit(f"No step records in {src} or {rd}/logs/*.log")

    prev = None
    for r in rows:
        t = ts(r.get("timestamp"))
        r["step_seconds"] = round(t - prev, 3) if (t is not None and prev is not None) else None
        prev = t
    with (rd / "steps.csv").open("w", newline="") as f:
        cols = ["step", "timestamp", "loss", "tps", "tps_per_gpu", "step_seconds", "mem", "num_tokens_per_step",
                "num_label_tokens", "grad_norm", "lr"]
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    steady = rows[a.warmup_steps:] or rows
    med = lambda k: statistics.median(v) if (v := [num(r.get(k)) for r in steady if num(r.get(k)) is not None]) else None
    meta = {}
    if (rd / "run_meta.yaml").exists():
        for line in (rd / "run_meta.yaml").read_text().splitlines():
            if ":" in line:
                k, v = line.split(":", 1)
                meta[k.strip()] = v.strip()
    val = []
    vpath = rd / "ckpt" / "validation.jsonl"
    if vpath.exists():
        for line in vpath.read_text().splitlines():
            try:
                r = json.loads(line)
                if "val_loss" in r:
                    val.append({"step": r["step"], "val_loss": r["val_loss"]})
            except json.JSONDecodeError:
                pass
    nodes_ok = [json.loads(p.read_text()) for p in (rd / "nodes").glob("*.exit.json")] if (rd / "nodes").exists() else []
    summary = {
        "run": rd.name, **meta, "steps_logged": len(rows), "warmup_excluded": min(a.warmup_steps, len(rows) - 1),
        "median_tps": med("tps"), "median_tps_per_gpu": med("tps_per_gpu"), "median_step_seconds": med("step_seconds"),
        "peak_mem_gib": max((num(r.get("mem")) or 0) for r in rows),
        "first_loss": num(rows[0].get("loss")), "last_loss": num(rows[-1].get("loss")),
        "validation": val, "node_exits": nodes_ok,
        "tps_samples": [num(r.get("tps")) for r in steady],
    }
    (rd / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"{rd.name}: {len(rows)} steps | median tps={summary['median_tps']} | median step="
          f"{summary['median_step_seconds']}s | loss {summary['first_loss']} -> {summary['last_loss']} | "
          f"peak mem {summary['peak_mem_gib']} GiB" + (f" | val {val[-1]['val_loss']:.4f}" if val else ""))


if __name__ == "__main__":
    main()
