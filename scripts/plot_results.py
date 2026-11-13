#!/usr/bin/env python3
"""Charts for the demo, built only from measured files.

    python scripts/plot_results.py --runs results/runs/single-1n-r* results/runs/replicate-2n-r* \
        --scores results/eval/scores.json --nccl results/nccl/nccl_report.json

Runs are grouped by (mode, nodes); repeats (r1, r2, r3) give the error bars. Runs are only
compared if they share the same global batch size and sequence length.
"""
import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def load_runs(dirs):
    runs = []
    for d in map(Path, dirs):
        s = d / "summary.json"
        if not s.exists():
            print(f"skip {d}: no summary.json (run scripts/parse_train_log.py --run-dir {d})")
            continue
        r = json.loads(s.read_text())
        r["dir"] = d
        r["nodes"] = int(float(r.get("nodes", 1)))
        r["gbs"] = int(float(r.get("gbs", 0)))
        r["seq_length"] = int(float(r.get("seq_length", 0)))
        runs.append(r)
    return runs


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="*", default=[])
    ap.add_argument("--scores", default=None)
    ap.add_argument("--nccl", default=None)
    ap.add_argument("--out", default="results/charts")
    ap.add_argument("--force", action="store_true", help="plot even if gbs/seq_length differ")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    summary = {}

    runs = load_runs(a.runs)
    if runs:
        sigs = {(r["gbs"], r["seq_length"]) for r in runs}
        if len(sigs) > 1 and not a.force:
            raise SystemExit(f"Runs are not comparable (gbs, seq_length) = {sigs}. Use matching runs or --force.")
        groups = defaultdict(list)
        for r in runs:
            if r.get("median_tps"):
                groups[(r.get("mode", "?"), r["nodes"])].append(r)
        keys = sorted(groups, key=lambda k: (k[1], k[0]))
        labels = [f"{m}\n{n} node{'s' if n > 1 else ''}" for m, n in keys]
        tps = [[g["median_tps"] for g in groups[k]] for k in keys]
        mid = [statistics.median(v) for v in tps]
        err = [[m - min(v) for m, v in zip(mid, tps)], [max(v) - m for m, v in zip(mid, tps)]]

        fig, ax = plt.subplots(figsize=(8, 5))
        bars = ax.bar(labels, mid, yerr=err, capsize=6, color=["#7f8c8d" if k[1] == 1 else "#76b900" for k in keys])
        for b, v, k in zip(bars, mid, keys):
            ax.text(b.get_x() + b.get_width() / 2, v, f"{v:,.0f}\n(n={len(groups[k])})", ha="center", va="bottom")
        ax.set(ylabel="Training tokens / second (median, steady state)",
               title=f"FinScale Nemotron-30B LoRA throughput (GBS={runs[0]['gbs']}, seq={runs[0]['seq_length']})")
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(out / "throughput.png", dpi=160)
        plt.close(fig)

        base = [statistics.median(v) for k, v in zip(keys, tps) if k[1] == 1]
        if base:
            b0 = base[0]
            for k, m in zip(keys, mid):
                summary[f"{k[0]}-{k[1]}n"] = {"median_tps": round(m, 1), "speedup_vs_1node": round(m / b0, 3),
                                              "scaling_efficiency_pct": round(100 * m / b0 / k[1], 1),
                                              "repeats": len(groups[k])}

        fig, ax = plt.subplots(figsize=(9, 5))
        for r in runs:
            steps = list(csv.DictReader((r["dir"] / "steps.csv").open())) if (r["dir"] / "steps.csv").exists() else []
            xs = [int(s["step"]) for s in steps if s.get("loss")]
            ys = [float(s["loss"]) for s in steps if s.get("loss")]
            if xs:
                ax.plot(xs, ys, label=f"{r['run']} ({r.get('mode')}, {r['nodes']}n)", alpha=0.8)
        ax.set(xlabel="Optimizer step", ylabel="Training loss", title="Loss curves (same data, same global batch)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(out / "loss_curves.png", dpi=160)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(8, 5))
        mem = [statistics.median(g["peak_mem_gib"] for g in groups[k]) for k in keys]
        ax.bar(labels, mem, color="#2e86c1")
        ax.set(ylabel="Peak GPU memory per node (GiB, AutoModel)", title="Memory per node")
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(out / "memory_per_node.png", dpi=160)
        plt.close(fig)

    if a.scores and Path(a.scores).exists():
        sc = json.loads(Path(a.scores).read_text())
        metrics = ["json_valid_rate", "class_macro_f1", "severity_accuracy", "quote_fidelity"]
        systems = list(sc["systems"])
        fig, ax = plt.subplots(figsize=(9, 5))
        w = 0.8 / max(1, len(systems))
        for i, s in enumerate(systems):
            vals = [sc["systems"][s][m] for m in metrics]
            xs = [j + i * w for j in range(len(metrics))]
            ax.bar(xs, vals, w, label=s)
            for x, v in zip(xs, vals):
                ax.text(x, v, f"{v:.2f}", ha="center", va="bottom", fontsize=8)
        ax.set_xticks([j + w * (len(systems) - 1) / 2 for j in range(len(metrics))], metrics)
        ax.set(ylim=(0, 1.1), title=f"Held-out companies: base vs fine-tuned ({sc['gold_source']}, n={sc['n_gold']})")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(out / "eval_quality.png", dpi=160)
        plt.close(fig)
        summary["eval"] = {s: {m: sc["systems"][s][m] for m in metrics} for s in systems}

    if a.nccl and Path(a.nccl).exists():
        rep = json.loads(Path(a.nccl).read_text())
        bw = rep["bandwidth"]
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.plot([b["size_mb"] for b in bw], [b["busbw_GBps"] for b in bw], marker="o", color="#76b900")
        ax.set_xscale("log")
        ax.set(xlabel="All-reduce message size (MB)", ylabel="Bus bandwidth (GB/s)",
               title=f"NCCL across {len(rep['distinct_hosts'])} hosts: {', '.join(rep['distinct_hosts'])}")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(out / "nccl_bandwidth.png", dpi=160)
        plt.close(fig)
        summary["nccl_peak_busbw_GBps"] = max(b["busbw_GBps"] for b in bw)

    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"charts in {out}")


if __name__ == "__main__":
    main()
