#!/usr/bin/env python3
"""Re-split train/val by company so validation is actually held out.

build_sft_dataset.py makes TWO examples (summary + quote) from the SAME section
text and then shuffles rows, so the same 10-K context lands in both train and
val. That leaks and inflates validation results. This groups by ticker.
"""
import argparse
import json
import random
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--inputs", nargs="+", type=Path,
                    default=[Path("data/processed/train.jsonl"), Path("data/processed/val.jsonl")])
    ap.add_argument("--output-dir", type=Path, default=Path("data/processed/by_ticker"))
    ap.add_argument("--val-tickers", nargs="*", default=None, help="Explicit held-out tickers")
    ap.add_argument("--val-fraction", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=17)
    a = ap.parse_args()

    rows = [json.loads(l) for p in a.inputs for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
    tickers = sorted({r.get("metadata", {}).get("ticker", "UNKNOWN") for r in rows})
    if a.val_tickers:
        val = {t.upper() for t in a.val_tickers}
    else:
        shuffled = tickers[:]
        random.Random(a.seed).shuffle(shuffled)
        val = set(shuffled[:max(1, round(len(shuffled) * a.val_fraction))])
    a.output_dir.mkdir(parents=True, exist_ok=True)
    split = {"train": [r for r in rows if r.get("metadata", {}).get("ticker") not in val],
             "val": [r for r in rows if r.get("metadata", {}).get("ticker") in val]}
    for name, data in split.items():
        with (a.output_dir / f"{name}.jsonl").open("w", encoding="utf-8") as f:
            for r in data:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"{name}: {len(data)} rows")
    print("held-out tickers:", " ".join(sorted(val)))


if __name__ == "__main__":
    main()
