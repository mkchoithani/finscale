#!/usr/bin/env python3
"""Score prediction files against the test labels (teacher labels, or human-corrected gold).

    python scripts/08_score.py results/eval/pred-base.jsonl results/eval/pred-finetuned.jsonl
    python scripts/08_score.py results/eval/pred-*.jsonl --gold data/sft/gold_review.csv   # human-checked rows only

Metrics: JSON valid rate, risk-class accuracy and macro-F1, severity accuracy,
quote fidelity (quote really appears in the passage), median latency.
"""
import argparse
import csv
import json
import statistics
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from finscale.common import RISK_CLASSES, read_jsonl, validate_label, quote_in_passage  # noqa: E402


def macro_f1(pairs):
    f1s = []
    for c in RISK_CLASSES:
        tp = sum(1 for g, p in pairs if g == c and p == c)
        fp = sum(1 for g, p in pairs if g != c and p == c)
        fn = sum(1 for g, p in pairs if g == c and p != c)
        if tp + fn == 0:  # class absent from gold: skip, standard for small test sets
            continue
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn)
        f1s.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
    return sum(f1s) / len(f1s) if f1s else 0.0


def load_gold(test_rows, gold_csv):
    gold = {r["id"]: {"class": r["label"]["risk_class"], "severity": r["label"]["severity"], "text": r["text"]}
            for r in test_rows}
    if not gold_csv:
        return gold, "teacher labels"
    reviewed = {}
    with open(gold_csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["id"] in gold:
                g = dict(gold[row["id"]])
                g["class"] = (row.get("human_class") or row["teacher_class"]).strip().lower()
                g["severity"] = (row.get("human_severity") or row["teacher_severity"]).strip().lower()
                reviewed[row["id"]] = g
    return reviewed, f"human-reviewed gold ({len(reviewed)} rows)"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("preds", nargs="+")
    ap.add_argument("--test", default="data/sft/test.jsonl")
    ap.add_argument("--gold", default=None, help="gold_review.csv with human_class / human_severity filled in")
    ap.add_argument("--out", default="results/eval/scores.json")
    a = ap.parse_args()

    gold, source = load_gold(read_jsonl(a.test), a.gold)
    if not gold:
        sys.exit("No gold rows")
    report = {"gold_source": source, "n_gold": len(gold), "systems": {}}
    for path in a.preds:
        preds = {p["id"]: p for p in read_jsonl(path)}
        name = next(iter(preds.values()))["system"] if preds else Path(path).stem
        n = valid = schema_ok = quote_ok = 0
        cls_pairs, sev_hits, lat, confusion = [], 0, [], Counter()
        for gid, g in gold.items():
            p = preds.get(gid)
            n += 1
            if p is None:
                cls_pairs.append((g["class"], None))
                continue
            lat.append(p.get("latency_s", 0))
            obj = p.get("parsed")
            if isinstance(obj, dict):
                valid += 1
            pc = str(obj.get("risk_class", "")).strip().lower() if isinstance(obj, dict) else None
            ps = str(obj.get("severity", "")).strip().lower() if isinstance(obj, dict) else None
            cls_pairs.append((g["class"], pc))
            confusion[(g["class"], pc or "INVALID")] += 1
            sev_hits += int(ps == g["severity"])
            if isinstance(obj, dict):
                schema_ok += int(validate_label(obj, g["text"])[0] is not None)
                quote_ok += int(quote_in_passage(str(obj.get("evidence_quote", "")), g["text"]))
        report["systems"][name] = {
            "file": path, "n": n,
            "json_valid_rate": round(valid / n, 4),
            "fully_valid_rate": round(schema_ok / n, 4),
            "class_accuracy": round(sum(g == p for g, p in cls_pairs) / n, 4),
            "class_macro_f1": round(macro_f1(cls_pairs), 4),
            "severity_accuracy": round(sev_hits / n, 4),
            "quote_fidelity": round(quote_ok / n, 4),
            "median_latency_s": round(statistics.median(lat), 3) if lat else None,
            "confusion": {f"{g}->{p}": c for (g, p), c in sorted(confusion.items())},
        }
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    keys = ["json_valid_rate", "class_accuracy", "class_macro_f1", "severity_accuracy", "quote_fidelity",
            "median_latency_s"]
    print(f"Gold: {source}, n={len(gold)}")
    print(f"{'system':<14}" + "".join(f"{k:>19}" for k in keys))
    for name, s in report["systems"].items():
        print(f"{name:<14}" + "".join(f"{str(s[k]):>19}" for k in keys))
    md = ["| system | " + " | ".join(keys) + " |", "|---" * (len(keys) + 1) + "|"]
    md += [f"| {n} | " + " | ".join(str(s[k]) for k in keys) + " |" for n, s in report["systems"].items()]
    out.with_suffix(".md").write_text(f"Gold: {source}, n={len(gold)}\n\n" + "\n".join(md) + "\n")
    print(f"saved {out} and {out.with_suffix('.md')}")


if __name__ == "__main__":
    main()
