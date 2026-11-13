#!/usr/bin/env python3
"""Build the training / validation / test sets from labeled passages.

* Split by COMPANY (ticker), so no filing text appears in more than one split.
* train.jsonl / val.jsonl are chat rows {"messages": [system, user, assistant]} for NeMo AutoModel.
* test.jsonl keeps the passage + teacher label for evaluation (never trained on).
* gold_review.csv: a class-balanced sample of the test set for humans to check/correct.
* manifest.json: counts, class balance and SHA-256 of every file (proves which data was used).

    python scripts/04_build_sft.py
    python scripts/04_build_sft.py --test-tickers JPM NVDA XOM --val-tickers MSFT WMT
"""
import argparse
import csv
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from finscale.common import canonical_answer, read_jsonl, sha256_file, student_messages, write_jsonl  # noqa: E402


def pick_tickers(tickers, explicit, frac, rng, taken):
    if explicit:
        return {t.upper() for t in explicit}
    pool = sorted(set(tickers) - taken)
    rng.shuffle(pool)
    return set(pool[: max(1, round(len(tickers) * frac))])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--passages", type=Path, default=Path("data/passages.jsonl"))
    ap.add_argument("--labels", type=Path, default=Path("data/labels.jsonl"))
    ap.add_argument("--out-dir", type=Path, default=Path("data/sft"))
    ap.add_argument("--test-tickers", nargs="*")
    ap.add_argument("--val-tickers", nargs="*")
    ap.add_argument("--test-frac", type=float, default=0.15)
    ap.add_argument("--val-frac", type=float, default=0.10)
    ap.add_argument("--max-other-frac", type=float, default=0.25,
                    help="cap the share of 'other' in TRAIN so the model does not learn to always say other")
    ap.add_argument("--gold-size", type=int, default=60)
    ap.add_argument("--seed", type=int, default=17)
    a = ap.parse_args()

    passages = {p["id"]: p for p in read_jsonl(a.passages)}
    labels = [r for r in read_jsonl(a.labels) if r.get("status") == "ok" and r["id"] in passages]
    if not labels:
        sys.exit("No valid labels. Run scripts/03_label_risk.py first.")
    if any(r.get("teacher") == "dry-run" for r in labels):
        print("WARNING: dry-run labels present. Fine for a pipeline test, NOT for real training.", file=sys.stderr)

    rows = [{**passages[r["id"]], "label": r["label"], "teacher": r.get("teacher")} for r in labels]
    rng = random.Random(a.seed)
    tickers = sorted({r["ticker"] for r in rows})
    if len(tickers) < 3:
        sys.exit(f"Need at least 3 companies for a company-level split, have {tickers}")
    test_t = pick_tickers(tickers, a.test_tickers, a.test_frac, rng, set())
    val_t = pick_tickers(tickers, a.val_tickers, a.val_frac, rng, test_t)
    if test_t & val_t:
        sys.exit(f"Test and val tickers overlap: {test_t & val_t}")
    split = defaultdict(list)
    for r in rows:
        split["test" if r["ticker"] in test_t else "val" if r["ticker"] in val_t else "train"].append(r)

    # Cap 'other' in train only (val/test keep the natural distribution).
    train = split["train"]
    other = [r for r in train if r["label"]["risk_class"] == "other"]
    rest = [r for r in train if r["label"]["risk_class"] != "other"]
    max_other = int(a.max_other_frac * len(rest) / max(1e-9, 1 - a.max_other_frac))
    if len(other) > max_other:
        rng.shuffle(other)
        other = other[:max_other]
    split["train"] = rest + other
    for k in split:
        rng.shuffle(split[k])

    out = a.out_dir
    # Only "messages": extra keys could be misread by the dataset loader.
    to_chat = lambda r: {"messages": student_messages(r, canonical_answer(r["label"]))}
    write_jsonl(out / "train.jsonl", map(to_chat, split["train"]))
    write_jsonl(out / "val.jsonl", map(to_chat, split["val"]))
    write_jsonl(out / "test.jsonl", ({k: r[k] for k in ("id", "ticker", "company", "filing_date", "item",
                                                       "item_title", "text", "label", "teacher")}
                                     for r in split["test"]))

    # Gold review sample: round-robin over classes so rare classes are represented.
    by_cls = defaultdict(list)
    for r in split["test"]:
        by_cls[r["label"]["risk_class"]].append(r)
    gold = []
    while len(gold) < a.gold_size and any(by_cls.values()):
        for c in sorted(by_cls):
            if by_cls[c] and len(gold) < a.gold_size:
                gold.append(by_cls[c].pop())
    with (out / "gold_review.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "ticker", "item", "passage_start", "teacher_class", "teacher_severity",
                    "teacher_quote", "human_class", "human_severity", "notes"])
        for r in gold:
            L = r["label"]
            w.writerow([r["id"], r["ticker"], r["item"], r["text"][:600].replace("\n", " "), L["risk_class"],
                        L["severity"], L["evidence_quote"], "", "", ""])

    manifest = {
        "seed": a.seed,
        "tickers": {"train": sorted({r["ticker"] for r in split["train"]}), "val": sorted(val_t),
                    "test": sorted(test_t)},
        "counts": {k: len(v) for k, v in split.items()},
        "class_distribution": {k: dict(Counter(r["label"]["risk_class"] for r in v)) for k, v in split.items()},
        "severity_distribution": {k: dict(Counter(r["label"]["severity"] for r in v)) for k, v in split.items()},
        "teachers": dict(Counter(r["teacher"] for r in rows)),
        "sha256": {p.name: sha256_file(p) for p in sorted(out.glob("*.jsonl"))},
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps({k: manifest[k] for k in ("counts", "tickers")}, indent=2))
    print("train classes:", manifest["class_distribution"]["train"])
    print(f"Wrote {out}/train.jsonl, val.jsonl, test.jsonl, gold_review.csv ({len(gold)} rows), manifest.json")


if __name__ == "__main__":
    main()
