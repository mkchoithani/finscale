#!/usr/bin/env python3
"""Cut risk-relevant 10-K sections into short passages (one passage = one training example).

Sections used: Item 1A Risk Factors, 1C Cybersecurity, 3 Legal Proceedings,
7 MD&A, 7A Market Risk. Passages are ~4,000 characters (~1,000 tokens), so the
prompt + JSON answer fits easily in a 2,048-token training sequence and the
answer can never be truncated away.

Example:
    python scripts/02_make_passages.py --max-per-filing 40
"""
import argparse
import hashlib
import random
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from finscale.common import norm_text, read_jsonl, write_jsonl  # noqa: E402

TARGET_ITEMS = {"1A": "Risk Factors", "1C": "Cybersecurity", "3": "Legal Proceedings",
                "7": "Management's Discussion and Analysis", "7A": "Market Risk"}
# Any Item heading at the start of a line, e.g. "Item 1A. Risk Factors", "ITEM 7A —"
HEADING_RE = re.compile(r"^\s*item\s+(1a|1b|1c|1|2|3|4|5|6|7a|7|8|9a|9b|9c|9|10|11|12|13|14|15|16)\b\.?",
                        re.I | re.M)
NOISE_RE = re.compile(r"^(table of contents|\d{1,3}|page \d+|.{0,3})$", re.I)


def find_sections(text: str) -> dict[str, str]:
    """For each target item, keep the LONGEST span from its heading to the next heading of a
    DIFFERENT item. Consecutive headings of the same item are merged, because some filers (e.g. MSFT)
    repeat "Item 1A" as a page header on every page. The table of contents lists every item too,
    but those spans are only a few characters long, so the longest span wins."""
    heads = [(m.start(), m.group(1).upper()) for m in HEADING_RE.finditer(text)]
    best: dict[str, str] = {}
    i = 0
    while i < len(heads):
        pos, item = heads[i]
        j = i + 1
        while j < len(heads) and heads[j][1] == item:
            j += 1
        end = heads[j][0] if j < len(heads) else len(text)
        if item in TARGET_ITEMS and end - pos > len(best.get(item, "")):
            best[item] = text[pos:end]
        i = j
    return {k: v for k, v in best.items() if len(v) > 1500}


def clean_lines(span: str) -> list[str]:
    lines = [l.strip() for l in span.splitlines()]
    # drop the heading line, page headers like "PART I Item 1A", page numbers and other noise
    return [l for l in lines[1:] if l and not NOISE_RE.match(l)
            and not (len(l) <= 30 and re.search(r"\bitem\s+\d+[a-c]?\b", l, re.I))]


def chunk(lines: list[str], max_chars: int, min_chars: int) -> list[str]:
    out, buf = [], ""
    for para in lines:
        pieces = [para] if len(para) <= max_chars else re.split(r"(?<=[.;])\s+", para)
        for piece in pieces:
            if buf and len(buf) + len(piece) + 1 > max_chars:
                out.append(buf)
                buf = ""
            buf = f"{buf}\n{piece}" if buf else piece[:max_chars]
    if buf:
        out.append(buf)
    return [c for c in out if len(c) >= min_chars]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", type=Path, default=Path("data/raw/filings.jsonl"))
    ap.add_argument("--output", type=Path, default=Path("data/passages.jsonl"))
    ap.add_argument("--max-chars", type=int, default=4000)
    ap.add_argument("--min-chars", type=int, default=600)
    ap.add_argument("--max-per-filing", type=int, default=40)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()

    filings = read_jsonl(a.input)
    if not filings:
        sys.exit(f"No filings in {a.input}. Run scripts/01_download_edgar.py first.")
    rows, seen, per_item, missing = [], set(), Counter(), []
    for f in filings:
        sections = find_sections(f["text"])
        if not sections:
            missing.append(f["ticker"])
            continue
        rng = random.Random(f"{a.seed}-{f['ticker']}-{f['filing_date']}")
        pools = {}
        for item, span in sections.items():
            chunks = chunk(clean_lines(span), a.max_chars, a.min_chars)
            rng.shuffle(chunks)
            pools[item] = chunks
        # Round-robin across sections so Risk Factors does not crowd out the others.
        picked, order = [], sorted(pools)
        while len(picked) < a.max_per_filing and any(pools.values()):
            for item in order:
                if pools[item] and len(picked) < a.max_per_filing:
                    picked.append((item, pools[item].pop()))
        for k, (item, text) in enumerate(picked):
            h = hashlib.sha1(norm_text(text).encode()).hexdigest()
            if h in seen:
                continue
            seen.add(h)
            per_item[item] += 1
            rows.append({"id": f"{f['ticker']}-{f['filing_date']}-{item}-{k:03d}", "ticker": f["ticker"],
                         "company": f.get("company", ""), "filing_date": f["filing_date"], "item": item,
                         "item_title": TARGET_ITEMS[item], "text": text, "n_chars": len(text)})
    n = write_jsonl(a.output, rows)
    print(f"{n} passages from {len(filings)} filings -> {a.output}")
    print("per section:", dict(sorted(per_item.items())))
    if missing:
        print(f"WARNING: no sections found for {missing} (check their text layout)", file=sys.stderr)


if __name__ == "__main__":
    main()
