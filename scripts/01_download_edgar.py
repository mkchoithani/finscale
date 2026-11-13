#!/usr/bin/env python3
"""Download the latest 10-K filings from SEC EDGAR and store them as plain text.

SEC requires a real contact in the User-Agent. Set it once:
    export SEC_USER_AGENT="FinScale Hackathon your.name@yourdomain.com"

Example:
    python scripts/01_download_edgar.py                      # 30 default companies, 1 filing each
    python scripts/01_download_edgar.py --tickers AAPL JPM --filings-per-company 2

Resumable: companies already present in the output file are skipped.
"""
import argparse
import json
import os
import re
import sys
import time
from html.parser import HTMLParser
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from finscale.common import DEFAULT_TICKERS, read_jsonl  # noqa: E402

TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
OLDER_URL = "https://data.sec.gov/submissions/{name}"
ARCHIVE_URL ="https://www.sec.gov/Archives/edgar/data/{cik_int}/{acc}/{doc}"

BLOCK_TAGS = {"p", "div", "br", "tr", "li", "h1", "h2", "h3", "h4", "h5", "h6", "table", "section", "td"}
SKIP_TAGS = {"script", "style", "head", "ix:header", "title"}


class TextExtractor(HTMLParser):
    """HTML -> text with line breaks at block elements; drops hidden iXBRL headers."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts, self.skip = [], 0

    def handle_starttag(self, tag, attrs):
        tag = tag.lower()
        if tag in SKIP_TAGS:
            self.skip += 1
        elif tag in BLOCK_TAGS:
            self.parts.append("\n" if tag != "td" else " ")

    def handle_endtag(self, tag):
        tag = tag.lower()
        if tag in SKIP_TAGS:
            self.skip = max(0, self.skip - 1)
        elif tag in BLOCK_TAGS and tag != "td":
            self.parts.append("\n")

    def handle_data(self, data):
        if not self.skip:
            self.parts.append(data)


def html_to_text(html: str) -> str:
    ex = TextExtractor()
    ex.feed(html)
    text = "".join(ex.parts).replace("\xa0", " ")
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def get(session, url, pause, retries=4):
    for attempt in range(retries):
        r = session.get(url, timeout=60)
        if r.status_code == 200:
            time.sleep(pause)
            return r
        if r.status_code in (403, 429, 500, 502, 503):
            time.sleep(2 ** attempt + pause)
            continue
        r.raise_for_status()
    raise RuntimeError(f"GET failed after {retries} tries: {url} (last status {r.status_code})")


def find_10k(session, cik, n, pause):
    """Newest-first list of (accession, primary document, filing date) for 10-K filings.
    The submissions file only holds the most recent ~1,000 filings; companies that file many
    other forms (e.g. XOM) can push their 10-K into the older pages listed under filings.files."""
    sub = get(session, SUBMISSIONS_URL.format(cik=cik), pause).json()
    pages = [sub["filings"]["recent"]]
    older = [f["name"] for f in sub["filings"].get("files", [])]
    found = []
    while True:
        page = pages.pop(0)
        for i, form in enumerate(page["form"]):
            if form == "10-K" and page["primaryDocument"][i]:
                found.append((page["accessionNumber"][i], page["primaryDocument"][i], page["filingDate"][i]))
        if len(found) >= n or not older:
            break
        pages.append(get(session, OLDER_URL.format(name=older.pop(0)), pause).json())
    found.sort(key=lambda x: x[2], reverse=True)
    return found[:n]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tickers", nargs="+", default=None)
    ap.add_argument("--filings-per-company", type=int, default=1)
    ap.add_argument("--output", type=Path, default=Path("data/raw/filings.jsonl"))
    ap.add_argument("--pause", type=float, default=0.2, help="seconds between SEC requests (keep >=0.1)")
    a = ap.parse_args()

    ua = os.environ.get("SEC_USER_AGENT", "").strip()
    if "@" not in ua:
        sys.exit('Set SEC_USER_AGENT to "<project> <real contact email>" (SEC fair-access rule).')
    if a.pause < 0.1:
        sys.exit("--pause must be >= 0.1 (SEC allows at most 10 requests/second)")

    tickers = [t.upper() for t in (a.tickers or DEFAULT_TICKERS)]
    done = {}
    for r in read_jsonl(a.output):
        done[r["ticker"]] = done.get(r["ticker"], 0) + 1
    todo = [t for t in tickers if done.get(t, 0) < a.filings_per_company]
    print(f"{len(tickers)} companies requested, {len(tickers) - len(todo)} already downloaded")

    s = requests.Session()
    s.headers.update({"User-Agent": ua, "Accept-Encoding": "gzip, deflate"})
    lookup = {v["ticker"].upper(): (str(v["cik_str"]).zfill(10), v["title"])
              for v in get(s, TICKERS_URL, a.pause).json().values()}

    a.output.parent.mkdir(parents=True, exist_ok=True)
    ok = failed = 0
    with a.output.open("a", encoding="utf-8") as out:
        for t in todo:
            if t not in lookup:
                print(f"  {t}: unknown ticker, skipped", file=sys.stderr)
                failed += 1
                continue
            cik, company = lookup[t]
            try:
                filings = find_10k(s, cik, a.filings_per_company, a.pause)
                if not filings:
                    print(f"  {t}: no 10-K found (checked recent and older filing pages)", file=sys.stderr)
                    failed += 1
                    continue
                for acc, doc, filed in filings[done.get(t, 0):]:
                    url = ARCHIVE_URL.format(cik_int=int(cik), acc=acc.replace("-", ""), doc=doc)
                    text = html_to_text(get(s, url, a.pause).text)
                    rec = {"ticker": t, "cik": cik, "company": company, "form": "10-K",
                           "filing_date": filed, "accession": acc, "url": url, "text": text}
                    out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    out.flush()
                    ok += 1
                    print(f"  {t} {rec['filing_date']}: {len(text):,} chars")
            except Exception as e:  # keep going: one bad company must not stop the run
                print(f"  {t}: FAILED {e}", file=sys.stderr)
                failed += 1
    print(f"Done: {ok} new filing(s), {failed} problem(s) -> {a.output}")


if __name__ == "__main__":
    main()
