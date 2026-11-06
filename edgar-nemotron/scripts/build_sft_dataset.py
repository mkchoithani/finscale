#!/usr/bin/env python3
"""Generate filing-grounded message JSONL. Labels come from Nemotron via NVIDIA API (or extractive fallback with --no-api)."""
import argparse, json, os, random, re, sys, time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

MODEL = "nvidia/nemotron-3.5-lightning-30b-a3b"
BASE_URL = "https://integrate.api.nvidia.com/v1"
WANTED_ITEMS = ["1", "1A", "3", "7", "7A"]
MIN_BODY = 1500
MAX_BODY = 150_000
MAX_CONTEXT = 12000
MAX_TOKENS = 1024
_client = None

SYSTEM_SUMMARY = ("You are a careful financial filing assistant. Ground every statement in the "
                  "supplied source; do not invent facts.")
SYSTEM_QUOTE = "Answer from the SEC filing text only."


def get_client():
    global _client
    if _client is None:
        from openai import OpenAI
        key = os.environ.get("NVIDIA_API_KEY")
        if not key:
            raise SystemExit("NVIDIA_API_KEY is not set. Run: export NVIDIA_API_KEY='...'")
        _client = OpenAI(base_url=BASE_URL, api_key=key, timeout=120, max_retries=0)
    return _client


def nemotron_label(system, prompt, retries=5):
    """Return (text, reason). text is None on failure; reason says why."""
    reason = "api error"
    for attempt in range(retries):
        try:
            r = get_client().chat.completions.create(
                model=MODEL,
                messages=[{"role": "system", "content": system},
                          {"role": "user", "content": prompt}],
                temperature=0.1, top_p=0.95, max_tokens=MAX_TOKENS,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}})
            ch = r.choices[0]
            text = (ch.message.content or "").strip()
            if text and ch.finish_reason == "stop":
                return text, None
            return None, f"empty/truncated (finish_reason={ch.finish_reason})"
        except Exception as e:
            status = getattr(e, "status_code", None)
            reason = f"api error {status}" if status else f"api error: {str(e)[:60]}"
            if status == 429:
                retry_after = None
                resp = getattr(e, "response", None)
                if resp is not None:
                    try:
                        retry_after = float(resp.headers.get("retry-after", ""))
                    except (TypeError, ValueError):
                        pass
                wait = retry_after if retry_after else min(60, 10 * (2 ** attempt)) + random.random() * 3
            else:
                wait = 2 * (attempt + 1)
            print(f"  API error (attempt {attempt+1}/{retries}, status={status}): waiting {wait:.0f}s",
                  file=sys.stderr)
            if attempt < retries - 1:
                time.sleep(wait)
    return None, reason


# [ \t] instead of \s so a match can never span lines (page header 'Item 1A' + real heading on next line).
HEAD_RE = re.compile(
    r"(?im)^[ \t]*item[ \t]+(\d{1,2}[a-c]?)[ \t]*[\.\:\-\u2013\u2014]?[ \t]*([^\n]{0,160})$")

# Titles for ALL items. Used to find section boundaries; only WANTED_ITEMS are kept.
# Matched against the title with all whitespace removed (fixes 'B USINESS', 'RIS K FACTORS').
BOUNDARY = {
    "1": r"business",
    "1A": r"riskfactors",
    "1B": r"unresolvedstaff",
    "1C": r"cybersecurity",
    "2": r"properties",
    "3": r"legalproceedings",
    "4": r"minesafety",
    "5": r"marketfor",
    "6": r"(\[?reserved|selectedfinancial)",
    "7": r"management.{0,3}sdiscussion",
    "7A": r"quantitativeandqualitative",
    "8": r"financialstatements",
    "9": r"changesinanddisagreements",
    "9A": r"controlsandprocedures",
    "9B": r"otherinformation",
    "9C": r"disclosureregardingforeign",
    "10": r"directors",
    "11": r"executivecompensation",
    "12": r"securityownership",
    "13": r"certainrelationships",
    "14": r"principalaccountant",
    "15": r"exhibit",
    "16": r"form10-?ksummary",
}

PAGE_HEADER_RE = re.compile(r"item\s+\d{1,2}[a-c]?(\s*,\s*\d{1,2}[a-c]?)*", re.I)


def canon_item(num, rest):
    """Return 'Item N' if (num, rest) is a real heading for any known item, else None."""
    num = num.upper()
    if num not in BOUNDARY:
        return None
    rest = rest.strip()
    squashed = re.sub(r"\s+", "", rest).lower()
    if not re.match(BOUNDARY[num], squashed):
        return None
    body = rest.rstrip(" .")          # allow a single trailing period
    # reject narrative: too long, or contains sentence-internal periods
    if len(body) > 120 or "." in body:
        return None
    return f"Item {num}"


def clean(body):
    lines = []
    for ln in body.splitlines():
        s = ln.strip()
        if re.fullmatch(r"\d{1,3}", s):  # bare page numbers
            continue
        if s.lower() == "table of contents":
            continue
        if PAGE_HEADER_RE.fullmatch(s):  # running page headers: 'Item 1A', 'Item 2, 3, 4'
            continue
        lines.append(ln)
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


def sections(text):
    """Return [(heading, body)] choosing, per wanted Item, the longest real body.

    Boundaries come from ALL canonical item headings (1..16); only WANTED_ITEMS
    are returned. Bodies outside [MIN_BODY, MAX_BODY] are dropped.
    """
    ms = [m for m in HEAD_RE.finditer(text) if canon_item(m.group(1), m.group(2))]
    best = {}
    for n, m in enumerate(ms):
        item = m.group(1).upper()
        if item not in WANTED_ITEMS:
            continue
        # body ends at the next canonical heading of a DIFFERENT item
        end = len(text)
        for k in range(n + 1, len(ms)):
            if ms[k].group(1).upper() != item:
                end = ms[k].start()
                break
        body = clean(text[m.end():end])
        if not (MIN_BODY <= len(body) <= MAX_BODY):
            continue
        heading = canon_item(m.group(1), m.group(2))
        if item not in best or len(body) > len(best[item][1]):
            best[item] = (heading, body)
    return [best[i] for i in WANTED_ITEMS if i in best]


def verbatim_ok(answer, source):
    norm = lambda s: re.sub(r"\s+", " ", s).lower()
    quoted = re.findall(r'["\u201c]([^"\u201d]{25,})["\u201d]', answer)
    return bool(quoted) and all(norm(q) in norm(source) for q in quoted)


def make_prompts(ticker, date, heading, context):
    summ = (f"Summarize {ticker}'s filing section {heading} (filed {date}) using only the source. "
            f"Write 3-5 sentences of plain prose (max 150 words). No headings, no bullet points, no markdown. "
            f"End with one sentence noting what the source does not cover.\n\nSource section:\n{context}")
    quote = (f"Identify the first substantive disclosed detail in {ticker}'s section {heading}. "
             f"Reply with exactly one sentence copied verbatim from the source, in double quotes, "
             f"followed by one short sentence explaining it.\n\nSource section:\n{context}")
    return [(summ, SYSTEM_SUMMARY, "nemotron-summary"), (quote, SYSTEM_QUOTE, "nemotron-quote")]


def label_one(args):
    """Return (answer, drop_reason)."""
    prompt, system, kind, use_api, context = args
    if use_api:
        ans, reason = nemotron_label(system, prompt)
        if not ans:
            return None, reason
        if kind == "nemotron-quote" and not verbatim_ok(ans, context):
            return None, "quote not verbatim"
        return ans, None
    lead = re.split(r"(?<=[.!?])\s+", context)[0][:1200].strip()
    return (lead, None) if lead else (None, "empty context")


def build(filing, max_sections, use_api, workers):
    ticker = filing.get("ticker", "company")
    date = filing.get("filing_date", "unknown")
    jobs, metas = [], []
    for heading, body in sections(filing.get("text", ""))[:max_sections]:
        context = body[:MAX_CONTEXT]
        for prompt, system, kind in make_prompts(ticker, date, heading, context):
            k = kind if use_api else ("extractive-lead-pseudo-summary" if kind == "nemotron-summary" else "extractive")
            jobs.append((prompt, system, kind, use_api, context))
            metas.append((prompt, system, k, heading))
    with ThreadPoolExecutor(max_workers=workers) as ex:
        results = list(ex.map(label_one, jobs))
    out, reasons = [], Counter()
    for (prompt, system, k, heading), (ans, reason) in zip(metas, results):
        if not ans:
            reasons[reason or "unknown"] += 1
            continue
        out.append({"messages": [{"role": "system", "content": system},
                                 {"role": "user", "content": prompt},
                                 {"role": "assistant", "content": ans}],
                    "metadata": {"ticker": ticker, "filing_date": date,
                                 "accession": filing.get("accession"),
                                 "section": heading, "label_type": k}})
    dropped = sum(reasons.values())
    detail = f" ({', '.join(f'{n}x {r}' for r, n in reasons.items())})" if dropped else ""
    print(f"{ticker} {date}: {len(out)} kept, {dropped} dropped{detail}, {len(metas)//2} sections")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", type=Path, default=Path("data/raw/edgar_10k.jsonl"))
    ap.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    ap.add_argument("--val-fraction", type=float, default=.1)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--max-sections", type=int, default=5)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--no-api", action="store_true")
    a = ap.parse_args()
    if not 0 <= a.val_fraction < 1:
        ap.error("--val-fraction must be in [0, 1)")
    use_api = not a.no_api
    rows = []
    with a.input.open(encoding="utf-8") as src:
        for n, line in enumerate(src, 1):
            if line.strip():
                try:
                    rows.extend(build(json.loads(line), a.max_sections, use_api, a.workers))
                except (json.JSONDecodeError, TypeError) as e:
                    raise SystemExit(f"Bad JSONL line {n}: {e}")
    if not rows:
        raise SystemExit("No examples produced. Check input and section headings.")
    random.Random(a.seed).shuffle(rows)
    nv = int(len(rows) * a.val_fraction)
    a.output_dir.mkdir(parents=True, exist_ok=True)
    for name, data in (("train", rows[nv:]), ("val", rows[:nv])):
        if name == "train" and not data:
            raise SystemExit("Training split is empty")
        path = a.output_dir / f"{name}.jsonl"
        with path.open("w", encoding="utf-8") as f:
            for row in data:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"{path}: {len(data)} examples")
    if use_api:
        print(f"Labels generated by {MODEL}. Spot-check a few rows before training.")
    else:
        print("WARNING: labels are extractive opening sentences, not human-validated; inspect before training.")


if __name__ == "__main__":
    main()
