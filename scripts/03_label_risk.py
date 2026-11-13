#!/usr/bin/env python3
"""Label each passage with {risk_class, severity, evidence_quote, rationale} using a large
"teacher" model on the NVIDIA API (OpenAI-compatible). Every label is validated:
class/severity must be in the taxonomy and the quote must appear word for word in the passage.

    export NVIDIA_API_KEY=nvapi-...
    python scripts/03_label_risk.py --limit 20          # quick sample, inspect output first
    python scripts/03_label_risk.py --workers 8         # full run (resumable)
    python scripts/03_label_risk.py --list-models       # see which teacher models your key can use

--dry-run makes fake labels without the API, only to test the pipeline end to end.
"""
import argparse
import json
import os
import random
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from finscale.common import (RISK_CLASSES, SEVERITIES, TEACHER_SYSTEM, parse_json_answer,  # noqa: E402
                             read_jsonl, validate_label)

API_URL = "https://integrate.api.nvidia.com/v1"
DEFAULT_TEACHER = "nvidia/nemotron-3-ultra-550b-a55b"  # verified working with our key (super-120b gave 404)


def teacher_prompt(p):
    return (f"Company: {p['ticker']} ({p.get('company', '')}), 10-K filed {p['filing_date']}, "
            f"Item {p['item']} {p['item_title']}.\n\nPassage:\n{p['text']}\n\nReturn the JSON object.")


def make_client(base_url):
    from openai import OpenAI
    key = os.environ.get("NVIDIA_API_KEY", "").strip()
    if not key:
        sys.exit("Set NVIDIA_API_KEY")
    return OpenAI(base_url=base_url, api_key=key, timeout=120, max_retries=0)


def call(client, model, messages, max_tokens, temperature, no_think):
    kwargs = dict(model=model, messages=messages, max_tokens=max_tokens, temperature=temperature)
    if no_think:
        kwargs["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}
    try:
        r = client.chat.completions.create(**kwargs)
    except Exception as e:
        # Some endpoints reject chat_template_kwargs; retry once without it.
        if no_think and "chat_template_kwargs" in str(e):
            kwargs.pop("extra_body")
            r = client.chat.completions.create(**kwargs)
        else:
            raise
    return r.choices[0].message.content or ""


FATAL_CODES = {400, 401, 403, 404, 422}  # wrong model / key / request: retrying will not help


def label_one(client, args, p):
    msgs = [{"role": "system", "content": TEACHER_SYSTEM}, {"role": "user", "content": teacher_prompt(p)}]
    reason, raw = "", ""
    for attempt in range(args.max_attempts):
        try:
            raw = call(client, args.model, msgs, args.max_tokens, args.temperature, not args.allow_thinking)
        except Exception as e:
            code = getattr(e, "status_code", None)
            reason = f"api error {code}: {str(e)[:200]}"
            if code in FATAL_CODES:
                return {"id": p["id"], "status": "api_error", "code": code, "reason": reason, "teacher": args.model}
            time.sleep(min(60, 2 ** attempt * 2) + random.random())  # rate limit / network: back off
            continue
        label, reason = validate_label(parse_json_answer(raw), p["text"])
        if label:
            return {"id": p["id"], "status": "ok", "label": label, "teacher": args.model, "attempts": attempt + 1}
        # Tell the teacher what was wrong and ask again.
        msgs = msgs[:2] + [{"role": "assistant", "content": raw[:2000]},
                           {"role": "user", "content": f"Invalid: {reason}. Return only the corrected JSON object; "
                                                       "evidence_quote must be copied exactly from the passage."}]
    status = "api_error" if reason.startswith("api error") else "invalid"
    return {"id": p["id"], "status": status, "reason": reason, "raw": raw[:500], "teacher": args.model}


PROBE_MODELS = [
    "nvidia/nemotron-3-super-120b-a12b", "nvidia/nemotron-3-ultra-550b-a55b",
    "nvidia/llama-3.1-nemotron-ultra-253b-v1", "nvidia/llama-3.1-nemotron-70b-instruct",
    "nvidia/nemotron-3.5-lightning-30b-a3b", "nvidia/nemotron-nano-3-30b-a3b",
    "meta/llama-3.3-70b-instruct", "meta/llama-3.1-405b-instruct",
]


def probe(base_url, extra):
    """Send one tiny request to each candidate model and report which ones this key can use."""
    client = make_client(base_url)
    try:
        listed = {m.id for m in client.models.list().data}
    except Exception:
        listed = set()
    print("model".ljust(45), "result")
    for m in dict.fromkeys(([extra] if extra else []) + PROBE_MODELS):
        t = time.time()
        try:
            out = call(client, m, [{"role": "user", "content": 'Reply with exactly: {"ok": true}'}], 30, 0.0, True)
            res = f"OK ({time.time() - t:.1f}s): {out.strip()[:40]!r}"
        except Exception as e:
            res = f"FAIL {getattr(e, 'status_code', '')} {str(e)[:80]}"
        print(m.ljust(45), res, "" if m in listed or not listed else "(not in model list)")
    print("\nPick the largest model that says OK, then: python scripts/03_label_risk.py --model <name> --limit 20")


def dry_label(p):
    """Fake but schema-valid label (first long sentence as quote). Pipeline testing only."""
    sent = next((s.strip() for s in p["text"].replace("\n", " ").split(". ") if len(s.strip()) > 40), p["text"][:80])
    rng = random.Random(p["id"])
    return {"id": p["id"], "status": "ok", "teacher": "dry-run",
            "label": {"risk_class": rng.choice(RISK_CLASSES), "severity": rng.choice(SEVERITIES),
                      "evidence_quote": sent[:200], "rationale": "dry-run placeholder"}}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", type=Path, default=Path("data/passages.jsonl"))
    ap.add_argument("--output", type=Path, default=Path("data/labels.jsonl"))
    ap.add_argument("--model", default=os.environ.get("FINSCALE_TEACHER", DEFAULT_TEACHER))
    ap.add_argument("--base-url", default=API_URL)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit", type=int, default=0, help="label only the first N unlabeled passages")
    ap.add_argument("--max-attempts", type=int, default=3)
    ap.add_argument("--max-tokens", type=int, default=600)
    ap.add_argument("--temperature", type=float, default=0.1)
    ap.add_argument("--allow-thinking", action="store_true", help="let the teacher reason (slower)")
    ap.add_argument("--retry-invalid", action="store_true", help="re-label passages previously marked invalid")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--list-models", action="store_true")
    ap.add_argument("--probe", action="store_true", help="test which teacher models your key can call")
    a = ap.parse_args()

    if a.probe:
        probe(a.base_url, os.environ.get("FINSCALE_TEACHER"))
        return
    if a.list_models:
        for m in make_client(a.base_url).models.list().data:
            if "nemotron" in m.id or "llama" in m.id:
                print(m.id)
        return

    passages = read_jsonl(a.input)
    if not passages:
        sys.exit(f"No passages in {a.input}. Run scripts/02_make_passages.py first.")
    previous = {r["id"]: r for r in read_jsonl(a.output)}
    # Keep good labels; keep content-invalid rows unless --retry-invalid; ALWAYS retry API errors.
    # (older files stored API errors as status "invalid" with reason "api error...", so check the reason too)
    keep = {i: r for i, r in previous.items()
            if r["status"] == "ok" or (r["status"] == "invalid" and not a.retry_invalid
                                       and not r.get("reason", "").startswith("api error"))}
    todo = [p for p in passages if p["id"] not in keep]
    if a.limit:
        todo = todo[: a.limit]
    print(f"{len(passages)} passages, {len(keep)} already labeled, labeling {len(todo)} with "
          f"{'DRY-RUN' if a.dry_run else a.model}")

    # Rewrite the file without stale invalid rows we are about to retry, then append.
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with a.output.open("w", encoding="utf-8") as f:
        for r in keep.values():
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    lock, stats, t0 = threading.Lock(), Counter(), time.time()
    client = None if a.dry_run else make_client(a.base_url)
    with a.output.open("a", encoding="utf-8") as out, ThreadPoolExecutor(max_workers=a.workers) as pool:
        futs = [pool.submit(dry_label if a.dry_run else label_one, *(() if a.dry_run else (client, a)), p)
                for p in todo]
        for n, fut in enumerate(as_completed(futs), 1):
            r = fut.result()
            with lock:
                out.write(json.dumps(r, ensure_ascii=False) + "\n")
                out.flush()
                stats[r["status"]] += 1
                if r["status"] == "ok":
                    stats["class:" + r["label"]["risk_class"]] += 1
            if n >= 5 and stats["api_error"] == n and r.get("code") in FATAL_CODES:
                for f in futs:
                    f.cancel()
                sys.exit(f"\nSTOPPED: first {n} requests all failed with API error {r.get('code')} "
                         f"for model {a.model}.\n  {r['reason'][:200]}\n"
                         "Run: python scripts/03_label_risk.py --probe   (shows which models your key can use)")
            if n % 25 == 0 or n == len(todo):
                rate = n / max(time.time() - t0, 1e-6)
                print(f"  {n}/{len(todo)} ok={stats['ok']} invalid={stats['invalid']} api_error={stats['api_error']} "
                      f"({rate:.2f}/s, ~{(len(todo) - n) / max(rate, 1e-6) / 60:.1f} min left)", flush=True)

    all_rows = read_jsonl(a.output)
    ok = [r for r in all_rows if r["status"] == "ok"]
    print(f"\nTotal valid labels: {len(ok)} / {len(passages)} passages")
    print("class distribution:", dict(Counter(r["label"]["risk_class"] for r in ok).most_common()))
    print("severity distribution:", dict(Counter(r["label"]["severity"] for r in ok).most_common()))
    bad = Counter(r.get("reason", "")[:40] for r in all_rows if r["status"] != "ok")
    if bad:
        print("invalid reasons:", dict(bad.most_common(5)))


if __name__ == "__main__":
    main()
