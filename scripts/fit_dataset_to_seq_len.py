#!/usr/bin/env python3
"""Fix the zero-supervised-token blocker BEFORE training.

Root cause: build_sft_dataset.py puts up to 12,000 characters of 10-K text
(~3,000 tokens) in the *user* turn, and the assistant answer comes last.
ChatDataset truncation keeps the prefix, so at seq_length=512 (or even 2048)
the assistant answer is cut away entirely and every label is -100.

This script renders each row with the real tokenizer chat template and:
  * reports the token-length distribution of the FULL conversation,
  * mode=drop (default): keeps only rows whose full render fits seq_length,
  * mode=trim: shortens only the "Source section:" text inside the user turn
    until the full render fits; quote rows are re-verified against the trimmed
    source and dropped if the quoted sentence is no longer present.

The assistant answer is never truncated. Output rows are training-ready.

Example:
  python scripts/fit_dataset_to_seq_len.py --model /workspace/models/nemotron-30b \
      --input data/processed/train.jsonl --output data/processed/train_fit.jsonl \
      --seq-length 4096
"""
import argparse
import json
import re
import statistics
import sys
from pathlib import Path

from transformers import AutoTokenizer

SOURCE_MARKER = "Source section:\n"


def render_len(tok, messages, template_kwargs):
    # Render to text, then tokenize: apply_chat_template(tokenize=True) returns a
    # BatchEncoding in transformers>=5 and a list in 4.x, so avoid relying on it.
    text = tok.apply_chat_template(messages, tokenize=False, **template_kwargs)
    return len(tok(text, add_special_tokens=False)["input_ids"])


def quotes_present(answer, source):
    norm = lambda s: re.sub(r"\s+", " ", s).lower()
    quoted = re.findall(r'["“]([^"”]{25,})["”]', answer)
    return all(norm(q) in norm(source) for q in quoted)


def trim_to_fit(tok, messages, seq_length, template_kwargs, min_source_chars):
    """Binary-search the longest source prefix that fits. Returns (messages, n_tokens) or None."""
    user_idx = next((i for i, m in enumerate(messages) if m["role"] == "user"), None)
    if user_idx is None or SOURCE_MARKER not in messages[user_idx]["content"]:
        return None
    head, source = messages[user_idx]["content"].split(SOURCE_MARKER, 1)

    def with_source(n):
        msgs = [dict(m) for m in messages]
        msgs[user_idx]["content"] = head + SOURCE_MARKER + source[:n]
        return msgs

    lo, hi, best = min_source_chars, len(source), None
    while lo <= hi:
        mid = (lo + hi) // 2
        n_tok = render_len(tok, with_source(mid), template_kwargs)
        if n_tok <= seq_length:
            best, lo = (mid, n_tok), mid + 1
        else:
            hi = mid - 1
    if best is None:
        return None
    # Cut on a sentence boundary so the context does not end mid-word.
    n_chars = best[0]
    cut = source[:n_chars].rfind(". ")
    if cut > min_source_chars:
        n_chars = cut + 1
    msgs = with_source(n_chars)
    return msgs, render_len(tok, msgs, template_kwargs)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="Local model dir or HF id (tokenizer only is loaded)")
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--seq-length", type=int, default=4096)
    ap.add_argument("--mode", choices=["drop", "trim"], default="drop")
    ap.add_argument("--min-source-chars", type=int, default=1500,
                    help="trim mode: never keep less source text than this")
    ap.add_argument("--enable-thinking", action="store_true",
                    help="Render with reasoning enabled (teacher labels were generated with it OFF)")
    a = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True)
    template_kwargs = {"enable_thinking": a.enable_thinking}

    rows = [json.loads(l) for l in a.input.read_text(encoding="utf-8").splitlines() if l.strip()]
    lengths, kept, stats = [], [], {"fit": 0, "trimmed": 0, "dropped_too_long": 0,
                                    "dropped_quote_lost": 0, "dropped_no_assistant": 0}
    for row in rows:
        msgs = row["messages"]
        if not msgs or msgs[-1]["role"] != "assistant" or not msgs[-1]["content"].strip():
            stats["dropped_no_assistant"] += 1
            continue
        n = render_len(tok, msgs, template_kwargs)
        lengths.append(n)
        if n <= a.seq_length:
            kept.append(row)
            stats["fit"] += 1
            continue
        if a.mode == "drop":
            stats["dropped_too_long"] += 1
            continue
        res = trim_to_fit(tok, msgs, a.seq_length, template_kwargs, a.min_source_chars)
        if res is None:
            stats["dropped_too_long"] += 1
            continue
        new_msgs, _ = res
        user = next(m for m in new_msgs if m["role"] == "user")["content"]
        label_type = row.get("metadata", {}).get("label_type", "")
        if "quote" in label_type and not quotes_present(new_msgs[-1]["content"], user):
            stats["dropped_quote_lost"] += 1
            continue
        if "summary" in label_type:
            # Summaries were written from the full source; trimming may remove supporting text.
            row.setdefault("metadata", {})["context_trimmed"] = True
        kept.append({**row, "messages": new_msgs})
        stats["trimmed"] += 1

    if lengths:
        q = sorted(lengths)
        pct = lambda p: q[min(len(q) - 1, int(p * len(q)))]
        print(f"Full-render tokens over {len(q)} rows: min={q[0]} median={statistics.median(q):.0f} "
              f"p90={pct(.9)} p99={pct(.99)} max={q[-1]}")
        for L in (512, 1024, 2048, 3072, 4096, 6144, 8192):
            print(f"  fits in {L:>5}: {sum(x <= L for x in q):>4}/{len(q)}")
    print(json.dumps(stats, indent=2))
    if not kept:
        sys.exit(f"FAIL: no rows fit seq_length={a.seq_length}. Increase it or use --mode trim.")
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with a.output.open("w", encoding="utf-8") as f:
        for row in kept:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Wrote {len(kept)}/{len(rows)} rows -> {a.output} (seq_length={a.seq_length})")


if __name__ == "__main__":
    main()
