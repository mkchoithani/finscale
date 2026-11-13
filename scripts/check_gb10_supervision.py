#!/usr/bin/env python3
"""Check that FinScale ChatDataset produces COMPLETE supervised answers.

The old check passed if any single row had >0 supervised tokens, which still
allows training on answers that are truncated mid-sentence (no EOS). This
version compares each row's supervised-token count at the target seq_length
with the count for the untruncated render and requires the full answer.
Works on any CUDA host (GB10, RunPod H100/H200, etc.); only the tokenizer is loaded.
"""
import argparse
import sys

import torch
from transformers import AutoTokenizer
from nemo_automodel.components.datasets.llm.chat_dataset import ChatDatasetConfig


def build(data, tok, seq_length, truncation, template_kwargs):
    kw = dict(path_or_dataset_id=data, seq_length=seq_length, truncation=truncation)
    if template_kwargs is not None:
        kw["chat_template_kwargs"] = template_kwargs
    try:
        return ChatDatasetConfig(**kw).build(tokenizer=tok)
    except TypeError:
        # Older AutoModel (e.g. 0.8.0) has no chat_template_kwargs field.
        kw.pop("chat_template_kwargs", None)
        print("WARNING: installed AutoModel ignores chat_template_kwargs", file=sys.stderr)
        return ChatDatasetConfig(**kw).build(tokenizer=tok)


def supervised(row):
    return int((torch.as_tensor(row["labels"]) != -100).sum().item())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--lengths", type=int, nargs="+", default=[2048, 3072, 4096, 6144])
    p.add_argument("--limit", type=int, default=0, help="0 = check every row")
    p.add_argument("--min-complete-frac", type=float, default=1.0,
                   help="Fraction of checked rows whose answer must be fully retained")
    p.add_argument("--enable-thinking", action="store_true")
    args = p.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    tkw = {"enable_thinking": args.enable_thinking}
    full = build(args.data, tok, None, "do_not_truncate", tkw)
    n = len(full) if args.limit <= 0 else min(args.limit, len(full))
    ref = [supervised(full[i]) for i in range(n)]
    ref_len = [len(full[i]["input_ids"]) for i in range(n)]
    print(f"rows checked={n}; untruncated tokens max={max(ref_len)}; rows with 0 answer tokens even "
          f"untruncated={sum(r == 0 for r in ref)}", flush=True)
    if any(r == 0 for r in ref):
        print("  -> those rows have a role/template problem, not a length problem", flush=True)

    chosen = None
    for length in sorted(args.lengths):
        ds = build(args.data, tok, length, "longest_first", tkw)
        complete = partial = empty = 0
        for i in range(n):
            s = supervised(ds[i])
            if ref[i] > 0 and s == ref[i]:
                complete += 1
            elif s > 0:
                partial += 1
            else:
                empty += 1
        frac = complete / n
        print(f"seq_length={length}: complete={complete} partial={partial} empty={empty} "
              f"({frac:.1%} complete)", flush=True)
        if chosen is None and frac >= args.min_complete_frac:
            chosen = length
    if chosen is None:
        sys.exit("FAIL: no checked seq_length keeps every answer intact. Run "
                 "scripts/fit_dataset_to_seq_len.py first, or raise --lengths. Do not train.")
    print(f"PASS: use seq_length >= {chosen}", flush=True)


if __name__ == "__main__":
    main()
