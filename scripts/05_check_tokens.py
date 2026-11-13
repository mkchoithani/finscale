#!/usr/bin/env python3
"""Gate before training: prove every JSON answer survives tokenization at --seq-length.

Two checks:
  1. Tokenizer only (runs anywhere): full conversation length vs seq_length, answer token count.
  2. NeMo AutoModel ChatDataset (on the GPU node): the exact loader used in training must keep
     ALL answer tokens as labels (labels != -100) for EVERY row.

    python scripts/05_check_tokens.py --model $MODEL_DIR --data data/sft/train.jsonl --seq-length 3072
Exit code 0 = PASS. Anything else: do not train.
"""
import argparse
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from finscale.common import read_jsonl  # noqa: E402


def render_len(tok, msgs, tkw):
    text = tok.apply_chat_template(msgs, tokenize=False, **tkw)
    return len(tok(text, add_special_tokens=False)["input_ids"])


def automodel_check(data, tok, seq_length, tkw, limit):
    import torch
    from nemo_automodel.components.datasets.llm.chat_dataset import ChatDatasetConfig

    def build(sl, trunc):
        kw = dict(path_or_dataset_id=str(Path(data).resolve()), seq_length=sl, truncation=trunc,
                  chat_template_kwargs=tkw)
        try:
            return ChatDatasetConfig(**kw).build(tokenizer=tok)
        except TypeError:
            kw.pop("chat_template_kwargs")
            print("  note: this AutoModel version has no chat_template_kwargs; using template defaults")
            return ChatDatasetConfig(**kw).build(tokenizer=tok)

    full, cut = build(None, "do_not_truncate"), build(seq_length, "longest_first")
    n = len(full) if not limit else min(limit, len(full))
    sup = lambda row: int((torch.as_tensor(row["labels"]) != -100).sum())
    complete = partial = empty = 0
    for i in range(n):
        ref, got = sup(full[i]), sup(cut[i])
        if ref > 0 and got == ref:
            complete += 1
        elif got > 0:
            partial += 1
        else:
            empty += 1
    print(f"  AutoModel ChatDataset @ {seq_length}: complete={complete} partial={partial} empty={empty} of {n}")
    return complete == n


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="local model dir or HF id (only the tokenizer is loaded)")
    ap.add_argument("--data", nargs="+", default=["data/sft/train.jsonl", "data/sft/val.jsonl"])
    ap.add_argument("--seq-length", type=int, default=3072)
    ap.add_argument("--limit", type=int, default=0, help="AutoModel check: rows to check (0 = all)")
    ap.add_argument("--enable-thinking", action="store_true")
    ap.add_argument("--skip-automodel", action="store_true")
    a = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True)
    tkw = {"enable_thinking": a.enable_thinking}
    ok = True
    for path in a.data:
        rows = read_jsonl(path)
        if not rows:
            print(f"FAIL {path}: empty or missing")
            ok = False
            continue
        lens = [render_len(tok, r["messages"], tkw) for r in rows]
        ans = [len(tok(r["messages"][-1]["content"], add_special_tokens=False)["input_ids"]) for r in rows]
        over = sum(l > a.seq_length for l in lens)
        q = sorted(lens)
        print(f"{path}: {len(rows)} rows | tokens median={statistics.median(q):.0f} p95={q[int(.95 * (len(q) - 1))]} "
              f"max={q[-1]} | answer tokens median={statistics.median(ans):.0f} max={max(ans)} | "
              f"over {a.seq_length}: {over}")
        if over:
            ok = False
            print(f"  FAIL: {over} rows longer than {a.seq_length}. Raise --seq-length or lower "
                  "02_make_passages.py --max-chars.")
        if not a.skip_automodel:
            try:
                ok &= automodel_check(path, tok, a.seq_length, tkw, a.limit)
            except ImportError:
                print("  (nemo_automodel not installed here: AutoModel check skipped; run it on the GPU node)")
    print("PASS" if ok else "FAIL: do not train")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
