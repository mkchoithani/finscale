#!/usr/bin/env python3
"""Evaluate base vs LoRA adapter on held-out messages JSONL."""
import argparse, json, re, sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def load(model_name, adapter=None):
    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    m = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="cuda")
    if adapter:
        from peft import PeftModel
        m = PeftModel.from_pretrained(m, adapter)
    m.eval()
    return tok, m

def lcs_len(a, b):
    prev = [0]*(len(b)+1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b):
            cur.append(prev[j]+1 if x == y else max(prev[j+1], cur[j]))
        prev = cur
    return prev[-1]

def rouge_l(pred, ref):
    p, r = pred.lower().split(), ref.lower().split()
    if not p or not r: return 0.0
    l = lcs_len(p, r)
    if l == 0: return 0.0
    pr, rc = l/len(p), l/len(r)
    return 2*pr*rc/(pr+rc)

def grounding(pred, source, n=6):
    w = re.findall(r"\w+", pred.lower())
    if len(w) < n: return 1.0
    src = " ".join(re.findall(r"\w+", source.lower()))
    grams = [" ".join(w[i:i+n]) for i in range(len(w)-n+1)]
    return sum(g in src for g in grams)/len(grams)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="nvidia/Llama-3.1-Nemotron-Nano-8B-v1")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-new", type=int, default=256)
    ap.add_argument("--max-input", type=int, default=6000)
    a = ap.parse_args()

    tok, model = load(a.model, a.adapter)
    rows = [json.loads(l) for l in open(a.data)]
    results = []
    for i, r in enumerate(rows):
        msgs = r["messages"]
        prompt_msgs, ref = msgs[:-1], msgs[-1]["content"]
        source = prompt_msgs[-1]["content"]
        ids = tok.apply_chat_template(prompt_msgs, add_generation_prompt=True, return_tensors="pt")
        if not torch.is_tensor(ids):
            ids = ids["input_ids"]
        if ids.shape[1] > a.max_input:
            print(f"[{i}] skipped: prompt {ids.shape[1]} tokens", file=sys.stderr); continue
        with torch.no_grad():
            out = model.generate(ids.to("cuda"), max_new_tokens=a.max_new, do_sample=False,
                                 pad_token_id=tok.pad_token_id)
        pred = tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True).strip()
        results.append({"i": i, "pred": pred, "ref": ref,
                        "rougeL": rouge_l(pred, ref), "grounding": grounding(pred, source),
                        "len_words": len(pred.split())})
        print(f"[{i}] rougeL={results[-1]['rougeL']:.3f} grounding={results[-1]['grounding']:.3f}", flush=True)

    n = max(len(results), 1)
    summary = {"adapter": a.adapter, "n": len(results),
               "rougeL": sum(x["rougeL"] for x in results)/n,
               "grounding": sum(x["grounding"] for x in results)/n,
               "avg_words": sum(x["len_words"] for x in results)/n}
    json.dump({"summary": summary, "results": results}, open(a.out, "w"), indent=2)
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
