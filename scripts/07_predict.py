#!/usr/bin/env python3
"""Run a model on the held-out test set with the SAME prompt used in training.

Backends:
  api  Any OpenAI-compatible endpoint. Default = NVIDIA API serving the untuned base model:
         python scripts/07_predict.py --name base --backend api --model nvidia/nemotron-nano-3-30b-a3b
       A local vLLM server with the LoRA adapter works the same way:
         vllm serve $MODEL_DIR --enable-lora --lora-modules finscale=<adapter dir> --trust-remote-code
         python scripts/07_predict.py --name finetuned --backend api --base-url http://localhost:8000/v1 \
             --model finscale --api-key-env NONE
  hf   Transformers + PEFT on a GPU node (slower, no server needed):
         python scripts/07_predict.py --name finetuned --backend hf --model-path $MODEL_DIR \
             --adapter results/runs/<run>/ckpt/LATEST/model
"""
import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from finscale.common import parse_json_answer, read_jsonl, student_messages, write_jsonl  # noqa: E402


def api_predictor(a):
    from openai import OpenAI
    key = "EMPTY" if a.api_key_env == "NONE" else os.environ.get(a.api_key_env, "")
    if not key:
        sys.exit(f"Set {a.api_key_env}")
    client = OpenAI(base_url=a.base_url, api_key=key, timeout=180, max_retries=4)

    def predict(row):
        kw = dict(model=a.model, messages=student_messages(row), temperature=0.0, max_tokens=a.max_new_tokens)
        if not a.thinking:
            kw["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}
        t = time.perf_counter()
        try:
            out = client.chat.completions.create(**kw).choices[0].message.content or ""
        except Exception as e:
            out = f"__ERROR__ {e}"
        return out, time.perf_counter() - t
    return predict


def hf_predictor(a):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.adapter or a.model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(a.model_path, trust_remote_code=True, torch_dtype=torch.bfloat16,
                                                 device_map="cuda")
    if a.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, a.adapter)
    model.eval()

    def predict(row):
        ids = tok.apply_chat_template(student_messages(row), add_generation_prompt=True, return_tensors="pt",
                                      enable_thinking=a.thinking)
        ids = (ids["input_ids"] if hasattr(ids, "keys") else ids).to("cuda")
        t = time.perf_counter()
        with torch.no_grad():
            out = model.generate(input_ids=ids, max_new_tokens=a.max_new_tokens, do_sample=False,
                                 pad_token_id=tok.pad_token_id or tok.eos_token_id)
        text = tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)
        return text, time.perf_counter() - t
    return predict


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True, help="label for this system, e.g. base / finetuned")
    ap.add_argument("--backend", choices=["api", "hf"], required=True)
    ap.add_argument("--test", default="data/sft/test.jsonl")
    ap.add_argument("--out-dir", default="results/eval")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-new-tokens", type=int, default=400)
    ap.add_argument("--thinking", action="store_true")
    ap.add_argument("--base-url", default="https://integrate.api.nvidia.com/v1")
    ap.add_argument("--model", default="nvidia/nemotron-nano-3-30b-a3b")
    ap.add_argument("--api-key-env", default="NVIDIA_API_KEY", help="NONE for a local server without auth")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--model-path", help="hf backend: local base model dir")
    ap.add_argument("--adapter", help="hf backend: LoRA adapter dir (…/ckpt/LATEST/model)")
    a = ap.parse_args()

    rows = read_jsonl(a.test)
    if not rows:
        sys.exit(f"No rows in {a.test}")
    rows = rows[: a.limit] if a.limit else rows
    if a.backend == "hf" and not a.model_path:
        sys.exit("--model-path is required for --backend hf")
    predict = api_predictor(a) if a.backend == "api" else hf_predictor(a)
    workers = a.workers if a.backend == "api" else 1
    print(f"Predicting {len(rows)} rows as '{a.name}' ({a.backend}, {a.model if a.backend == 'api' else a.model_path})")
    with ThreadPoolExecutor(max_workers=workers) as pool:
        outs = list(pool.map(predict, rows))
    preds = [{"id": r["id"], "system": a.name, "raw": raw, "parsed": parse_json_answer(raw),
              "latency_s": round(lat, 3)} for r, (raw, lat) in zip(rows, outs)]
    path = Path(a.out_dir) / f"pred-{a.name}.jsonl"
    write_jsonl(path, preds)
    errors = sum(p["raw"].startswith("__ERROR__") for p in preds)
    print(f"Wrote {path} | parsed JSON: {sum(p['parsed'] is not None for p in preds)}/{len(preds)} | errors: {errors}")


if __name__ == "__main__":
    main()
