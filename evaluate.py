#!/usr/bin/env python3
"""Evaluate base and fine-tuned FinScale classifiers on data/mlx/test.jsonl."""
from __future__ import annotations
import argparse, json, re, statistics, sys
from pathlib import Path


def parse_json_output(text):
    text=(text or "").strip()
    try:
        obj=json.loads(text)
        return obj if isinstance(obj,dict) else None
    except Exception:
        # Accept a single JSON object surrounded by incidental prose/fences for field metrics,
        # but count it as invalid JSON because the model did not return a clean object.
        cleaned=re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.I)
        a,b=cleaned.find("{"),cleaned.rfind("}")
        if a>=0 and b>a:
            try:
                obj=json.loads(cleaned[a:b+1]); return obj if isinstance(obj,dict) else None
            except Exception: pass
    return None


def read_test(path):
    rows=[]
    for n,line in enumerate(path.read_text(encoding="utf-8").splitlines(),1):
        if not line.strip(): continue
        try: row=json.loads(line)
        except Exception as e: raise ValueError(f"Invalid test JSONL line {n}: {e}")
        msgs=row.get("messages")
        if not isinstance(msgs,list) or len(msgs)<3: raise ValueError(f"Line {n}: expected chat messages list")
        try: gold=json.loads(msgs[-1]["content"])
        except Exception as e: raise ValueError(f"Line {n}: final assistant message must contain JSON label: {e}")
        rows.append((msgs, gold))
    return rows


def macro_f1(truth, pred, labels):
    vals=[]
    for label in labels:
        tp=sum(t==label and p==label for t,p in zip(truth,pred))
        fp=sum(t!=label and p==label for t,p in zip(truth,pred))
        fn=sum(t==label and p!=label for t,p in zip(truth,pred))
        denom=2*tp+fp+fn
        vals.append((2*tp/denom) if denom else 0.0)
    return sum(vals)/len(vals) if vals else 0.0


def load_and_generate(backend, model_id, adapter, messages, max_tokens):
    if backend=="mlx":
        from mlx_lm import load, generate
        model, tokenizer=load(model_id, adapter_path=str(adapter) if adapter else None)
        prompt=tokenizer.apply_chat_template(messages[:-1], tokenize=False, add_generation_prompt=True)
        return generate(model,tokenizer,prompt=prompt,max_tokens=max_tokens)
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel
    tokenizer=AutoTokenizer.from_pretrained(model_id)
    model=AutoModelForCausalLM.from_pretrained(model_id,torch_dtype="auto",device_map="auto")
    if adapter: model=PeftModel.from_pretrained(model,str(adapter))
    prompt=tokenizer.apply_chat_template(messages[:-1],tokenize=False,add_generation_prompt=True)
    inputs=tokenizer(prompt,return_tensors="pt").to(model.device)
    with torch.inference_mode(): out=model.generate(**inputs,max_new_tokens=max_tokens,do_sample=False,pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(out[0][inputs["input_ids"].shape[1]:],skip_special_tokens=True)


def score(rows, backend, model_id, adapter, max_tokens):
    cat_true=[]; cat_pred=[]; sev_true=[]; sev_pred=[]; valid=0; detail=[]
    model=None; tokenizer=None
    # Keep one loaded model for the full test run.
    if backend=="mlx":
        from mlx_lm import load, generate
        model,tokenizer=load(model_id,adapter_path=str(adapter) if adapter else None)
    else:
        import torch
        from transformers import AutoTokenizer,AutoModelForCausalLM
        from peft import PeftModel
        tokenizer=AutoTokenizer.from_pretrained(model_id)
        model=AutoModelForCausalLM.from_pretrained(model_id,torch_dtype="auto",device_map="auto")
        if adapter: model=PeftModel.from_pretrained(model,str(adapter))
    for msgs,gold in rows:
        prompt=tokenizer.apply_chat_template(msgs[:-1],tokenize=False,add_generation_prompt=True)
        if backend=="mlx": raw=generate(model,tokenizer,prompt=prompt,max_tokens=max_tokens)
        else:
            import torch
            inputs=tokenizer(prompt,return_tensors="pt").to(model.device)
            with torch.inference_mode(): out=model.generate(**inputs,max_new_tokens=max_tokens,do_sample=False,pad_token_id=tokenizer.eos_token_id)
            raw=tokenizer.decode(out[0][inputs["input_ids"].shape[1]:],skip_special_tokens=True)
        pred=parse_json_output(raw)
        clean_valid=False
        try:
            candidate=json.loads((raw or "").strip())
            clean_valid=isinstance(candidate,dict)
        except Exception: pass
        valid+=int(clean_valid)
        cat_true.append(gold.get("category")); sev_true.append(gold.get("severity"))
        cat_pred.append(pred.get("category") if pred else None); sev_pred.append(pred.get("severity") if pred else None)
        detail.append({"gold_category":gold.get("category"),"pred_category":cat_pred[-1],"gold_severity":gold.get("severity"),"pred_severity":sev_pred[-1],"valid_json":clean_valid,"raw_output":raw})
    cats=sorted(set(cat_true)); sevs=sorted(set(sev_true))
    return {"examples":len(rows),"valid_json_count":valid,"valid_json_rate":valid/len(rows) if rows else 0,
       "category_accuracy":sum(a==b for a,b in zip(cat_true,cat_pred))/len(rows) if rows else 0,
       "category_macro_f1":macro_f1(cat_true,cat_pred,cats),
       "severity_accuracy":sum(a==b for a,b in zip(sev_true,sev_pred))/len(rows) if rows else 0,
       "category_labels_scored":cats,"per_example":detail}


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--test",type=Path,default=Path("data/mlx/test.jsonl")); ap.add_argument("--backend",choices=("mlx","peft"),default="mlx")
    ap.add_argument("--model",default="Qwen/Qwen2.5-3B-Instruct"); ap.add_argument("--adapter",type=Path,default=Path("out/adapters"))
    ap.add_argument("--out-dir",type=Path,default=Path("out")); ap.add_argument("--max-tokens",type=int,default=384)
    args=ap.parse_args()
    if not args.test.exists(): ap.error(f"test split not found: {args.test}; run train.py first")
    rows=read_test(args.test)
    if not rows: ap.error("test split has no examples")
    base=score(rows,args.backend,args.model,None,args.max_tokens)
    tuned=score(rows,args.backend,args.model,args.adapter,args.max_tokens)
    results={"backend":args.backend,"model":args.model,"adapter":str(args.adapter),"test_file":str(args.test),"base":base,"fine_tuned":tuned}
    args.out_dir.mkdir(parents=True,exist_ok=True)
    (args.out_dir/"eval_results.json").write_text(json.dumps(results,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    cols=["Model","Valid JSON","Category accuracy","Category macro-F1","Severity accuracy"]
    lines=["| " + " | ".join(cols) + " |","|"+"|".join(["---"]*len(cols))+"|"]
    for name,m in (("Base",base),("Fine-tuned",tuned)):
        lines.append(f"| {name} | {m['valid_json_rate']:.1%} | {m['category_accuracy']:.1%} | {m['category_macro_f1']:.3f} | {m['severity_accuracy']:.1%} |")
    md="\n".join(lines)+"\n"
    (args.out_dir/"eval_results.md").write_text(md,encoding="utf-8")
    print(md,end="")

if __name__=="__main__": main()
