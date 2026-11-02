#!/usr/bin/env python3
"""Fallback token memmap writer; custom flat .bin + JSON .idx, NOT Megatron indexed format."""
import argparse,json
from pathlib import Path
import numpy as np
def main():
    ap=argparse.ArgumentParser(description=__doc__); ap.add_argument("input",type=Path); ap.add_argument("--tokenizer",required=True); ap.add_argument("--output-prefix",type=Path,required=True); ap.add_argument("--dtype",choices=("uint32","int32","int64"),default="uint32"); a=ap.parse_args()
    try: from transformers import AutoTokenizer
    except ImportError as e: raise SystemExit("Install transformers") from e
    tok=AutoTokenizer.from_pretrained(a.tokenizer,token=True); dtype=np.dtype(a.dtype); a.output_prefix.parent.mkdir(parents=True,exist_ok=True)
    bp=a.output_prefix.with_suffix(".bin"); ip=a.output_prefix.with_suffix(".idx"); encoded=[]; offsets=[]; total=0
    with a.input.open(encoding="utf-8") as src:
        for line in src:
            if not line.strip(): continue
            messages=json.loads(line)["messages"]
            if tok.chat_template: ids=tok.apply_chat_template(messages,tokenize=True,add_generation_prompt=False)
            else: ids=tok("\n".join(f"{m['role']}: {m['content']}" for m in messages),add_special_tokens=True)["input_ids"]
            arr=np.asarray(ids,dtype=dtype); encoded.append(arr); offsets.append([total,len(arr)]); total+=len(arr)
    if not total: raise SystemExit("No tokens produced")
    mm=np.memmap(bp,dtype=dtype,mode="w+",shape=(total,)); cursor=0
    for arr in encoded: mm[cursor:cursor+len(arr)]=arr; cursor+=len(arr)
    mm.flush(); del mm
    ip.write_text(json.dumps({"format":"edgar-nemotron-flat-token-v1","dtype":dtype.name,"token_count":total,"documents":offsets},indent=2)+"\n",encoding="utf-8")
    print(f"Wrote {bp} and {ip}; custom token stream, not compatible with Megatron preprocess_data indexed datasets.")
if __name__=="__main__": main()
