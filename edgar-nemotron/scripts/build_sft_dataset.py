#!/usr/bin/env python3
"""Generate filing-grounded message JSONL. Summary responses are extractive pseudo-labels."""
import argparse,json,random,re
from pathlib import Path
def sections(text):
    matches=list(re.finditer(r"(?im)^\s*(?:item\s+\d{1,2}[a-z]?\.?\s+[^\n]{3,160})\s*$",text)); result=[]
    for n,m in enumerate(matches):
        body=text[m.end():matches[n+1].start() if n+1<len(matches) else len(text)].strip()
        if len(body)>=400: result.append((" ".join(m.group(0).split()),body))
    return result
def build(filing,max_sections):
    out=[]; ticker=filing.get("ticker","company"); date=filing.get("filing_date","unknown")
    for heading,body in sections(filing.get("text",""))[:max_sections]:
        context=body[:12000]; lead=re.split(r"(?<=[.!?])\s+",context)[0][:1200].strip()
        if not lead: continue
        prompts=[(f"Summarize {ticker}'s filing section {heading} (filed {date}) using only the source. State uncertainty.\n\nSource section:\n{context}","You are a careful financial filing assistant. Ground every statement in the supplied source; do not invent facts.","extractive-lead-pseudo-summary"),
                 (f"Identify the opening disclosed detail in {ticker}'s section {heading}. Quote the evidence.\n\nSource section:\n{context}","Answer from the SEC filing text only.","extractive")]
        for prompt,system,kind in prompts:
            out.append({"messages":[{"role":"system","content":system},{"role":"user","content":prompt},{"role":"assistant","content":lead}],"metadata":{"ticker":ticker,"filing_date":date,"accession":filing.get("accession"),"section":heading,"label_type":kind}})
    return out
def main():
    ap=argparse.ArgumentParser(description=__doc__); ap.add_argument("--input",type=Path,default=Path("data/raw/edgar_10k.jsonl")); ap.add_argument("--output-dir",type=Path,default=Path("data/processed")); ap.add_argument("--val-fraction",type=float,default=.1); ap.add_argument("--seed",type=int,default=17); ap.add_argument("--max-sections",type=int,default=8); a=ap.parse_args()
    if not 0<=a.val_fraction<1: ap.error("--val-fraction must be in [0, 1)")
    rows=[]
    with a.input.open(encoding="utf-8") as src:
        for n,line in enumerate(src,1):
            if line.strip():
                try: rows.extend(build(json.loads(line),a.max_sections))
                except (json.JSONDecodeError,TypeError) as e: raise SystemExit(f"Bad JSONL line {n}: {e}")
    if not rows: raise SystemExit("No examples produced. Check input and section headings.")
    random.Random(a.seed).shuffle(rows); nv=int(len(rows)*a.val_fraction); a.output_dir.mkdir(parents=True,exist_ok=True)
    for name,data in (("train",rows[nv:]),("val",rows[:nv])):
        if name=="train" and not data: raise SystemExit("Training split is empty")
        path=a.output_dir/f"{name}.jsonl"
        with path.open("w",encoding="utf-8") as f:
            for row in data: f.write(json.dumps(row,ensure_ascii=False)+"\n")
        print(f"{path}: {len(data)} examples")
    print("WARNING: summary labels are extractive opening sentences, not human-validated summaries; inspect before training.")
if __name__=="__main__": main()
