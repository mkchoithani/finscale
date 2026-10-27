import json, random, re
from pathlib import Path
from tqdm import tqdm
from mlx_lm import load, generate

LABEL_MODEL = "mlx-community/Qwen2.5-7B-Instruct-4bit"
CATEGORIES = ["market", "credit", "liquidity", "operational",
              "regulatory_legal", "cybersecurity", "supply_chain", "other"]
SEVERITIES = ["low", "medium", "high"]

SYSTEM = (
    "You are a financial risk analyst. Classify the risk disclosure excerpt. "
    "Respond with ONLY a JSON object with keys: "
    '"category" (one of ' + ", ".join(CATEGORIES) + '), '
    '"severity" (one of low, medium, high), '
    '"evidence" (a short verbatim quote from the excerpt, max 25 words), '
    '"explanation" (1-2 sentences justifying the label, grounded in the evidence).'
)

CHUNKS_PER_FILE = 25
MIN_WORDS, MAX_WORDS = 60, 220

def chunk_text(text):
    paras = [p.strip() for p in re.split(r"\n+", text) if p.strip()]
    chunks, cur = [], []
    for p in paras:
        cur.append(p)
        words = sum(len(x.split()) for x in cur)
        if words >= MIN_WORDS:
            if words <= MAX_WORDS:
                chunks.append(" ".join(cur))
            cur = []
    return chunks

def parse(raw, excerpt):
    m = re.search(r"\{.*\}", raw, re.S)
    if not m: return None
    try:
        d = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    if d.get("category") not in CATEGORIES: return None
    if d.get("severity") not in SEVERITIES: return None
    ev, ex = d.get("evidence", ""), d.get("explanation", "")
    if not ev or not ex: return None
    # evidence must actually appear in the excerpt (anti-hallucination filter)
    norm = lambda s: re.sub(r"\s+", " ", s.lower()).strip(" .\"'")
    if norm(ev) not in norm(excerpt): return None
    return {"category": d["category"], "severity": d["severity"],
            "evidence": ev.strip(), "explanation": ex.strip()}

def main():
    random.seed(7)
    files = sorted(Path("data/raw").glob("*.txt"))
    assert files, "No files in data/raw. Run download_edgar.py first."
    model, tok = load(LABEL_MODEL)

    by_company = {}
    for f in files:
        by_company.setdefault(f.stem.split("_")[0], []).append(f)
    tickers = sorted(by_company)
    random.shuffle(tickers)
    n_eval = max(2, len(tickers) // 5)
    eval_t = set(tickers[:n_eval])
    print("Held-out eval companies:", sorted(eval_t))

    rows = {"train": [], "eval": []}
    for ticker, flist in by_company.items():
        split = "eval" if ticker in eval_t else "train"
        for f in flist:
            chunks = chunk_text(f.read_text())
            random.shuffle(chunks)
            for ch in tqdm(chunks[:CHUNKS_PER_FILE], desc=f.name):
                msgs = [{"role": "system", "content": SYSTEM},
                        {"role": "user", "content": f"Excerpt:\n{ch}"}]
                prompt = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
                raw = generate(model, tok, prompt=prompt, max_tokens=250, verbose=False)
                label = parse(raw, ch)
                if label is None:
                    continue
                rows[split].append({"ticker": ticker, "excerpt": ch, "label": label})

    Path("data").mkdir(exist_ok=True)
    for split, items in rows.items():
        with open(f"data/{split}_raw.jsonl", "w") as fh:
            for it in items:
                fh.write(json.dumps(it) + "\n")
        print(split, len(items))

if __name__ == "__main__":
    main()
