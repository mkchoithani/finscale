import json, random, os
random.seed(7)
CATS = {
 "liquidity_risk": ["cash reserves declined {p}% and short-term debt of ${n}M matures within 12 months","the revolving credit facility of ${n}M may be unavailable if covenants are breached"],
 "credit_risk": ["{p}% of receivables are more than 90 days past due","loan loss provisions rose {p}% amid borrower defaults totaling ${n}M"],
 "market_risk": ["a {p}% move in interest rates would change net income by ${n}M","foreign exchange exposure of ${n}M is unhedged"],
 "operational_risk": ["a cyber incident disrupted systems for {p} days at a cost of ${n}M","supply chain failures delayed {p}% of shipments"],
 "regulatory_risk": ["an ongoing investigation may result in fines up to ${n}M","new capital rules could require raising ${n}M of additional capital"],
 "low_risk": ["the company reported stable earnings and maintained all covenants","no material changes to risk exposure occurred during the period"],
}
def make(n):
    rows = []
    for _ in range(n):
        cat = random.choice(list(CATS))
        text = random.choice(CATS[cat]).format(p=random.randint(2, 40), n=random.randint(5, 900))
        rows.append({"text": text, "label": cat})
    return rows
SYS = "Classify the financial disclosure into one risk category and explain briefly using evidence from the text."
def fmt(r):
    ans = f"Category: {r['label']}\nEvidence: \"{r['text']}\""
    return {"prompt": f"{SYS}\n\nDisclosure: {r['text']}", "completion": ans, "label": r["label"]}
os.makedirs("data", exist_ok=True)
for name, n in [("train", 2000), ("eval", 300)]:
    with open(f"data/{name}.jsonl", "w") as f:
        for r in make(n): f.write(json.dumps(fmt(r)) + "\n")
print("wrote data/train.jsonl, data/eval.jsonl")
