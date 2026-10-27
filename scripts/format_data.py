import json
SYSTEM = ("You are a financial risk analyst. Classify the risk disclosure excerpt "
          "and respond with a JSON object with keys: category, severity, evidence, explanation.")

for split in ["train", "eval"]:
    with open(f"data/{split}_raw.jsonl") as fi, open(f"data/{split}.jsonl", "w") as fo:
        for line in fi:
            r = json.loads(line)
            fo.write(json.dumps({"messages": [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": f"Excerpt:\n{r['excerpt']}"},
                {"role": "assistant", "content": json.dumps(r["label"])},
            ]}) + "\n")
