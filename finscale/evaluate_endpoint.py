import json, sys, urllib.request
base_url, model = sys.argv[1], sys.argv[2]
rows = [json.loads(l) for l in open("data/eval.jsonl")]
ok = 0
for r in rows:
    body = json.dumps({"model": model, "messages": [{"role": "user", "content": r["prompt"]}], "max_tokens": 80, "temperature": 0}).encode()
    req = urllib.request.Request(base_url + "/chat/completions", body, {"Content-Type": "application/json"})
    out = json.load(urllib.request.urlopen(req))["choices"][0]["message"]["content"]
    ok += r["label"] in out
print(json.dumps({"model": model, "n": len(rows), "accuracy": round(ok / len(rows), 3)}))
