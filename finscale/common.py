"""Shared schema, prompts, validation and file helpers.

Training, labeling and evaluation all import the prompts from here, so the
student sees exactly the same instructions at train time and at eval time.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Iterable

RISK_CLASSES = [
    "market",
    "credit",
    "liquidity",
    "operational",
    "regulatory_legal",
    "cybersecurity",
    "supply_chain",
    "other",
]
SEVERITIES = ["low", "medium", "high"]

CLASS_GUIDE = {
    "market": "prices, rates, FX, commodity, equity or demand/competition swings that change revenue or valuation",
    "credit": "counterparty or customer default, receivables, lending losses, credit ratings",
    "liquidity": "cash, funding, debt maturities, covenants, access to capital, refinancing",
    "operational": "execution, people, facilities, IT outages (non-security), product quality, internal processes",
    "regulatory_legal": "laws, regulation, litigation, investigations, compliance, tax, government action",
    "cybersecurity": "security breaches, attacks, data privacy incidents, cyber risk governance",
    "supply_chain": "suppliers, sourcing, manufacturing partners, logistics, component shortages",
    "other": "no material risk, or a risk that fits none of the classes above",
}
SEVERITY_GUIDE = {
    "low": "generic or boilerplate exposure; unlikely or small impact",
    "medium": "specific exposure with a plausible, moderate financial impact",
    "high": "specific, likely or already-occurring exposure that could materially hurt results or solvency",
}

ANSWER_KEYS = ["risk_class", "severity", "evidence_quote", "rationale"]
MIN_QUOTE_CHARS = 20


def _guide_text() -> str:
    classes = "\n".join(f"- {k}: {v}" for k, v in CLASS_GUIDE.items())
    sev = "\n".join(f"- {k}: {v}" for k, v in SEVERITY_GUIDE.items())
    return f"Risk classes:\n{classes}\n\nSeverity:\n{sev}"


# Student prompt: used for SFT rows AND for evaluating the base / fine-tuned model.
STUDENT_SYSTEM = (
    "You are FinScale, a financial risk analyst. Read one passage from an SEC 10-K filing and return the "
    "single most material risk it discloses as one JSON object with keys risk_class, severity, "
    "evidence_quote, rationale. evidence_quote must be copied word for word from the passage. "
    "Return only the JSON object.\n\n" + _guide_text()
)

# Teacher prompt: more detailed rules for the large labeling model.
TEACHER_SYSTEM = (
    "You label SEC 10-K passages for a financial risk classifier. Follow the taxonomy exactly.\n\n"
    + _guide_text()
    + "\n\nRules:\n"
    "1. Pick the ONE most material risk in the passage.\n"
    "2. evidence_quote: one sentence or clause copied EXACTLY from the passage (at least 20 characters, "
    "at most 300). Do not paraphrase, fix typos, or join separate sentences.\n"
    "3. rationale: one short sentence explaining the class and severity.\n"
    "4. If the passage is boilerplate, a table fragment, or has no real risk, use risk_class \"other\" "
    "and severity \"low\".\n"
    "5. Output ONLY a JSON object with keys risk_class, severity, evidence_quote, rationale. "
    "No markdown, no extra text."
)


def student_user_prompt(p: dict) -> str:
    return (
        f"Company: {p['ticker']} | Filing date: {p['filing_date']} | Section: Item {p['item']} "
        f"({p.get('item_title', '')})\n\nPassage:\n{p['text']}"
    )


def student_messages(p: dict, answer: str | None = None) -> list[dict]:
    msgs = [
        {"role": "system", "content": STUDENT_SYSTEM},
        {"role": "user", "content": student_user_prompt(p)},
    ]
    if answer is not None:
        msgs.append({"role": "assistant", "content": answer})
    return msgs


def canonical_answer(label: dict) -> str:
    """Fixed key order so every training target has the same shape."""
    return json.dumps({k: label[k] for k in ANSWER_KEYS}, ensure_ascii=False)


# ---------------------------------------------------------------- parsing / validation
_THINK_RE = re.compile(r"<think>.*?</think>", re.S | re.I)
_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.M)


def parse_json_answer(text: str | None) -> dict | None:
    """Extract the first JSON object from a model reply. Returns None if there is none."""
    if not text:
        return None
    t = _THINK_RE.sub("", text)
    if "</think>" in t:  # template that omits the opening tag
        t = t.split("</think>", 1)[1]
    t = _FENCE_RE.sub("", t.strip())
    start = t.find("{")
    while start != -1:
        depth, in_str, esc = 0, False, False
        for i in range(start, len(t)):
            c = t[i]
            if in_str:
                if esc:
                    esc = False
                elif c == "\\":
                    esc = True
                elif c == '"':
                    in_str = False
            elif c == '"':
                in_str = True
            elif c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    try:
                        obj = json.loads(t[start : i + 1])
                        return obj if isinstance(obj, dict) else None
                    except json.JSONDecodeError:
                        break
        start = t.find("{", start + 1)
    return None


def norm_text(s: str) -> str:
    s = s.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    s = s.replace("—", "-").replace("–", "-").replace("\xa0", " ")
    return re.sub(r"\s+", " ", s).strip().lower()


def quote_in_passage(quote: str, passage: str) -> bool:
    q = norm_text(quote).strip(" .\"'")
    return len(q) >= MIN_QUOTE_CHARS and q in norm_text(passage)


def validate_label(obj: dict | None, passage: str) -> tuple[dict | None, str]:
    """Return (normalized_label, "") if valid, else (None, reason)."""
    if not isinstance(obj, dict):
        return None, "no JSON object"
    missing = [k for k in ANSWER_KEYS if k not in obj]
    if missing:
        return None, f"missing keys {missing}"
    rc = str(obj["risk_class"]).strip().lower().replace(" ", "_").replace("/", "_")
    rc = {"regulatory": "regulatory_legal", "legal": "regulatory_legal", "cyber": "cybersecurity",
          "supply": "supply_chain"}.get(rc, rc)
    sev = str(obj["severity"]).strip().lower()
    if rc not in RISK_CLASSES:
        return None, f"bad risk_class {obj['risk_class']!r}"
    if sev not in SEVERITIES:
        return None, f"bad severity {obj['severity']!r}"
    quote = str(obj["evidence_quote"]).strip()
    if len(quote) > 400:
        return None, "quote too long"
    if not quote_in_passage(quote, passage):
        return None, "quote not found verbatim in passage"
    rationale = str(obj["rationale"]).strip()
    if not rationale:
        return None, "empty rationale"
    return {"risk_class": rc, "severity": sev, "evidence_quote": quote, "rationale": rationale[:400]}, ""


# ---------------------------------------------------------------- IO helpers
def read_jsonl(path: str | Path) -> list[dict]:
    p = Path(path)
    if not p.exists():
        return []
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def write_jsonl(path: str | Path, rows: Iterable[dict]) -> int:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with p.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    return n


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


DEFAULT_TICKERS = [
    "AAPL", "NVDA", "MSFT", "AMZN", "GOOGL", "META", "TSLA", "JPM", "BAC", "WFC",
    "GS", "MS", "V", "MA", "PYPL", "NFLX", "DIS", "WMT", "COST", "HD",
    "UNH", "JNJ", "PFE", "ABBV", "XOM", "CVX", "BA", "CAT", "AMD", "INTC",
]
