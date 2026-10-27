import json, re, time
from pathlib import Path
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

USER_AGENT = "FinScale Research your.name@example.com"  # <-- EDIT THIS
HEADERS = {"User-Agent": USER_AGENT, "Accept-Encoding": "gzip, deflate"}

# Ticker -> CIK (10-digit zero padded). Add more as you like.
COMPANIES = {
    "AAPL": "0000320193", "MSFT": "0000789019", "JPM": "0000019617",
    "XOM": "0000034088", "PFE": "0000078003", "TSLA": "0001318605",
    "AMZN": "0001018724", "GS": "0000886982", "WMT": "0000104169",
    "BA": "0000012927", "NFLX": "0001065280", "INTC": "0000050863",
}
FILINGS_PER_COMPANY = 2
OUT = Path("data/raw"); OUT.mkdir(parents=True, exist_ok=True)

def get(url):
    time.sleep(0.2)  # SEC limit is 10 req/s; stay well under
    r = requests.get(url, headers=HEADERS, timeout=60)
    r.raise_for_status()
    return r

def list_10k_filings(cik, n):
    data = get(f"https://data.sec.gov/submissions/CIK{cik}.json").json()
    rec = data["filings"]["recent"]
    out = []
    for form, acc, doc, date in zip(rec["form"], rec["accessionNumber"],
                                    rec["primaryDocument"], rec["filingDate"]):
        if form == "10-K":
            out.append((acc, doc, date))
        if len(out) >= n:
            break
    return out

def extract_risk_factors(html):
    text = BeautifulSoup(html, "lxml").get_text("\n")
    text = re.sub(r"\n\s*\n+", "\n", text)
    # Find all "Item 1A" starts; the real section is the longest span
    # between an "Item 1A" heading and the next "Item 1B" / "Item 2".
    starts = [m.start() for m in re.finditer(r"item\s*1a\.?\s*risk\s*factors", text, re.I)]
    best = ""
    for s in starts:
        m = re.search(r"item\s*(1b|2)\.?\s", text[s + 200:], re.I)
        end = s + 200 + m.start() if m else min(len(text), s + 200000)
        chunk = text[s:end]
        if len(chunk) > len(best):
            best = chunk
    return best

def main():
    for ticker, cik in tqdm(COMPANIES.items(), desc="Companies"):
        try:
            filings = list_10k_filings(cik, FILINGS_PER_COMPANY)
        except Exception as e:
            print(f"[skip] {ticker}: {e}"); continue
        for acc, doc, date in filings:
            path = OUT / f"{ticker}_{date}.txt"
            if path.exists():
                continue
            url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{acc.replace('-', '')}/{doc}"
            try:
                rf = extract_risk_factors(get(url).text)
            except Exception as e:
                print(f"[skip] {ticker} {date}: {e}"); continue
            if len(rf) < 5000:
                print(f"[skip] {ticker} {date}: risk section too short"); continue
            path.write_text(rf)

if __name__ == "__main__":
    main()
