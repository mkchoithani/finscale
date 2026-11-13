# Rebuild the labeled dataset from scratch

Use this if the dataset made before the hackathon cannot be used, or is lost, and the data must be
rebuilt from nothing: no `data/` folder on disk and none in Git. It takes about **45 minutes** and
about **1,100 NVIDIA API calls**. Every known problem from the first build is already fixed in the
scripts, so no extra flags or workarounds are needed.

The first build (commit `f712592`, before the event) gave 986 labeled passages from 29 companies.
A rebuild gives similar but **not identical** numbers: SEC may have newer filings and the teacher
model answers slightly differently each time. That is normal.

---

## 1. Start clean

Open a **new Terminal window** (this clears old settings such as `FINSCALE_TEACHER`), then:

```bash
cd ~/hara-manish/FinscaleNvidiaPlatform        # or wherever the repo is cloned
git switch finscale-multinode
git pull mk finscale-multinode                 # "mk" = github.com/mkchoithani/finscale
```

If you are on a new machine:
```bash
git clone --branch finscale-multinode https://github.com/mkchoithani/finscale.git FinscaleNvidiaPlatform
cd FinscaleNvidiaPlatform
```
(On a fresh clone the remote is called `origin`, so use `origin` wherever this file says `mk`.)

## 2. Remove the old dataset (only if it must not be used)

From Git **and** from disk:
```bash
git rm -r --quiet data/passages.jsonl data/labels.jsonl data/sft
git commit -m "Remove pre-event dataset for a clean rebuild"
git push mk finscale-multinode
```
From disk, together with the Python environment and tokenizer files:
```bash
deactivate 2>/dev/null
rm -rf data .venv models
```
The old dataset is still in Git history (commit `f712592`), so nothing is lost for good.

## 3. Set up Python

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 4. Set your keys (type them in Terminal; never put them in a file in the repo)

```bash
export NVIDIA_API_KEY="nvapi-..."
export HF_TOKEN="hf_..."
export SEC_USER_AGENT="FinScale Hackathon mkchoithani@gmail.com"   # SEC requires a real contact
```
Check: `echo ${NVIDIA_API_KEY:0:6}` must print `nvapi-`.

## 5. Build the data

Run one command at a time and compare with the table in step 6.

```bash
python scripts/01_download_edgar.py                 # ~5 min
python scripts/02_make_passages.py                  # <1 min
python scripts/03_label_risk.py --limit 20          # quick test: 20 labels
python scripts/03_label_risk.py --workers 8         # ~15–20 min, all passages
python scripts/03_label_risk.py --retry-invalid     # ~2 min, second chance for rejected labels
python scripts/04_build_sft.py                      # train / val / test split by company
python -c "from huggingface_hub import snapshot_download as d; d('nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16', allow_patterns=['*.json','*.jinja','*.py'], local_dir='models/nemotron-tokenizer')"
python scripts/05_check_tokens.py --model models/nemotron-tokenizer
```

The 20-label test is worth a look: `head -3 data/labels.jsonl`. Each line should have
`"status": "ok"`, a sensible `risk_class` and `severity`, and a real sentence as `evidence_quote`.

## 6. What a good run looks like

| Step | Expected output | First build (reference) |
|---|---|---|
| 01 download | ~29 filings; only **XOM** reports a problem | 29 filings |
| 02 passages | ~1,000–1,100 passages; only the **WFC** warning | 1,036 passages |
| 03 test | `20/20 ok=20`, model `nvidia/nemotron-3-ultra-550b-a55b` | 20/20 ok |
| 03 full | ~93% ok; a few `invalid` (mostly "quote too long") | 968 ok |
| 03 retry | total valid ~95% | 986 valid |
| 04 build | train / val / test counts, 4 test companies | 728 / 106 / 152 |
| 05 check | `over 3072: 0` for train and val, then **PASS** | PASS |

XOM and WFC missing is expected (XOM: the SEC lookup does not find its 10-K; WFC: its risk text
is in a separate Exhibit 13). 29 companies is plenty.

## 7. Save the new dataset

```bash
git add data/passages.jsonl data/labels.jsonl data/sft
git commit -m "Rebuilt labeled dataset"
git push mk finscale-multinode
```
`data/raw/` (the downloaded reports, several hundred MB) stays local on purpose.
Every training run reads `data/sft/`, and `data/sft/manifest.json` records the hash of each file.

---

## If something goes wrong

| Symptom | Fix |
|---|---|
| `Set SEC_USER_AGENT ...` | Step 4: export `SEC_USER_AGENT` with a real email |
| `STOPPED: ... API error 404` | `python scripts/03_label_risk.py --probe`; pick a model that says OK; rerun with `--model <name>` |
| `api error 401` / every model FAILs in `--probe` | Key wrong or expired: re-export `NVIDIA_API_KEY` (must start with `nvapi-`) |
| Many `api error 429` | Rate limit: wait a minute and rerun the same command (it resumes), or use `--workers 4` |
| Labeling stopped half-way | Rerun the same command; finished labels are kept |
| `requires jinja2` | `pip install -r requirements.txt` (inside the activated `.venv`) |
| `05_check_tokens` FAIL, rows over 3072 | Rebuild passages shorter: `python scripts/02_make_passages.py --max-chars 3500`, then rerun 03 → 05 |
| A big company shows "no sections found" | Paste the warning to Claude; WFC is the only expected one |
| `PyTorch was not found` warning | Ignore: only the tokenizer is needed on the laptop |
