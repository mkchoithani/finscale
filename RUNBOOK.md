# FinScale runbook: SEC risk classifier, Nemotron 30B LoRA, 1 → 2 GB10 nodes

One operator runs every command; the team watches, records evidence, and checks each **gate**.
Do not move past a gate that fails. Every command is run from the repo root.

| Phase | Where | Time | Gate |
|---|---|---|---|
| A. Data | laptop (CPU + NVIDIA API) | ~1.5 h | `05_check_tokens.py` prints PASS |
| B. Cluster setup | Fuzzball | ~1 h | `01_smoke` PASS on a GB10 |
| C. NCCL | 2 nodes | ~30 min | `nccl_report.json` status PASS, 2 distinct hosts |
| D. 1-node baseline | 1 node | ~1–2 h | finite loss, adapter saved |
| E. 2-node training | 2 nodes | ~2–3 h | both nodes' logs show steps; tokens/s measured |
| F. Repeats + capacity | 1 and 2 nodes | ~3 h | ≥3 repeats each; optional capacity run |
| G. Evaluation | API + 1 node | ~1.5 h | scores for base vs fine-tuned |
| H. Charts + demo | laptop | ~1 h | charts in `results/charts/` |

---

## A. Data (laptop)

> **The labeled dataset is already committed** (commit `f712592`: 986 rows, 29 companies).
> If `data/` is deleted or damaged, restore it instead of rebuilding:
> `git checkout -- data/` (or `git restore data/`). Rebuilding gives a *different* dataset
> (newer SEC filings, slightly different teacher answers, different hashes), so only rebuild on purpose.
> Known, accepted gaps: XOM (10-K not found by the SEC lookup) and WFC (its risk text is in Exhibit 13).

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export NVIDIA_API_KEY=nvapi-...                 # never commit
export HF_TOKEN=hf_...
export SEC_USER_AGENT="FinScale Hackathon <your real email>"   # SEC rule: real contact
```

1. Download 30 companies' latest 10-K (~5 min):
   ```bash
   python scripts/01_download_edgar.py
   ```
2. Cut into passages from Items 1A, 1C, 3, 7, 7A (~1,000 tokens each, max 40 per company):
   ```bash
   python scripts/02_make_passages.py
   ```
   Expect roughly 900–1,200 passages. Watch the per-section counts; a company with no sections
   found is printed as a WARNING.
3. Label a **sample of 20 first** and read them:
   ```bash
   python scripts/03_label_risk.py --limit 20
   head -5 data/labels.jsonl | python -m json.tool --json-lines | less
   ```
   Check the class makes sense, the severity is reasonable and the quote is real text.
   Default teacher: `nvidia/nemotron-3-ultra-550b-a55b`. If labeling stops with "API error 404",
   run `python scripts/03_label_risk.py --probe` and pass a model that says OK with `--model <id>`.
4. Label everything (resumable; re-run after any failure; ~20–40 min):
   ```bash
   python scripts/03_label_risk.py --workers 8
   python scripts/03_label_risk.py --retry-invalid      # second pass for rejected labels
   ```
5. Build train/val/test, **split by company**:
   ```bash
   python scripts/04_build_sft.py
   ```
   Read the printed counts and class balance. Note the test tickers; they are never trained on.
6. **GATE A:** every answer must fit in the training length.
   ```bash
   python -c "from huggingface_hub import snapshot_download as d; d('nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16', \
       allow_patterns=['*.json','*.jinja','*.py'], local_dir='models/nemotron-tokenizer')"
   python scripts/05_check_tokens.py --model models/nemotron-tokenizer --seq-length 3072
   ```
   Must print `PASS`. (Real data: median ~1,100 tokens, longest ~2,950; answers 100–155 tokens.)
7. Team task during labeling: open `data/sft/gold_review.csv` and fill in `human_class` /
   `human_severity` where the teacher is wrong. This becomes the human-checked test set.
8. Commit the data so every node gets the identical files:
   ```bash
   git add data/passages.jsonl data/labels.jsonl data/sft && git commit -m "Labeled dataset" && git push
   ```

## B. Cluster setup (Fuzzball)

Ask CIQ staff (write the answers into the team notes):
1. Fuzzball UI/API address, login, version (need ≥ 4.2 for MCP).
2. ARM64 container image with CUDA PyTorch **and** NeMo AutoModel **and** `mamba-ssm` + `causal-conv1d`
   (Nemotron 3 Nano needs the Mamba kernels). If none exists: NGC `nvcr.io/nvidia/nemo-automodel:<tag>`
   for arm64, if available.
3. Shared persistent volume name and storage provisioner (`use:` value).
4. TCP port allowed between the two nodes (avoid 29500 unless approved), and the NIC name that links
   the two GB10s (for `NCCL_SOCKET_IFNAME`). Is RoCE/IB enabled between them?
5. **Memory limit for jobs.** GB10 memory is shared by CPU and GPU, so the job memory request must cover
   the ~59 GiB model plus training. Training Fuzzfiles request `100GB`; ask what the maximum is.
6. Do jobs have internet (GitHub, Hugging Face)?

Install the CLI and log in:
```bash
fuzzball context login                      # details from staff
claude mcp add fuzzball -s user -- fuzzball mcp serve --allow-exec   # optional: Claude drives Fuzzball
```

Fill in the Fuzzfiles, commit `fuzzball/cluster.env`:
```bash
IMAGE=docker://<image> VOLUME=<volume> PROVISIONER=<provisioner> PORT=<port> NCCL_IFACE=<nic> \
  bash fuzzball/render.sh
git add fuzzball/cluster.env && git commit -m "Cluster settings" && git push
```

Stage code, data and the model on the shared volume (~20–30 min for 59 GiB):
```bash
fuzzball workflow start fuzzball/rendered/00_setup.fz
fuzzball workflow list --tail 1
fuzzball workflow log <workflow-id> stage -f
```
**GATE B:**
```bash
fuzzball workflow start fuzzball/rendered/01_smoke.fz
fuzzball workflow log <workflow-id> preflight -f      # must end with "OK: 0 failure(s)"
```
If `mamba_ssm` or `nemo_automodel` is MISSING, the image is wrong: go back to staff before anything else.

## C. Two-node NCCL proof

```bash
fuzzball workflow start fuzzball/rendered/02_nccl_2node.fz
fuzzball workflow log <workflow-id> nccl-2node -f
```
**GATE C:** the log shows `"status": "PASS"` with **two different host names**, and a bandwidth table.
`results/nccl/nccl_report.json` is on the shared volume. In the NCCL INFO lines, note which transport
was used (`NET/IB` or `NET/Socket`) and the interface name; this goes on a slide.
If it hangs: wrong port or NIC. Fix `PORT` / `NCCL_IFACE`, re-render, push, re-run 00_setup (git pull), retry.

## D. Single-node baseline

Generate configs **on the laptop**, commit, push, then pull on the cluster (re-run `00_setup.fz`).
Model path is the shared-volume path.
```bash
export MODEL_DIR=/shared/models/nemotron-3-nano-30b
# 1-step smoke first
python scripts/06_make_config.py --mode single --run smoke-1n --steps 1 --no-save --val ''
# the real baseline (same GBS/seq as the 2-node runs!)
python scripts/06_make_config.py --mode single --run single-1n-r1 --steps 40 --gbs 16 --seq-length 3072
git add configs && git commit -m "Configs" && git push
```
Point the 1-node Fuzzfile at the smoke config (render.sh remembers the cluster values) and run it:
```bash
CONFIG_1N=configs/smoke-1n.yaml bash fuzzball/render.sh
fuzzball workflow start fuzzball/rendered/00_setup.fz     # git pull on the shared volume
fuzzball workflow start fuzzball/rendered/03_train_1node.fz
```
**GATE D (smoke):** the log shows `step 1 | ... | loss <finite number>` and exit=0.
- Out of memory / process killed → job memory limit (ask staff) or lower `--seq-length 2048` and drop the few longer rows.
- Kernel / Triton error in Mamba → save the log, ask staff for an image with working `mamba-ssm`.
Then run `single-1n-r1` (40 steps). Watch `results/runs/single-1n-r1/summary.json`: median tokens/s,
loss going down, adapter saved under `results/runs/single-1n-r1/ckpt/LATEST/model/`.

## E. Two-node training (the core result)

```bash
python scripts/06_make_config.py --mode replicate --nodes 2 --run replicate-2n-r1 --steps 40 --gbs 16 --seq-length 3072
git add configs && git commit -m "2-node config" && git push
CONFIG_2N=configs/replicate-2n-r1.yaml bash fuzzball/render.sh
fuzzball workflow start fuzzball/rendered/00_setup.fz     # git pull on the shared volume
fuzzball workflow start fuzzball/rendered/04_train_2node.fz
```
**GATE E:** `results/runs/replicate-2n-r1/logs/` has **one log per host**, both showing training,
`nodes/*.exit.json` exit 0 on both, and `summary.json` has median tokens/s.

Why `replicate` first: each GB10 holds a full copy of the 30B model (it fits in 128 GB), and only the
small LoRA gradients cross the network, so scaling should be close to 2×.
If `replicate` fails at startup, try `--mode ddp`, then `--mode shard`.

## F. Repeats and the capacity story

Throughput claims need ≥3 repeats of each setting:
```bash
for r in r2 r3; do
  python scripts/06_make_config.py --mode single --run single-1n-$r --steps 40 --gbs 16 --no-save
  python scripts/06_make_config.py --mode replicate --nodes 2 --run replicate-2n-$r --steps 40 --gbs 16 --no-save
done
```
**Capacity (optional, after E works):** find a setting one node cannot run, then run it on two.
- `shard` puts half the weights on each node → try `--seq-length 8192` or `--mbs 2` on 1 node
  (expect out of memory) and on 2 nodes with `--mode shard` (expect success).
- `ep` splits MoE experts across nodes: `--mode ep --nodes 2`.
Save the failing 1-node log too: it is the evidence that the second node unlocked something.

## G. Evaluation (base vs fine-tuned, held-out companies)

Run both on a GB10 (inside a 1-node Fuzzball job or interactive shell): the same 30B weights,
first without and then with the adapter, same prompt as training. (The NVIDIA API does not serve
`nemotron-nano-3-30b-a3b` to our key, and same-weights is the fairer comparison anyway.)
```bash
python scripts/07_predict.py --name base --backend hf --model-path $MODEL_DIR
```
Optional extra reference, the teacher itself: `--name teacher --backend api --model nvidia/nemotron-3-ultra-550b-a55b`.
```bash
pip install "peft>=0.19.1"
python scripts/07_predict.py --name finetuned --backend hf --model-path $MODEL_DIR \
    --adapter results/runs/single-1n-r1/ckpt/LATEST/model
```
(If the vLLM image supports Nemotron 3 + LoRA on ARM64, serving is faster, see the script help.)
Score:
```bash
python scripts/08_score.py results/eval/pred-base.jsonl results/eval/pred-finetuned.jsonl
python scripts/08_score.py results/eval/pred-*.jsonl --gold data/sft/gold_review.csv --out results/eval/scores_gold.json
```

## H. Charts and summary

```bash
python scripts/plot_results.py --runs results/runs/single-1n-r* results/runs/replicate-2n-r* \
    --scores results/eval/scores.json --nccl results/nccl/nccl_report.json
```
Produces `results/charts/throughput.png`, `loss_curves.png`, `memory_per_node.png`, `eval_quality.png`,
`nccl_bandwidth.png` and `summary.json` (speedup, scaling efficiency).
Commit `results/` (logs, CSV, JSON, charts; adapters are ignored).

---

## What the team watches and records

- Screenshot the Fuzzball UI showing **two hosts** allocated for the 2-node jobs.
- For each run: workflow ID, config file, git commit, start/end time (in `results/runs/<run>/nodes/`).
- Any failure: copy the last 50 log lines into the notes with what was changed to fix it.
- Gold review: correct `data/sft/gold_review.csv` (A.7).

## Claims we may make (and only these)

| Claim | Evidence needed |
|---|---|
| "Training runs across 2 GB10 nodes" | per-host logs from one 2-node run, both exit 0 |
| "2 nodes give X× throughput" | ≥3 repeats each, same GBS/seq, `summary.json` speedup |
| "2 nodes unlock Y" | failing 1-node log + succeeding 2-node log, same settings |
| "Fine-tuning improves risk classification" | `scores.json` base vs fine-tuned on held-out companies |

## Known issues and fixes (from the RunPod rehearsal, 13 Nov 2026)

Rehearsal on 1× H100 80GB, image `nvcr.io/nvidia/nemo-automodel:26.10.00` (amd64; an arm64 build of the
same tag exists for GB10). Evidence: [results/rehearsal-runpod/](results/rehearsal-runpod/).
Versions that worked: AutoModel 0.7.0, transformers 5.18.0, torch 2.14 (CUDA 13.4), mamba_ssm 2.3.0,
causal_conv1d 1.6.0, peft 0.20.0. **If the event image reports different versions, run `smoke` first.**

| Symptom (exact text to look for) | Cause | Status |
|---|---|---|
| `ChatDatasetConfig got unexpected dataset config field(s): chat_template_kwargs` | AutoModel 0.7.0 lacks the field | Fixed in `06_make_config.py` (field omitted; template renders identically) |
| `cache_position[-1] ... 'NoneType' object is not subscriptable` in `modeling_nemotron_h.py` during generate | Model repo's remote code vs transformers 5.18 | Fixed in `07_predict.py` (uses built-in `NemotronHForCausalLM`, falls back to remote code) |
| Model download "stuck": `.incomplete` files not growing, 0 MB/s network, process still busy | Hugging Face xet download reassembling files; slower without a token | Not an error: wait. Set `HF_TOKEN`; start the download first. Took 24 min unauthenticated |
| Step 0 takes ~60 s, later steps ~8 s | One-time warm-up / kernel compile | Normal. `parse_train_log.py` excludes the first 3 steps from throughput |
| `torchrun ... ChildFailedError` at the end of a log | Only the wrapper reporting a child died | Search ~30 lines above for the real `Error:` line |

Measured on H100 (reference only, **not** GB10 numbers): peak memory 61–64 GiB at seq 3,072;
~2,200 training tokens/s; 8 steps (GBS 16) ≈ 2.5 min incl. load; loss 1.17 → 0.56, val loss 0.50.
Eval, 60 held-out rows: class accuracy 0.68 → 0.82, macro-F1 0.49 → 0.83, severity 0.47 → 0.70,
JSON valid 0.98 → 1.00, quote fidelity 0.97 → 0.93.
