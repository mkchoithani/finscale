# FinScale: SEC 10-K risk classification on a multi-node GB10 cluster

Fine-tunes **NVIDIA Nemotron 3 Nano 30B-A3B** (LoRA, NeMo AutoModel) to read a 10-K passage and return

```json
{"risk_class": "liquidity", "severity": "high",
 "evidence_quote": "<sentence copied from the filing>", "rationale": "<one sentence>"}
```

with `risk_class` ∈ market, credit, liquidity, operational, regulatory_legal, cybersecurity, supply_chain, other
and `severity` ∈ low, medium, high. Training is scaled from one to two **GB10** nodes, scheduled with **CIQ Fuzzball**.

**Track:** Multi-Node Training. **What the second node unlocks:** higher training throughput
(data-parallel replicas, only LoRA gradients cross the network) and, with sharding / expert parallelism,
settings that do not fit on one node.

## Pipeline

| Step | Script | Output |
|---|---|---|
| 1 | `scripts/01_download_edgar.py` | latest 10-K text for 30 companies |
| 2 | `scripts/02_make_passages.py` | ~1,000-token passages from Items 1A, 1C, 3, 7, 7A |
| 3 | `scripts/03_label_risk.py` | teacher labels from Nemotron 3 Ultra 550B (NVIDIA API), quote-verified |
| 4 | `scripts/04_build_sft.py` | train / val / test split **by company**, gold review sheet, SHA-256 manifest |
| 5 | `scripts/05_check_tokens.py` | gate: every answer survives tokenization at the training length |
| 6 | `scripts/06_make_config.py` | AutoModel config: `single`, `replicate`, `shard`, `ep`, `ddp` |
| 7 | `scripts/launch_train.sh` | torchrun on 1 or N nodes + per-node manifest, telemetry, logs |
| 8 | `scripts/07_predict.py`, `08_score.py` | base vs fine-tuned on held-out companies |
| 9 | `scripts/plot_results.py` | throughput, scaling efficiency, loss, memory, quality, NCCL charts |

Cluster side: `fuzzball/templates/*.fz` (setup, smoke, 2-node NCCL, 1-node and 2-node training),
filled in by `fuzzball/render.sh`. `scripts/nccl_bench.py` proves and measures cross-node NCCL.

**Start with [RUNBOOK.md](RUNBOOK.md).**

## Design rules

- The student sees the same prompt at training and evaluation (`finscale/common.py`).
- Passages are short, so the JSON answer is never truncated away (checked by step 5).
- No company appears in more than one of train / val / test.
- 1-node and 2-node runs use the same global batch size, sequence length, data and seed.
  Only measured numbers go on charts.
