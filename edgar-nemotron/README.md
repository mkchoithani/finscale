# edgar-nemotron

SEC 10-K data preparation starter plus two intentionally separate SFT paths: (1) documented Nemotron 3 Nano SFT recipe on NeMo + Megatron-Bridge, and (2) small Hugging Face/PEFT fallback. Dataset generation is free/SEC-public; training may incur GPU/provider charges. Never commit filings, tokens, adapters, or checkpoints.

## Verified Nemotron repository findings

Inspected public GitHub checkout `NVIDIA-NeMo/Nemotron` on 2026-11-01 (shallow clone, HEAD `6442e0f19245d3a1bf71fcd2d441f79613ed2882`). Top-level folders include `src/nemotron/{steps,recipes,data_prep}`, `docs/nemotron`, `usage-cookbook`, `use-case-examples`, `skills`, `tests`, `scripts`, `research`, `deploy`, and docs. README describes open training recipes, modular `nemotron steps` CLI, deployment cookbooks, and agent examples.

The repo README lists recipes for Nemotron 3 Nano (31.6B/3.6B active), 3 Super (120.6B/12.7B), 3 Ultra (550B/55B), 3.5 Super, 3.5 Lightning, 3.5 Transcribe, 3 Nano Omni, Embed, and Rerank. The Nano guide explicitly links `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-Base-BF16`, `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16`, and FP8 ID. Super guide includes `nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-Base-BF16`; Ultra uses `nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16`. Repo cookbook also names the smaller legacy HF checkpoint `nvidia/Llama-3.1-Nemotron-Nano-8B-v1`, used as fallback here. Repo release README also links Super 3.5 VL / Lightning; see source docs for evolving releases.

Nano SFT uses Megatron-Bridge `finetune()` through `uv run nemotron nano3 sft` (the repo install guide says `uv sync`); Nano RL uses NeMo-RL/Ray. Nano SFT preparation converts OpenAI-style `messages` with `role`/`content` into chat-template tokenized, role-masked and packed sequences (default pack length 4096) stored as packed Parquet; docs distinguish `input_ids`, `loss_mask`, and `seq_start_id`. Other pipelines use raw JSONL, Parquet, and for pretraining Megatron bin/idx. A cookbook demonstrates LoRA/PEFT via Megatron-Bridge `torchrun ... scripts/training/run_recipe.py --recipe nemotron_omni_cord_v2_peft_config ...`, rank 16 / alpha 32; its visual/multimodal recipe is not a verified fit to Nano 3 or the EDGAR text-only task. Nano docs prerequisites state Slurm GPU nodes, H100 recommended; their complete recipe and cookbook examples are multi-GPU (cookbook is 8xH100 80GiB, 600GB checkpoint disk). Full Nano/Super recipes are not a one-GPU 8B-type starter. `configs/lora_sft.yaml` marks cross-recipe fields TODO.

Commands actually shown in checked-in Nano docs include `uv run nemotron nano3 data prep sft --run YOUR-CLUSTER`, `uv run nemotron nano3 sft --run YOUR-CLUSTER`, and `uv run nemotron nano3 sft --help`; CLI supports `-c/--config`, `-r/--run`, `-d/--dry-run`, and config overrides. The documented command consumes registered Packed-Parquet/W&B artifacts, not this project's raw chat JSONL; integration/import details must be completed and checked against repository revision. Wrapper intentionally refuses to launch unless explicitly enabled and will fail until required artifacts/profile are wired. No invented CLI flag is used.

## Run locally / in a GPU pod

Python 3.10+ (repo itself declares Python >=3.11,<3.14). Use matching NVIDIA CUDA PyTorch image for GPU work.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python scripts/smoke_test_cuda.py
export SEC_USER_AGENT='Your Name your-real-contact@example.org'  # required SEC identity; do not impersonate
python scripts/download_edgar.py --tickers AAPL MSFT JPM --limit 1
python scripts/build_sft_dataset.py
# Optional fallback; first ensure this legacy 8B model is available/compatible with installed Transformers.
HF_TOKEN=... torchrun --nproc_per_node=1 scripts/train_hf_lora.py --model nvidia/Llama-3.1-Nemotron-Nano-8B-v1
```

SEC `data.sec.gov` submissions and `www.sec.gov/Archives` are public/free endpoints. A truthful identifying User-Agent with contact is required; respect SEC fair-access guidance and do not raise request rate to evade limits. Filings can be large; section heuristics may not identify every item. The builder emits extractive opening-sentence targets (not human summaries): inspect/relabel and de-duplicate before serious training. The HF fallback is independent from the official Nemotron Megatron recipe, and the cited 8B ID is an older Llama-based Nano checkpoint, not Nano 3; compatibility, VRAM fit, PEFT modules, dtype, and license/access must be verified. `torchrun` compatible; throughput log is coarse steps/sec, not tokens/sec.

## Data formats

- Downloader: one JSON object/line with `ticker`, filing metadata, and raw text.
- Builder: one chat object/line with `messages: [{role, content}, ...]` and metadata; deterministic shuffled train/val files.
- Official Nano 3 SFT: chat messages ultimately prepared as packed Parquet with input IDs/loss masks/sequence boundaries through repo tooling.
- `to_binary.py`: explicitly a custom flat NumPy token stream and JSON offsets sidecar; **not** Megatron's indexed `.bin/.idx`, and not interoperable. Use the repo recipe's actual data prep/preprocess tool after selecting the exact version/config. TODO: wire direct converter only after verifying tool and invocation in the chosen checkout.

## Next steps for official recipe

1. `git clone https://github.com/NVIDIA-NeMo/Nemotron.git /workspace/Nemotron` and follow that revision's README.
2. Follow `docs/nemotron/nano3/README.md`, install with `uv sync`, set W&B, `env.toml`, and Slurm/NeMo container as appropriate.
3. Transform and register EDGAR conversations as the repo's Packed-Parquet `SFTDataArtifact`; align tokenizer/template, role loss masks, and artifacts. No raw JSONL CLI override is documented for the checked-in recipe.
4. Verify actual LoRA recipe/config interface against the selected Megatron-Bridge version and model. Run documented `uv run nemotron nano3 sft --help`, then official `--dry-run`/configured recipe before launching. TODO: this starter does not assert the YAML is executable.
5. Run `bash scripts/run_nemotron_sft.sh` only after fulfilling its README TODOs and setting `RUN_NEMOTRON_SFT=1`.

## Existing finscale.zip ideas reused

Inspected earlier `finscale.zip`: kept lightweight standalone stages, a generated train/eval split concept, GPU LoRA fallback, output ignores, and cloud-pod guidance. Its synthetic text-generation approach is not used to fabricate SEC disclosures; sample labels here derive from source text and are marked pseudo-labels.
