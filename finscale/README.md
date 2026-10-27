# FinScale
Financial-risk classifier: LoRA fine-tune of Qwen2.5-0.5B-Instruct, benchmarked 1 vs 2 GB10 nodes, served OpenAI-compatible.

## Local (Mac) smoke test
    python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
    python gen_data.py
    python train_gb10.py --epochs 1 --bs 2
## Lambda Labs fallback
See [LAMBDA_LABS.md](LAMBDA_LABS.md) for setup, single-/multi-node GPU training, serving, evaluation, and cleanup on Lambda Labs GPU Cloud.

## GB10 benchmark
    # 1 node
    bash scripts/launch_single.sh --out out1/adapter
    # 2 nodes (run on each, NODE_RANK=0/1, same MASTER_ADDR)
    NNODES=2 NODE_RANK=0 MASTER_ADDR=<head-ip> bash scripts/launch_multinode.sh --out out2/adapter
    python scripts/compare_scaling.py out1/train_summary.json out2/train_summary.json
## Fuzzball
Edit `<REGISTRY>`, `<TAG>` and every `TODO(fuzzball)` in `fuzzball/*.fz`, then submit.
## Image
    docker buildx build --platform linux/arm64 -t <REGISTRY>/finscale:<TAG> .
