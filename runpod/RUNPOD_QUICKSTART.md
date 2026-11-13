# FinScale on RunPod: Nemotron 3 Nano 30B-A3B LoRA quickstart

This uses the **NeMo AutoModel single-GPU LoRA recipe** (`nemotron_nano_v3_singlegpu_lora.yaml`), the same
path as the GB10 runbook, so a run that works here carries over to GB10 almost unchanged. It **replaces** the
Megatron-Bridge / `run_nemotron_sft.sh` / `configs/lora_sft.yaml` path. That path is not runnable, and
NVIDIA's recipe does not need 2×80 GB.

## 1. Create the pod

| Setting | Value |
| --- | --- |
| GPU | **1× H100 80GB** (NVIDIA's CI target for this recipe). For more headroom use **1× H200 141GB** or **1× RTX PRO 6000 96GB** |
| Container image | `nvcr.io/nvidia/nemo-automodel:26.10.00` (or the newest tag on NGC) |
| Container disk | 30 GB |
| Network / persistent volume | **≥ 200 GB** mounted at `/workspace` (model 59 GiB + HF cache + checkpoints) |
| Env vars | `HF_TOKEN` (RunPod secret), optionally `WANDB_API_KEY` |
| Shared memory | Use the default pod or raise `/dev/shm` if the dataloader complains |

RunPod hosts are **x86_64**. GB10 is **ARM64**. Do not copy compiled wheels between them. Throughput on an
H100 is **not** a GB10 result. Label it as a cloud baseline.

## 2. Bootstrap (inside the pod)

```bash
cd /workspace
git clone --branch Nemotron https://github.com/haranadhar/FinscaleNvidiaPlatform.git
# (or copy this runpod-setup branch over until it is merged)
bash /workspace/FinscaleNvidiaPlatform/runpod/setup_pod.sh 2>&1 | tee /workspace/setup.log
```

Copy the labeled data into the pod (it is not in Git):

```bash
# from your laptop:  runpodctl send finscale-edgar-sft-backup.tar.gz   (or scp to the pod's SSH port)
tar -C /workspace/FinscaleNvidiaPlatform -xzf /workspace/finscale-edgar-sft-backup.tar.gz
```

## 3. Fix the data (this is the P0 zero-supervised-token blocker)

```bash
cd /workspace/FinscaleNvidiaPlatform
M=/workspace/models/nemotron-30b
mkdir -p results/runpod

# (a) optional but recommended: company-held-out split (removes train/val context leakage)
python scripts/resplit_by_ticker.py --output-dir data/processed/by_ticker
D=data/processed/by_ticker      # or D=data/processed to keep the original split

# (b) measure lengths and keep only rows whose FULL conversation fits
for s in train val; do
  python scripts/fit_dataset_to_seq_len.py --model $M --input $D/$s.jsonl \
    --output data/processed/${s}_fit.jsonl --seq-length 4096 --mode drop \
    2>&1 | tee results/runpod/fit_$s.log
done

# (c) prove every answer survives tokenization at that length
python scripts/check_gb10_supervision.py --model $M --data data/processed/train_fit.jsonl \
  --lengths 4096 2>&1 | tee results/runpod/supervision.log      # must print PASS
```

If too many rows drop at 4096, rerun with `--mode trim`, which shortens only the source text, or use `--seq-length 6144`.

## 4. One real optimizer step

```bash
python scripts/prepare_gb10_config.py --model $M --seq-length 4096 --steps 1 \
  --output configs/nemotron_finscale_lora.yaml
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
set -o pipefail
/usr/bin/time -v automodel configs/nemotron_finscale_lora.yaml --nproc-per-node 1 \
  2>&1 | tee results/runpod/one_step.log
grep -iE "loss|step|error|out of memory" results/runpod/one_step.log | tail -20
```

To pass this gate, the log needs a finite loss and a completed step. Watch `nvidia-smi` in a second shell for peak memory.

## 5. Short real run with a saved adapter

```bash
python scripts/prepare_gb10_config.py --model $M --seq-length 4096 \
  --steps 40 --global-batch-size 8 --lr 1e-4 --val-every 10 \
  --save-checkpoint --checkpoint-dir /workspace/checkpoints/finscale-lora \
  --output configs/nemotron_finscale_lora_40.yaml
automodel configs/nemotron_finscale_lora_40.yaml --nproc-per-node 1 2>&1 | tee results/runpod/train_40.log
find /workspace/checkpoints/finscale-lora -name "*.safetensors" -o -name "adapter_config.json"
```

With about 150 rows, 40 steps at GBS 8 is roughly 2 epochs. Compare val loss with step 0, and compare
answers against the **frozen base model with the same prompt** before you claim any improvement.

## 6. Save evidence and stop paying

```bash
tar czf /workspace/finscale-runpod-evidence.tgz results/ configs/*.yaml /workspace/setup.log
# copy the adapter + evidence off the pod (runpodctl send / scp), then STOP the pod.
# The network volume keeps billing until you delete it.
```

## Optional: 2-GPU pod for the NCCL/multi-rank rehearsal

On a 2-GPU pod, `torchrun --nproc-per-node 2 scripts/nccl_multinode_proof.py` exercises the same script used for
the Fuzzball two-node proof. It runs on one host, so it does not prove inter-node networking.
