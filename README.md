## Hackathon master guide (Nov 13–15, 2026)

**[FINScale end-to-end master runbook](FINSCALE_HACKATHON_MASTER_RUNBOOK.md)** — business problem, model and LoRA choices, actual GB10 challenges, end-to-end single-node commands, Fuzzball CLI, two-node NCCL proof, multi-node model-training gates, performance metrics and demo. **[Fuzzball starter YAML](fuzzball/)** requires organizer-provided ARM64 CUDA image and staging; no two-node 30B optimizer step has yet been validated.

> **Hackathon Nov 13 update:** [GB10 verified training runbook](GB10_HACKATHON_RUNBOOK.md) documents the successful ARM64 Mamba/Causal CUDA builds, 59 GiB Nemotron 30B download, remaining **zero-supervised-token blocker**, and the real one-step LoRA launch. [Fuzzball two-node playbook](FINSCALE_FUZZBALL_TWO_NODE.md) includes rank proof, two-host workflow gates, fair performance measurement, and charts. New scripts: [supervision check](scripts/check_gb10_supervision.py), [recipe preparation](scripts/prepare_gb10_config.py), [NCCL two-host proof](scripts/nccl_multinode_proof.py), [training plots](scripts/plot_training_comparison.py). **No Nemotron 30B training step or two-node model training has been verified yet.**

# FinScale — Nemotron 3 Nano 30B-A3B + SEC EDGAR

**Branch:** `Nemotron` | **Student model:** `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16` | **Dataset:** SEC EDGAR 10-K filings from a target universe of 30 US public companies.

**GPUWerk DGX Spark GB10 (1×GB10, 128GB unified memory, 1TB shared storage):** **[GPUWERK_GB10.md](GPUWERK_GB10.md)** — GPUWerk SSH/container constraints, full system preflight, SEC restore/regeneration, device/process monitoring with `nvidia-smi dmon` and `pmon`, bounded GPU-compute benchmark, CSV collection, graph generation, and **30B Nano training readiness/gating**. Companion scripts: [`monitor_gpuwerk.py`](scripts/monitor_gpuwerk.py), [`benchmark_gpuwerk.py`](scripts/benchmark_gpuwerk.py), [`plot_gpuwerk.py`](scripts/plot_gpuwerk.py). **The 30B training recipe is still not yet fully wired; do not treat the GPU benchmark as a successful SFT run.**

**Start here on Lambda Labs:** **[LAMBDA.md](LAMBDA.md)** — this now contains the full, ordered commands for (a) backing up the existing teacher-labeled dataset, (b) provisioning a replacement 2 × 80GB GPU VM, (c) restoring labeled JSONL data, and (d) checking the NVIDIA official Nano3 recipe prerequisites. **RunPod:** [RUNPOD.md](RUNPOD.md).

> **Existing successful run (previous Lambda A100 40GB):** NVIDIA API teacher `nvidia/nemotron-3.5-lightning-30b-a3b` generated **166 train + 18 validation = 184 examples**. This run is **not** a completed 30B student LoRA training job. **These JSONL files are NOT in GitHub. Back up `data/raw/` and `data/processed/` from the old VM before terminating it.** If the old VM is already gone and no backup exists, regenerate the dataset from SEC EDGAR and NVIDIA API on the replacement VM.

## GPU choices for 30B and the DGX Spark hackathon

**Do not confuse GPU architecture with memory topology.** The hackathon's DGX Spark uses the **GB10 Grace Blackwell Superchip** with **128GB unified system memory** and an **ARM64 CPU**. A cloud **RTX PRO 6000 Blackwell 96GB** is also Blackwell, but uses **dedicated GDDR7 VRAM** and typically an **x86_64 host**; the two are not performance- or binary-equivalent. DGX Spark's unified memory is not 128GB of conventional dedicated GPU VRAM. Cross-architecture Docker images and compiled CUDA extensions may need separate ARM64 and x86_64 builds.

| GPU / configuration | Memory | Guidance for FinScale Nemotron 3 Nano 30B |
| --- | --- | --- |
| **2 × A100 80GB on one VM** | 80GB per GPU; 160GB aggregate (not pooled) | **Preferred practical Lambda choice** for the NVIDIA-documented two-80GB LoRA topology, subject to supported image, parallelism and actual availability. |
| **2 × RTX PRO 6000 Blackwell 96GB on one VM** | 96GB per GPU; 192GB aggregate (not pooled) | **Preferred Blackwell cloud comparison** when available on RunPod; validate NCCL/peer-to-peer links, CUDA/NeMo support and EP=2. Do not assume same performance as GB10. |
| **1 × RTX PRO 6000 Blackwell 96GB** | 96GB dedicated | Useful for Blackwell inference and quantized experiments; **not verified** for the published two-GPU BF16 LoRA recipe. |
| **1 × RTX 6000 Ada 48GB** | 48GB dedicated | Not suitable for the current unsharded BF16 30B training recipe; use only with a separately validated quantized/offload approach. |
| **1 × A100 40GB** | 40GB dedicated | Previous VM; good for SEC/API dataset preparation, **not** the existing BF16 30B LoRA recipe. |
| **1–2 × DGX Spark GB10** | 128GB unified per device (typical) | Hackathon hardware; optimize for ARM64, memory bandwidth, inter-node communication, and a separately validated quantized/PEFT path. Not a direct reproduction of the A100 EP=2 recipe. |

**No H100 is required or recommended for this plan.** NVIDIA's published Nano3 30B LoRA reference is a **two-80GB** configuration with expert parallelism **EP=2**; hardware fit is a prerequisite, **not** a guarantee that the current FinScale wrapper is runnable. Do not add VRAM across GPUs and assume a single process can use the sum. Verify supported device count, CUDA architecture, distributed training topology, RAM, disk, and network before renting.

**Provider discovery:** Lambda GPU inventory/pricing: https://lambda.com/pricing ; RunPod GPU inventory/pricing: https://www.runpod.io/gpu-models . Listings are not a guarantee of live availability. RTX **6000 Ada (48GB)**, older **Quadro RTX 6000 (24GB)**, and **RTX PRO 6000 Blackwell (96GB)** are **different products**.

### Record hardware and portability baseline

Run on **each** Lambda/RunPod/DGX Spark machine, and save the output with your benchmark:

```bash
uname -m
cat /etc/os-release | head -n 5
nvidia-smi -L
nvidia-smi --query-gpu=index,name,memory.total,memory.free,compute_cap --format=csv || nvidia-smi
nvidia-smi topo -m
nvcc --version || true
python3 - <<'PY'
import platform
print("CPU architecture:", platform.machine())
try:
    import torch
    print("torch:", torch.__version__, "CUDA:", torch.version.cuda)
    print("CUDA available:", torch.cuda.is_available())
    for i in range(torch.cuda.device_count()):
        print(i, torch.cuda.get_device_name(i))
except ImportError:
    print("torch not installed in current Python environment")
PY
```

For 2-GPU jobs, also verify `nvidia-smi topo -m` and run NCCL/distributed smoke tests **before** training. Two GPUs on different pods or instances are **not** automatically equivalent to two GPUs inside one machine.

## Previously encountered issues and verified fixes

| Problem observed on old Lambda A100 40GB | Cause | Correct action |
| --- | --- | --- |
| `git clone ... /workspace/Nemotron` → `Permission denied` | Lambda `ubuntu` could not create directories under RunPod-style `/workspace` | Clone official NVIDIA repo into **`$HOME/Nemotron`**. |
| `cd ~/FinscaleNvidiaPlatform` → no directory | NVIDIA repo had been cloned, but user's FinScale repo had not | Clone **both** repos separately into `$HOME`. |
| `scripts/run_nemotron_sft.sh: No such file` | Executing from the official NVIDIA repo, not FinScale | `cd "$HOME/FinscaleNvidiaPlatform"` before running project scripts. |
| `VIRTUAL_ENV ... does not match project environment` | FinScale venv active while running `uv` in NVIDIA repo | `deactivate 2>/dev/null || true; unset VIRTUAL_ENV` before NVIDIA `uv sync`. |
| `Not launching: set up W&B Packed-Parquet SFT artifact and env.toml` | Training wrapper is intentionally guarded; the recipe integration is incomplete | Prepare real Nano3 packed SFT artifact, checkpoint, valid executor/profile and verified LoRA settings. **Do not merely set the bypass flag.** |
| `SyntaxError: unterminated string literal` in `download_edgar.py` | Broken generated `print` line; SEC ticker directory also overwrote ticker list | Fixed on `Nemotron` branch (commit `f420c13`). On new VM use latest branch and `python -m py_compile scripts/*.py`. |
| Confusion about `--workers 2` with one GPU | Workers are concurrent NVIDIA **hosted API requests** | `--workers 2` works on a single GPU/CPU VM; it does **not** require two GPUs. |
| A100 40GB and 30B BF16 LoRA | ~60GB base BF16 weights before training overhead, and official two-GPU topology | Use **2 × A100 80GB** or investigate **2 × RTX PRO 6000 Blackwell 96GB**, with recipe validation. |
| Replacing VM risks losing 184 labeled examples | GitHub excludes `data/raw` and `data/processed` | Back up **166 train + 18 validation**, plus raw EDGAR JSONL, off the VM **before termination**. |

**Previously completed milestone:** 30-company EDGAR extraction and NVIDIA-hosted `nvidia/nemotron-3.5-lightning-30b-a3b` teacher generation produced **166 train / 18 validation / 184 total**. **This was data generation, not local 30B fine-tuning.**

## Hardware guidance

- **Previous:** single A100 40GB (suitable for preparation/teacher API labeling, *not* the published BF16 Nano3 30B LoRA recipe).
- **Recommended Lambda starting configuration:** **2 × A100 80GB on Lambda** or **2 × RTX PRO 6000 Blackwell 96GB on RunPod**, subject to availability and validation. No H100 is needed. The NVIDIA Nemotron 3 Nano NeMo Helix reference documents **LoRA: 2 × 80GB, TP=1, EP=2, PP=1**. This is a reference topology, not a guarantee that every stock Lambda image runs the recipe unchanged.
- **Do not confuse teacher vs. student:** API teacher = **Nemotron 3.5 Lightning 30B-A3B**; locally fine-tuned student = **Nemotron 3 Nano 30B-A3B**. `--workers 2` controls concurrent NVIDIA hosted API calls, **not** GPU count.
- Verify GPU devices, VRAM, storage, CUDA drivers, compatible training image, official recipe execution method, and costs before GPU training.

## Project scripts

| Path | Purpose | Status |
| --- | --- | --- |
| `scripts/download_edgar.py` | Fetch one or more SEC 10-K filings per company | Operational after syntax/ticker-list fix |
| `scripts/validate_edgar.py` | Audit coverage of 30 requested companies | Operational |
| `scripts/build_sft_dataset.py` | Extract filing sections and produce teacher-labeled OpenAI-style messages | Worked on previous Lambda instance |
| `scripts/smoke_test_cuda.py` | Check CUDA/PyTorch | Environment diagnostic |
| `scripts/run_nemotron_sft.sh` | Guarded handoff to NVIDIA official Nano3 SFT CLI | **Not an end-to-end launcher** |
| `configs/lora_sft.yaml` | Illustrative settings with TODO placeholders | **Not runnable** |
| `requirements.txt` | CPU and NVIDIA hosted-teacher API dependencies | Not a full GPU training environment |

## Default 30-company universe

```text
AAPL NVDA MSFT AMZN GOOGL META TSLA JPM BAC WFC
GS MS V MA PYPL NFLX DIS WMT COST HD
UNH JNJ PFE ABBV XOM CVX BA CAT AMD INTC
```

## Fast path on a NEW Lambda Ubuntu VM

**Before the replacement:** preserve the old labeled JSONL files via the backup/upload instructions in [LAMBDA.md](LAMBDA.md).

```bash
# Fresh replacement Lambda instance, after confirming two supported GPUs:
cd "$HOME"
git clone --branch Nemotron --single-branch https://github.com/haranadhar/FinscaleNvidiaPlatform.git
cd "$HOME/FinscaleNvidiaPlatform"
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m py_compile scripts/*.py
nvidia-smi -L

# Once you copy the backup archive into /home/ubuntu:
tar -C "$PWD" -xzf "$HOME/finscale-edgar-sft-backup.tar.gz"
python scripts/validate_edgar.py --strict
wc -l data/processed/train.jsonl data/processed/val.jsonl
```

If you do **not** have the archive, see [LAMBDA.md §3](LAMBDA.md#3-restore-the-archive-onto-the-new-instance-recommended-avoids-api-calls-and-regeneration) for fresh SEC/teacher API instructions.

## Official NVIDIA Nano3 30B LoRA: setup vs. an actual training run

```bash
# Run this OUTSIDE the activated FinScale virtual environment
deactivate 2>/dev/null || true
unset VIRTUAL_ENV
cd "$HOME"
git clone https://github.com/NVIDIA-NeMo/Nemotron.git "$HOME/Nemotron"

# Install uv first if unavailable, following LAMBDA.md
cd "$HOME/Nemotron"
uv sync
uv run nemotron nano3 data prep sft --help
uv run nemotron nano3 sft --help
```

**STOP:** the current FinScale repo does **not** yet connect your JSONL into an official packed-Parquet SFT artifact, import a compatible Nano3 base checkpoint, or define a verified Lambda executor/`env.toml` profile and LoRA configuration. Merely buying a 2 × 80GB VM and setting `RUN_NEMOTRON_SFT=1` **will not** train the model. The guarded wrapper is intentionally non-runnable until this integration is completed. Refer to the official NVIDIA [Nano3 training documentation](https://github.com/NVIDIA-NeMo/Nemotron/tree/main/docs/nemotron/nano3), [data prep](https://github.com/NVIDIA-NeMo/Nemotron/blob/main/docs/nemotron/data-prep.md), and [NeMo-Run execution](https://github.com/NVIDIA-NeMo/Nemotron/blob/main/docs/nemo_runspec/nemo-run.md).

## Data quality and research considerations

- The 184 API-labeled examples are a **small** supervised set and need answer-level spot checks.
- The current generator splits examples by **row**, not held-out company or held-out filing; compute a separate held-out benchmark before claiming generalization.
- Never commit SEC files, W&B/Hugging Face/NVIDIA keys, local checkpoints, output archives, or large artifacts into source control.
- Original FinScale Qwen pipeline remains on `main`; `Nemotron` is the adapted Nemotron-focused branch.
- Data preparation originated from [Manish's FinScale branch](https://github.com/mkchoithani/finscale/tree/edgar-nemotron-runpod).

**Reference model:** https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16.

## Portable workflow: Lambda A100 / RunPod RTX PRO Blackwell / DGX Spark GB10

The **portable artifacts** are FinScale Python code, SEC JSONL, labeled chat JSONL, evaluation definitions, and appropriately converted model adapters. The **non-portable parts** include compiled CUDA extensions, GPU-specific kernels, NCCL topology, image architecture and some training recipe settings.

```bash
# In the FinScale repo on ANY platform, after restoring the archive:
python -m py_compile scripts/*.py
python scripts/validate_edgar.py --strict
wc -l data/processed/train.jsonl data/processed/val.jsonl
# On a new instance, do NOT regenerate the labels unless backup is unavailable.
```

For the hackathon, measure separately: (1) model/adapter loading and inference on GB10, (2) single-node GPU utilization, (3) distributed communication across two GB10 nodes if supplied, and (4) the cloud training baseline. **Do not report cloud A100 throughput as DGX Spark performance.** Use model checkpoint and tokenizer compatibility checks before attempting adapter transfer.
