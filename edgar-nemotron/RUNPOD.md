# RunPod guide (starter; adapt to your account and current images)

This is a resource planning/checklist, not a guarantee that a particular GPU/image pairing is offered. Current RunPod SKUs, rates, templates and networking UI were not verified. Stop billing by stopping or terminating the pod after exporting results.

1. **Create pod.** Pick one 80 GB class GPU (A100 80GB or H100 80GB) as the cautious starting point. For the HF 8B PEFT fallback, VRAM depends on precision, activations/context and training config; test first. The official Nano 3 full SFT recipe docs call for H100-class Slurm GPU nodes and its examples use multi-GPU topology; one 80GB card is not confirmed adequate for that official recipe. Repository LoRA cookbook uses 8×H100 80GiB and documents 600GB checkpoint disk.
2. **Storage.** Example starter allocation: 100GB container disk for image/software, plus 200GB+ persistent network volume for model cache, raw SEC text, processed data, and outputs. Increase if converting/storing full checkpoints; official cookbook example needs 600GB for checkpoints. These are planning suggestions, not vendor minimums. Mount persistent network volume at `/workspace` if provider UI supports it; confirm the mount location before launch.
3. **Image/template.** Prefer a PyTorch image with CUDA and matching driver, or the Nemotron training image from the inspected Nano doc: `nvcr.io/nvidia/nemo:25.11.nemotron_3_nano`. Image-to-pod compatibility and access are not verified; alternatively follow repo version-specific docs. Do not install an arbitrary CUDA wheel over a prebuilt NVIDIA stack.
4. **Ports/access.** Enable provider SSH access. Optionally expose Jupyter port 8888 only behind authenticated/tunneled access. Do not run the cookbook's tokenless Jupyter command on a publicly exposed interface; use SSH tunnel or a token. RunPod-specific port UI settings were not verified.
5. **Persistent workspace and credentials.** Ensure network volume is mounted at `/workspace`. Set `HF_TOKEN` only if required by HF gated/private access and keep it out of shell history/repos; export actual `SEC_USER_AGENT` contact as described in README. Do not put credentials in command arguments or commit `.env`.
6. **Clone and install.**
   ```bash
   cd /workspace
   git clone https://github.com/<YOUR_GITHUB_USER>/edgar-nemotron.git
   cd edgar-nemotron
   python -m venv .venv && source .venv/bin/activate
   pip install -r requirements.txt
   python scripts/smoke_test_cuda.py
   ```
   For official recipe use separate clone and tested setup:
   ```bash
   git clone https://github.com/NVIDIA-NeMo/Nemotron.git /workspace/Nemotron
   cd /workspace/Nemotron && uv sync
   ```
7. **Prepare EDGAR data.**
   ```bash
   cd /workspace/edgar-nemotron
   export SEC_USER_AGENT='Your Name your-real-contact@example.org'
   python scripts/download_edgar.py --tickers AAPL MSFT JPM --limit 1
   python scripts/build_sft_dataset.py
   ```
8. **Train.** Quick fallback (verify model compatibility/storage first):
   ```bash
   torchrun --nproc_per_node=1 scripts/train_hf_lora.py --model nvidia/Llama-3.1-Nemotron-Nano-8B-v1
   ```
   Official recipe is not connected to this raw JSONL as-is. First create/register expected packed Parquet artifact, configure W&B/cluster and verify `configs/lora_sft.yaml` TODOs; then use documented repo command per README. Running full repository recipes may need Slurm/multi-GPU resources and a different container.
9. **Copy results out.** Write final adapter/results under `/workspace/edgar-nemotron/outputs` (persistent volume), then transfer to your local machine using SSH/SCP/rsync using your pod's actual connection string, or push artifacts to your approved object store. Confirm output files exist on persistent storage before stopping.
10. **Stop or terminate.** Stop the pod when idle to stop GPU runtime charges if persistent storage remains available; terminate when finished if no longer needed. Confirm billing and volume-retention rules in provider UI. Do not assume container-local files survive termination.

## Publish to a new GitHub repository

Create an empty GitHub repo first (do not initialize it with files if you want the following commands unchanged), then:

```bash
cd /workspace/edgar-nemotron
git init
git add README.md RUNPOD.md requirements.txt .gitignore scripts configs data/raw/.gitkeep data/processed/.gitkeep outputs/.gitkeep
git commit -m "Add EDGAR Nemotron SFT starter"
git branch -M main
git remote add origin https://github.com/<YOUR_GITHUB_USER>/edgar-nemotron.git
git push -u origin main
```

Review `git status` and staged filenames for private filings/tokens/checkpoints before push. GitHub username/URL placeholder must be replaced. The official Nemotron checkout may be added as a submodule later if pinning is desired; currently the wrapper clones/uses a separate install step and does not vendor upstream source.
