# FinScale on Lambda Labs GPU Cloud

This guide runs FinScale on Lambda Labs GPU Cloud as a fallback to the GB10/Fuzzball workflow. Lambda's instance catalog, regions, networking, and launch interfaces can change; confirm current availability and documentation in your Lambda account before provisioning.

## Prerequisites

- A Lambda Labs account with GPU capacity available in the region you plan to use.
- An SSH public key added to your Lambda account; keep the matching private key secure and use it when connecting.
- A checkout of this repository on branch `GB10withfullball`.
- For a single-node baseline, choose one GPU instance with **1x A100 or 1x H100** (whichever is available and fits your budget/workload).
- For multi-node testing, choose **two instances in the same region**. Lambda 1-Click Clusters and their private networking may simplify node-to-node communication, but are optional; verify what networking features are available for your selected instances.
- A Hugging Face access token (`HF_TOKEN`) if required by the model/repository access policy. Do not commit or paste the token into source control.

## Provision and connect

1. Launch the instance(s) through the Lambda dashboard, or use Lambda Cloud's current API/CLI workflow. API and CLI syntax can change: consult Lambda's current documentation and **verify the exact command/flags** rather than relying on copied examples.
2. Select the required GPU type, region, OS/image, SSH key, and (for multi-node) a networking arrangement that permits node-to-node traffic. Record the private IP of node 0 for multi-node runs.
3. SSH into each machine using the connection details shown by Lambda. For example, use `ssh -i <private-key> <user>@<public-ip>` with the username and address supplied for your image.
4. Confirm that the GPU and driver are visible:

   ```bash
   nvidia-smi
   ```

   If this fails, resolve the instance/driver issue before installing or running the workload.

## Set up the code and environment

On each instance, clone the repository and select the requested branch:

```bash
git clone <repository-url> finscale
cd finscale
git checkout GB10withfullball
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Check that the installed PyTorch build can see CUDA (`python -c 'import torch; print(torch.cuda.is_available(), torch.cuda.device_count())'`). If it cannot, install a PyTorch build compatible with the instance's CUDA/driver using the current official PyTorch instructions, then re-check. Authenticate as needed without putting the token in shell history or source control; for the current shell, set `HF_TOKEN` securely, e.g. `export HF_TOKEN=<your-token>`.

Generate the training and evaluation data from the repository root:

```bash
python gen_data.py
```

This writes `data/train.jsonl` and `data/eval.jsonl`. In multi-node runs, ensure both nodes have the same checkout, generated data, Python environment, and model access.

## Single-node training

The training script accepts `--model`, `--data`, `--out`, `--epochs`, `--bs`, `--lr`, and `--maxlen`. For example:

```bash
python train_gb10.py --epochs 1 --bs 4 --out out1/adapter
```

The script uses the available GPU; for an explicit one-process-per-GPU launch, use `torchrun` and set `N` to the number of GPUs on the instance:

```bash
torchrun --standalone --nproc_per_node=N train_gb10.py --epochs 1 --bs 4 --out out1/adapter
```

`train_gb10.py` writes the adapter under the requested output directory and a `train_summary.json` beside that directory (for this example, `out1/train_summary.json`). The repository's `scripts/launch_single.sh` wrapper currently runs one process (`--nproc_per_node=1`); use the direct command above when using more than one GPU.

## Two-node training

Use two same-region instances with working private connectivity. Generate data on each node, and use the private IP of node 0 as `<private-IP-node0>`. Open/allow the required inter-node traffic in the instance/network firewall or security configuration, including the rendezvous port `29500` and the communication ports required by NCCL. Restrict access to the participating nodes/network; do not expose these ports publicly without a need. Lambda 1-Click Clusters/private networking can be used if available, but verify their connectivity and firewall behavior.

On **both** nodes, from the repository root, set the node-specific rank (`0` on node 0, `1` on node 1) and run the same command. Set `N` to the number of GPUs **per node**:

```bash
export NCCL_DEBUG=INFO
# Inspect `ip a` on each node to find the private-network interface.
# If NCCL does not select it automatically, set this to the verified interface name:
export NCCL_SOCKET_IFNAME=<interface-from-ip-a>

NODE_RANK=<0-or-1> torchrun --nnodes=2 --node_rank=<0-or-1> --nproc_per_node=N \
  --master_addr=<private-IP-node0> --master_port=29500 \
  train_gb10.py --epochs 1 --bs 4 --out out2/adapter
```

Replace **both** occurrences of `<0-or-1>` with the node's rank on that machine. If NCCL works with automatic interface selection, omit `NCCL_SOCKET_IFNAME`; an incorrect interface name can prevent startup. Keep `--out` identical on both nodes. The script saves the adapter and `out2/train_summary.json` on rank 0 only; copy results from node 0. Run both node processes concurrently. Do not assume the shell wrapper `scripts/launch_multinode.sh` will configure Lambda networking or rank values for you; its variables must be set manually and it defaults to one process per node.

Compare the single-node and two-node summaries from the project root (adjust paths to match the runs):

```bash
python scripts/compare_scaling.py out1/train_summary.json out2/train_summary.json
```

This script takes exactly two positional arguments: the one-node summary path, then the two-node summary path.

## Merge, serve, and evaluate

To merge a trained adapter into the base model, pass the base model ID, adapter directory, and destination directory as positional arguments:

```bash
python merge_adapter.py Qwen/Qwen2.5-0.5B-Instruct out2/adapter out2/merged
```

Install vLLM in an appropriate serving environment (check the current vLLM/PyTorch/CUDA compatibility for the instance):

```bash
python -m pip install vllm
```

Lambda GPU instances in this workflow are **x86_64**, not arm64. The repository's Dockerfile/build example targets arm64 for the GB10/Fuzzball path; rebuild/adapt the image for x86_64 and the Lambda GPU runtime if using Docker, or run vLLM natively as below. Do not assume the existing arm64 image runs on Lambda.

Serve the merged model (or another model directory/model ID) using vLLM's OpenAI-compatible API:

```bash
python -m vllm.entrypoints.openai.api_server \
  --model out2/merged --served-model-name finscale --host 0.0.0.0 --port 8000
```

Keep port 8000 private or restrict it in the firewall; use SSH tunneling or a suitably protected network for remote access. From the serving instance, verify the endpoint:

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"finscale","messages":[{"role":"user","content":"Classify: cash reserves declined 12% and short-term debt matures soon."}],"max_tokens":80,"temperature":0}'
```

Then, from the FinScale repository root with `data/eval.jsonl` present, run the project's evaluator. It takes the base URL (without `/chat/completions`) and the served model name as positional arguments:

```bash
python evaluate_endpoint.py http://127.0.0.1:8000/v1 finscale
```

## Copy results back and stop billing

Copy desired outputs (for example, summaries and adapters) from the instance to a trusted local machine using `scp` or `rsync`; use the host/user details supplied by Lambda. Example:

```bash
scp -i <private-key> -r <user>@<public-ip>:<path-to-finscale>/out2 ./
```

Adapters and model artifacts may be large; verify the transfer before deleting anything. **IMPORTANT: terminate/delete the Lambda instance(s) in the Lambda dashboard or via the currently documented cloud workflow when finished. Stopping work in the shell is not enough—running instances continue to incur charges.** Confirm all instances and any separately billed resources have been removed.

## Troubleshooting

| Symptom | Checks / actions |
|---|---|
| `nvidia-smi` fails or PyTorch reports no CUDA | Check the selected GPU instance, driver status, and CUDA-compatible PyTorch installation; verify with `torch.cuda.is_available()` and `torch.cuda.device_count()`. |
| Hugging Face download/authentication error | Check `HF_TOKEN`, network access, and model access permissions; never place the token in the repository. |
| Multi-node rendezvous hangs or times out | Confirm both processes use `--nnodes=2`, complementary ranks 0/1, identical master address/port, reachable node-0 private IP, and that node-to-node traffic is allowed. |
| NCCL selects wrong interface or cannot connect | Inspect `ip a` on both nodes; set `NCCL_SOCKET_IFNAME` to the correct shared private-network interface and keep `NCCL_DEBUG=INFO` for diagnostics. |
| Output/summary not found on second node | Rank 0 writes the adapter and summary. Check node 0 and ensure both ranks used the same `--out` path. |
| vLLM installation/runtime failure | Verify x86_64 architecture and current vLLM, PyTorch, CUDA, and driver compatibility; rebuild any Docker image for the Lambda architecture/runtime or run natively. |
| Endpoint unreachable | Check that vLLM is running, the URL/model name is correct, and port 8000 is reachable only through the intended network/firewall path. |

## Lambda Labs vs. GB10/Fuzzball

| Area | Lambda Labs GPU Cloud | GB10/Fuzzball workflow in this repository |
|---|---|---|
| Hardware | Cloud GPU instances such as A100/H100; availability varies by region. | GB10 systems; the project compares one and two GB10 nodes. |
| Provisioning | Launch/terminate instances through Lambda's dashboard or current cloud API/CLI; instances incur charges while active. | Uses the repository's Fuzzball job/config files (`fuzzball/*.fz`) and platform-specific submission. |
| Multi-node setup | User selects same-region nodes, ranks, private master IP, network interface, and required firewall access; clusters/private networking may be optional features. | Fuzzball manages job placement/network configuration according to its environment; repository configs still contain TODOs to verify. |
| Container/architecture | Lambda instances in this guide are x86_64; use a compatible native install or x86_64 GPU image. | Dockerfile/build notes target `linux/arm64` for the GB10/Fuzzball path. |
| Training entry point | Same `train_gb10.py` arguments; launch directly with `torchrun` and configure the cloud networking yourself. | `scripts/launch_single.sh` and `scripts/launch_multinode.sh` provide wrappers, with Fuzzball environment details requiring confirmation. |

