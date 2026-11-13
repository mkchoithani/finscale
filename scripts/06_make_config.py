#!/usr/bin/env python3
"""Generate a NeMo AutoModel LoRA config for Nemotron 3 Nano 30B-A3B on 1 or N GB10 nodes.

Parallel modes (one GPU per GB10 node, world size = nodes):
  single     1 node. Baseline.
  replicate  N nodes, full model copy per node (HSDP, shard=1). Only LoRA gradients cross the
             network -> best throughput scaling. START HERE for the 2-node run.
  shard      N nodes, weights sharded across nodes (FSDP). Less memory per node -> longer
             sequences / bigger micro-batches ("capacity" story). Much more network traffic.
  ep         N nodes, MoE experts split across nodes (expert parallel). Capacity + MoE story.
  ddp        N nodes, torch DDP strategy. Fallback if 'replicate' fails.

Fair scaling rule: global batch size (--gbs) is identical in every mode; only the
per-node gradient-accumulation changes. Compare tokens/s at the same --gbs and --seq-length.

    python scripts/06_make_config.py --mode single    --run single-1n
    python scripts/06_make_config.py --mode replicate --nodes 2 --run replicate-2n
"""
import argparse
import copy
import importlib.util
import os
import sys
from pathlib import Path

import yaml

RECIPE_REL = "examples/llm_finetune/nemotron/nemotron_nano_v3_singlegpu_lora.yaml"

# Fallback if the installed AutoModel examples folder cannot be found. Mirrors the official
# single-GPU recipe (NVIDIA-NeMo/Automodel main, Nov 2026) minus the HellaSwag dataset.
TEMPLATE = {
    "recipe": "TrainFinetuneRecipeForNextTokenPrediction",
    "step_scheduler": {"global_batch_size": 8, "local_batch_size": 1, "ckpt_every_steps": 100000,
                       "val_every_steps": 100000, "max_steps": 20},
    "dist_env": {"backend": "nccl", "timeout_minutes": 5},
    "rng": {"_target_": "nemo_automodel.components.training.rng.StatefulRNG", "seed": 1111, "ranked": True},
    "model": {"_target_": "nemo_automodel.NeMoAutoModelForCausalLM.from_pretrained",
              "pretrained_model_name_or_path": "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
              "backend": {"_target_": "nemo_automodel.components.models.common.BackendConfig",
                          "rms_norm": "torch_fp32", "linear": "torch", "dispatcher": "torch"}},
    "compile": {"enabled": False},
    "checkpoint": {"enabled": False},
    "peft": {"_target_": "nemo_automodel.components._peft.lora.PeftConfig", "exclude_modules": ["*.out_proj"],
             "dim": 8, "alpha": 16, "use_triton": True, "use_memory_efficient_lora": True},
    "distributed": {"strategy": "fsdp2", "dp_size": 1, "tp_size": 1, "cp_size": 1, "ep_size": 1,
                    "sequence_parallel": False, "activation_checkpointing": True},
    "loss_fn": {"_target_": "nemo_automodel.components.loss.masked_ce.MaskedCrossEntropy"},
    "packed_sequence": {"packed_sequence_size": 0},
    "dataloader": {"_target_": "torchdata.stateful_dataloader.StatefulDataLoader",
                   "collate_fn": "nemo_automodel.components.datasets.utils.default_collater", "shuffle": True},
    "optimizer": {"_target_": "torch.optim.Adam", "betas": [0.9, 0.999], "eps": 1e-8, "lr": 1e-5,
                  "weight_decay": 0},
}


def find_recipe():
    roots = [os.environ.get("AUTOMODEL_HOME"), "/opt/Automodel", "/workspace/Automodel"]
    spec = importlib.util.find_spec("nemo_automodel")
    if spec and spec.origin:
        roots.append(str(Path(spec.origin).resolve().parents[1]))  # editable install: repo root
    for root in filter(None, roots):
        p = Path(root) / RECIPE_REL
        if p.is_file():
            return p
    return None


def chat_ds(path, seq_length, thinking):
    # Paths stay RELATIVE to the repo root (launch_train.sh cd's there), so a config generated on one
    # machine works on every node as long as each node runs from its own copy / the shared repo.
    return {"_target_": "nemo_automodel.components.datasets.llm.chat_dataset.ChatDatasetConfig",
            "path_or_dataset_id": str(path), "seq_length": seq_length,
            "truncation": "longest_first", "chat_template_kwargs": {"enable_thinking": thinking}}


def distributed_for(mode, world, base):
    d = copy.deepcopy(base)
    d.update({"tp_size": 1, "cp_size": 1, "ep_size": 1, "sequence_parallel": False,
              "activation_checkpointing": True})
    d.pop("dp_replicate_size", None)
    if mode == "single":
        d.update(strategy="fsdp2", dp_size=1)
    elif mode == "replicate":
        d.update(strategy="fsdp2", dp_size=world, dp_replicate_size=world)
    elif mode == "shard":
        d.update(strategy="fsdp2", dp_size=world)
    elif mode == "ep":
        d.update(strategy="fsdp2", dp_size=world, ep_size=world)
    elif mode == "ddp":
        d = {"strategy": "ddp", "dp_size": world, "tp_size": 1, "cp_size": 1, "ep_size": 1,
             "activation_checkpointing": True}
    return d


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["single", "replicate", "shard", "ep", "ddp"], required=True)
    ap.add_argument("--nodes", type=int, default=None, help="default: 1 for single, else 2")
    ap.add_argument("--gpus-per-node", type=int, default=1)
    ap.add_argument("--run", required=True, help="run name, e.g. replicate-2n-r1")
    ap.add_argument("--model", default=os.environ.get("MODEL_DIR", "/shared/models/nemotron-3-nano-30b"))
    ap.add_argument("--train", default="data/sft/train.jsonl")
    ap.add_argument("--val", default="data/sft/val.jsonl", help="'' to disable validation")
    ap.add_argument("--seq-length", type=int, default=2048)
    ap.add_argument("--gbs", type=int, default=16, help="global batch size (keep identical across runs)")
    ap.add_argument("--mbs", type=int, default=1, help="micro batch per GPU")
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lora-dim", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--val-every", type=int, default=0, help="0 = only at the end")
    ap.add_argument("--no-save", action="store_true", help="do not save the adapter (pure benchmark runs)")
    ap.add_argument("--thinking", action="store_true", help="render chat template with reasoning on")
    ap.add_argument("--timeout-min", type=int, default=30, help="NCCL timeout; model load on 2 nodes is slow")
    ap.add_argument("--seed", type=int, default=1111)
    ap.add_argument("--results-dir", default="results/runs")
    ap.add_argument("--base-recipe", default=None)
    a = ap.parse_args()

    nodes = a.nodes or (1 if a.mode == "single" else 2)
    if a.mode == "single" and nodes != 1:
        sys.exit("--mode single means 1 node")
    if a.mode != "single" and nodes < 2:
        sys.exit(f"--mode {a.mode} needs --nodes >= 2")
    world = nodes * a.gpus_per_node
    dp = 1 if a.mode == "single" else world
    if a.gbs % (a.mbs * dp):
        sys.exit(f"--gbs {a.gbs} must be divisible by mbs*dp = {a.mbs * dp}")
    for p in filter(None, [a.train, a.val]):
        if not Path(p).is_file():
            sys.exit(f"Missing {p}. Run scripts/04_build_sft.py first.")
    if not Path(a.model, "config.json").is_file():
        print(f"WARNING: {a.model}/config.json not found here. OK only if the path exists on the GPU nodes.",
              file=sys.stderr)

    recipe = Path(a.base_recipe) if a.base_recipe else find_recipe()
    if recipe:
        cfg = yaml.safe_load(recipe.read_text())
        print("Base recipe:", recipe)
    else:
        cfg = copy.deepcopy(TEMPLATE)
        print("Base recipe: built-in template (installed AutoModel examples not found)")

    run_dir = Path(a.results_dir) / a.run
    cfg["model"]["pretrained_model_name_or_path"] = a.model  # absolute path on the nodes (shared storage)
    cfg["dataset"] = chat_ds(a.train, a.seq_length, a.thinking)
    if a.val:
        cfg["validation_dataset"] = chat_ds(a.val, a.seq_length, a.thinking)
        cfg["validation_dataloader"] = {"_target_": "torchdata.stateful_dataloader.StatefulDataLoader",
                                        "collate_fn": "nemo_automodel.components.datasets.utils.default_collater"}
    else:
        cfg.pop("validation_dataset", None)
        cfg.pop("validation_dataloader", None)
    ss = cfg["step_scheduler"]
    ss.update(global_batch_size=a.gbs, local_batch_size=a.mbs, max_steps=a.steps,
              val_every_steps=a.val_every or a.steps, ckpt_every_steps=a.steps)
    cfg["optimizer"]["lr"] = a.lr
    cfg["peft"].update(dim=a.lora_dim, alpha=a.lora_alpha)
    cfg["rng"]["seed"] = a.seed
    cfg["dist_env"]["timeout_minutes"] = a.timeout_min
    cfg["distributed"] = distributed_for(a.mode, world, cfg.get("distributed", TEMPLATE["distributed"]))
    # checkpoint_dir is ALSO where AutoModel writes training.jsonl / validation.jsonl.
    cfg["checkpoint"] = {"enabled": not a.no_save, "checkpoint_dir": str(run_dir / "ckpt"),
                         "model_save_format": "safetensors", "save_consolidated": False}
    cfg.pop("ci", None)
    cfg["finscale"] = {"run": a.run, "mode": a.mode, "nodes": nodes, "gpus_per_node": a.gpus_per_node,
                       "world_size": world, "grad_accum_per_rank": a.gbs // (a.mbs * dp),
                       "seq_length": a.seq_length, "gbs": a.gbs}

    out = Path("configs") / f"{a.run}.yaml"
    out.parent.mkdir(parents=True, exist_ok=True)
    meta = cfg.pop("finscale")
    out.write_text("# FinScale run metadata (read by scripts/launch_train.sh and plot_results.py)\n"
                   + "".join(f"# finscale.{k}: {v}\n" for k, v in meta.items())
                   + yaml.safe_dump(cfg, sort_keys=False))
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "run_meta.yaml").write_text(yaml.safe_dump(meta, sort_keys=False))
    print(f"Wrote {out}  |  mode={a.mode} nodes={nodes} world={world} gbs={a.gbs} "
          f"grad_accum/rank={meta['grad_accum_per_rank']} seq={a.seq_length} steps={a.steps} lr={a.lr}")
    print(f"Distributed: {cfg['distributed']}")
    if nodes == 1:
        print(f"Launch:  bash scripts/launch_train.sh {out}")
    else:
        print(f"Launch (every node):  NNODES={nodes} NODE_RANK=<0..{nodes - 1}> HEAD_ADDR=<node0 IP> "
              f"PORT=<allowed port> bash scripts/launch_train.sh {out}")


if __name__ == "__main__":
    main()
