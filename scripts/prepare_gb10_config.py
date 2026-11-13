#!/usr/bin/env python3
"""Generate a FinScale LoRA config from the installed official Nano3 single-GPU recipe.

Works on GB10 (/workspace/Automodel editable install) and on RunPod using the
NGC nemo-automodel container (/opt/Automodel). Pass --recipe to override.
"""
import argparse
import os
from pathlib import Path

import yaml

RECIPE_REL = "examples/llm_finetune/nemotron/nemotron_nano_v3_singlegpu_lora.yaml"


def find_recipe():
    roots = [os.environ.get("AUTOMODEL_HOME"), "/opt/Automodel", "/workspace/Automodel"]
    for root in filter(None, roots):
        cand = Path(root) / RECIPE_REL
        if cand.is_file():
            return cand
    return None


def chat_dataset(path, seq_length, enable_thinking):
    return {
        "_target_": "nemo_automodel.components.datasets.llm.chat_dataset.ChatDatasetConfig",
        "path_or_dataset_id": str(Path(path).resolve()),
        "seq_length": seq_length,
        "truncation": "longest_first",
        "chat_template_kwargs": {"enable_thinking": enable_thinking},
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--recipe", default=None, help="Official recipe path (auto-detected if omitted)")
    ap.add_argument("--model", default="/workspace/models/nemotron-30b")
    ap.add_argument("--data", default="data/processed/train_fit.jsonl")
    ap.add_argument("--val-data", default="data/processed/val_fit.jsonl",
                    help="Validation JSONL; pass '' to disable")
    ap.add_argument("--output", default="configs/nemotron_finscale_lora.yaml")
    ap.add_argument("--seq-length", type=int, default=4096)
    ap.add_argument("--steps", type=int, default=1)
    ap.add_argument("--global-batch-size", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-4,
                    help="LoRA LR. Upstream 1e-5 barely moves a rank-8 adapter in ~20 steps")
    ap.add_argument("--val-every", type=int, default=0, help="0 = only at end (max_steps)")
    ap.add_argument("--save-checkpoint", action="store_true", help="Save the LoRA adapter")
    ap.add_argument("--checkpoint-dir", default="/workspace/checkpoints/finscale-lora")
    ap.add_argument("--enable-thinking", action="store_true")
    args = ap.parse_args()
    if args.steps < 1 or args.seq_length < 128 or args.global_batch_size < 1:
        ap.error("steps>=1, seq-length>=128, global-batch-size>=1 required")

    recipe = Path(args.recipe) if args.recipe else find_recipe()
    if recipe is None or not recipe.is_file():
        raise SystemExit(f"Recipe not found. Set AUTOMODEL_HOME or pass --recipe ({RECIPE_REL})")
    for p in filter(None, [args.data, args.val_data]):
        if not Path(p).is_file():
            raise SystemExit(f"Missing data file: {p} (run scripts/fit_dataset_to_seq_len.py)")
    if not (Path(args.model) / "config.json").is_file():
        raise SystemExit(f"Model dir has no config.json: {args.model}")

    cfg = yaml.safe_load(recipe.read_text())
    expected = "nemo_automodel.components.datasets.llm.hellaswag.HellaSwag"
    if cfg.get("dataset", {}).get("_target_") != expected:
        raise SystemExit("Official recipe changed upstream: inspect its dataset/peft schema manually")

    cfg["model"]["pretrained_model_name_or_path"] = str(Path(args.model).resolve())
    cfg["dataset"] = chat_dataset(args.data, args.seq_length, args.enable_thinking)
    if args.val_data:
        cfg["validation_dataset"] = chat_dataset(args.val_data, args.seq_length, args.enable_thinking)
    ss = cfg["step_scheduler"]
    ss["max_steps"] = args.steps
    ss["global_batch_size"] = args.global_batch_size
    ss["local_batch_size"] = 1
    ss["val_every_steps"] = args.val_every or args.steps
    ss["ckpt_every_steps"] = args.steps  # save once at the end when enabled
    cfg["optimizer"]["lr"] = args.lr
    if args.save_checkpoint:
        cfg["checkpoint"] = {
            "enabled": True,
            "checkpoint_dir": args.checkpoint_dir,
            "model_save_format": "safetensors",
            "save_consolidated": False,  # adapter only; the 59 GiB base stays frozen
        }
    cfg.pop("ci", None)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(yaml.safe_dump(cfg, sort_keys=False))
    print("Recipe:    ", recipe)
    print("Generated: ", out)
    print("seq_length:", args.seq_length, "steps:", args.steps, "GBS:", args.global_batch_size, "lr:", args.lr)
    print("Checkpoint:", cfg.get("checkpoint", {}).get("enabled"))
    print("NOTE: run scripts/check_gb10_supervision.py on the same data/seq_length first")


if __name__ == "__main__":
    main()
