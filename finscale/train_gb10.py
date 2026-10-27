"""LoRA fine-tune. Works single-process (Mac/CPU/MPS) or under torchrun (multi-node NCCL)."""
import os, json, time, argparse, torch, torch.distributed as dist
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader, DistributedSampler

p = argparse.ArgumentParser()
p.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
p.add_argument("--data", default="data/train.jsonl")
p.add_argument("--out", default="out/adapter")
p.add_argument("--epochs", type=int, default=1)
p.add_argument("--bs", type=int, default=4)
p.add_argument("--lr", type=float, default=2e-4)
p.add_argument("--maxlen", type=int, default=256)
a = p.parse_args()

ddp = "RANK" in os.environ
rank = int(os.environ.get("RANK", 0)); world = int(os.environ.get("WORLD_SIZE", 1))
local = int(os.environ.get("LOCAL_RANK", 0))
if torch.cuda.is_available():
    dev = torch.device("cuda", local); torch.cuda.set_device(local)
elif torch.backends.mps.is_available(): dev = torch.device("mps")
else: dev = torch.device("cpu")
if ddp: dist.init_process_group("nccl" if dev.type == "cuda" else "gloo")

tok = AutoTokenizer.from_pretrained(a.model)
if tok.pad_token is None: tok.pad_token = tok.eos_token
dtype = torch.bfloat16 if dev.type == "cuda" else torch.float32
model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=dtype).to(dev)
model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM"))
if ddp:
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local] if dev.type == "cuda" else None)

rows = [json.loads(l) for l in open(a.data)]
def collate(b):
    texts = [r["prompt"] + "\n" + r["completion"] + tok.eos_token for r in b]
    enc = tok(texts, padding=True, truncation=True, max_length=a.maxlen, return_tensors="pt")
    enc["labels"] = enc["input_ids"].clone(); enc["labels"][enc["attention_mask"] == 0] = -100
    return enc
sampler = DistributedSampler(rows, num_replicas=world, rank=rank) if ddp else None
dl = DataLoader(rows, batch_size=a.bs, sampler=sampler, shuffle=sampler is None, collate_fn=collate)
opt = torch.optim.AdamW([q for q in model.parameters() if q.requires_grad], lr=a.lr)

model.train(); t_start = time.time(); tokens = 0; steps = 0
for ep in range(a.epochs):
    if sampler: sampler.set_epoch(ep)
    for batch in dl:
        t0 = time.time()
        batch = {k: v.to(dev) for k, v in batch.items()}
        loss = model(**batch).loss
        loss.backward(); opt.step(); opt.zero_grad()
        if dev.type == "cuda": torch.cuda.synchronize()
        dt = time.time() - t0; tokens += int(batch["attention_mask"].sum()) * world; steps += 1
        if rank == 0 and steps % 10 == 0:
            print(json.dumps({"step": steps, "loss": round(loss.item(), 4), "step_time_s": round(dt, 3),
                  "tokens_per_s": round(batch["attention_mask"].sum().item() * world / dt),
                  "gpu_mem_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2) if dev.type == "cuda" else 0}), flush=True)
total = time.time() - t_start
if rank == 0:
    os.makedirs(a.out, exist_ok=True)
    (model.module if ddp else model).save_pretrained(a.out); tok.save_pretrained(a.out)
    summary = {"world_size": world, "epoch_time_s": round(total, 1), "total_tokens": tokens,
               "avg_tokens_per_s": round(tokens / total), "steps": steps}
    json.dump(summary, open(os.path.join(os.path.dirname(a.out) or ".", "train_summary.json"), "w"), indent=2)
    print("SUMMARY", summary)
if ddp: dist.destroy_process_group()
