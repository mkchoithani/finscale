#!/usr/bin/env python3
"""Small-model Hugging Face LoRA SFT fallback. Verify model+Transformers compatibility before GPU training."""
import argparse,json,time,inspect
from pathlib import Path

def main():
    ap=argparse.ArgumentParser(description=__doc__); ap.add_argument("--train",default="data/processed/train.jsonl"); ap.add_argument("--val",default="data/processed/val.jsonl"); ap.add_argument("--model",default="nvidia/Llama-3.1-Nemotron-Nano-8B-v1"); ap.add_argument("--output",default="outputs/hf-lora"); ap.add_argument("--epochs",type=float,default=1.0); ap.add_argument("--batch-size",type=int,default=1); ap.add_argument("--grad-accum",type=int,default=8); ap.add_argument("--max-length",type=int,default=2048); ap.add_argument("--lr",type=float,default=2e-4); ap.add_argument("--lora-r",type=int,default=8); ap.add_argument("--lora-alpha",type=int,default=16); ap.add_argument("--workers",type=int,default=0); a=ap.parse_args()
    try:
        import torch
        from datasets import load_dataset
        from transformers import AutoModelForCausalLM,AutoTokenizer,DataCollatorForLanguageModeling,Trainer,TrainingArguments
        from peft import LoraConfig,get_peft_model
    except ImportError as e: raise SystemExit(f"Missing HF fallback dependency: {e}") from e
    if not torch.cuda.is_available(): raise SystemExit("CUDA required for this training script; run smoke_test_cuda.py")
    data_files={"train":a.train}
    if Path(a.val).exists() and Path(a.val).stat().st_size: data_files["validation"]=a.val
    ds=load_dataset("json",data_files=data_files)
    tokenizer=AutoTokenizer.from_pretrained(a.model,token=True)
    if tokenizer.pad_token is None: tokenizer.pad_token=tokenizer.eos_token
    def render(row):
        text=tokenizer.apply_chat_template(row["messages"],tokenize=False,add_generation_prompt=False) if tokenizer.chat_template else "\n".join(f"{m['role']}: {m['content']}" for m in row["messages"])
        return {"text":text}
    ds=ds.map(render,remove_columns=ds["train"].column_names)
    def tokenize(batch): return tokenizer(batch["text"],truncation=True,max_length=a.max_length)
    ds=ds.map(tokenize,batched=True,remove_columns=["text"])
    model=AutoModelForCausalLM.from_pretrained(a.model,torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16)
    lora=LoraConfig(r=a.lora_r,lora_alpha=a.lora_alpha,lora_dropout=0.05,bias="none",task_type="CAUSAL_LM",target_modules="all-linear")
    model=get_peft_model(model,lora); model.print_trainable_parameters()
    class ThroughputTrainer(Trainer):
        def __init__(self,*args,**kwargs): super().__init__(*args,**kwargs); self._start=None
        def training_step(self,*args,**kwargs):
            if self._start is None: self._start=time.perf_counter()
            loss=super().training_step(*args,**kwargs)
            step=int(self.state.global_step)+1
            if step%10==0 and self._start:
                elapsed=max(time.perf_counter()-self._start,1e-6); print(json.dumps({"step":step,"elapsed_s":round(elapsed,2),"steps_per_s":round(step/elapsed,4)}),flush=True)
            return loss
    train_args={"output_dir":a.output,"num_train_epochs":a.epochs,"per_device_train_batch_size":a.batch_size,"per_device_eval_batch_size":1,"gradient_accumulation_steps":a.grad_accum,"learning_rate":a.lr,"logging_steps":10,"save_strategy":"epoch","bf16":torch.cuda.is_bf16_supported(),"fp16":not torch.cuda.is_bf16_supported(),"gradient_checkpointing":True,"report_to":"none","dataloader_num_workers":a.workers}
    eval_key="eval_strategy" if "eval_strategy" in inspect.signature(TrainingArguments.__init__).parameters else "evaluation_strategy"
    train_args[eval_key]="epoch" if "validation" in ds else "no"
    args=TrainingArguments(**train_args)
    trainer=ThroughputTrainer(model=model,args=args,train_dataset=ds["train"],eval_dataset=ds.get("validation"),data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer,mlm=False))
    trainer.train(); trainer.save_model(a.output); tokenizer.save_pretrained(a.output)
if __name__=="__main__": main()
