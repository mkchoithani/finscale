import sys, torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
base, adapter, out = sys.argv[1], sys.argv[2], sys.argv[3]
m = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.bfloat16)
m = PeftModel.from_pretrained(m, adapter).merge_and_unload()
m.save_pretrained(out); AutoTokenizer.from_pretrained(base).save_pretrained(out)
print("merged ->", out)
