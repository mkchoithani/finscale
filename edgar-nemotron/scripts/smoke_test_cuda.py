#!/usr/bin/env python3
"""Print PyTorch and CUDA runtime visibility."""
import json
try: import torch
except ImportError as e: raise SystemExit("PyTorch is not installed") from e
r={"torch":torch.__version__,"cuda_build":torch.version.cuda,"cuda_available":torch.cuda.is_available(),"device_count":torch.cuda.device_count()}
if r["cuda_available"]: r["devices"]=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
print(json.dumps(r,indent=2))
