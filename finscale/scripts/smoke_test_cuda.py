"""Minimal PyTorch/CUDA runtime check for the allocated GPU."""
import torch

print("torch.__version__:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available")
print("device name:", torch.cuda.get_device_name(0))
print("capability:", torch.cuda.get_device_capability(0))
left = torch.randn((32, 32), device="cuda")
right = torch.randn((32, 32), device="cuda")
result = left @ right
torch.cuda.synchronize()
print("GPU matmul:", tuple(result.shape), "finite:", bool(torch.isfinite(result).all()))
