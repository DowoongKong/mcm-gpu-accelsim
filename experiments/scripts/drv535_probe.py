# Probe: does a CUDA 12.6 PyTorch stack (cuBLAS + Triton JIT) run on host driver 535 (CUDA 12.2)?
import torch, triton, triton.language as tl
print("torch", torch.__version__, "cuda", torch.version.cuda, "triton", triton.__version__)
print("device", torch.cuda.get_device_name(0), "cap", torch.cuda.get_device_capability(0))
a = torch.randn(1024, 4096, device="cuda", dtype=torch.bfloat16)
b = torch.randn(4096, 1024, device="cuda", dtype=torch.bfloat16)
c = a @ b
torch.cuda.synchronize()
print("bf16 gemm ok", tuple(c.shape))

@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    tl.store(out_ptr + offs, tl.load(x_ptr + offs, mask=mask) + tl.load(y_ptr + offs, mask=mask), mask=mask)

x = torch.rand(100000, device="cuda"); y = torch.rand(100000, device="cuda"); o = torch.empty_like(x)
add_kernel[(triton.cdiv(x.numel(), 1024),)](x, y, o, x.numel(), BLOCK=1024)
torch.cuda.synchronize()
print("triton jit kernel ok, max err", (o - (x + y)).abs().max().item())
