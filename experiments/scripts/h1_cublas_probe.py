# Which cuBLAS kernel does F.linear(x[M,K], W[N,K]) dispatch for the Llama-70B layer-0 GEMM shapes as M (tokens) varies?
# Records the kernel names via torch.profiler (same names as in the NVBit traces) for each (M, shape).
import torch, json, sys
from torch.profiler import profile, ProfilerActivity
shapes = {"qkv": (8192, 10240), "o": (8192, 8192), "gate_up": (8192, 57344), "down": (28672, 8192)}  # (K, N_out)
Ms = [1, 4, 16, 64, 256, 1024]
torch.manual_seed(0)
out = {}
for name, (K, N) in shapes.items():
    W = torch.randn(N, K, device="cuda", dtype=torch.bfloat16)
    for M in Ms:
        x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
        for _ in range(2): torch.nn.functional.linear(x, W)   # warm-up (heuristic cache, lazy module load)
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            torch.nn.functional.linear(x, W); torch.cuda.synchronize()
        ks = [e for e in prof.events() if e.device_type.name == "CUDA"]
        names = [f"{e.name[:90]} grid={tuple(e.grid) if hasattr(e,'grid') else '?'}" for e in ks]
        out[f"{name}:M{M}"] = [e.name for e in ks]
        print(f"{name:8s} M={M:5d} K={K} N={N}: " + " | ".join(e.name[:70] for e in ks), flush=True)
json.dump(out, open("/work/probe_kernels.json", "w"), indent=1)
print("torch", torch.__version__, "cublas", torch.backends.cuda.preferred_blas_library() if hasattr(torch.backends.cuda, "preferred_blas_library") else "?", "sm", torch.cuda.get_device_capability(), torch.cuda.get_device_name())
