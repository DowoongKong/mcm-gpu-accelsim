#!/usr/bin/env python3
# Patch 30 identity test, address-insensitive part: per kernel, md5 of the raw-trace instruction stream with the address
# fields removed (everything after mem_width), plus the kernel list without MemcpyHtoD addresses. Run-to-run VA differences
# (allocator/ASLR) must not count; runs a and d use the same pre-patch tracer twice and set the baseline.
# Usage: s25_patch30_compare.py <root dir holding a/ b/ c/ d/>
import glob, hashlib, lzma, os, re, sys
root = sys.argv[1]
def digest(d):
    out = {}
    for f in sorted(glob.glob(os.path.join(d, "traces", "kernel-*.trace.xz"))):
        # the raw trace is written in message-arrival order (nondeterministic across warps), so hash the sorted multiset
        h = hashlib.md5(); lines = []
        with lzma.open(f, "rt", errors="replace") as fh:
            for line in fh:
                if line.startswith("-") or line.startswith("#"):
                    # header: keep only shape lines; kernel id, context, shmem/local base addresses differ per process
                    if line.startswith("-grid dim") or line.startswith("-block dim") or line.startswith("-nregs") or line.startswith("-shmem ="): lines.append(line.strip())
                    continue
                t = line.split()
                if len(t) < 8: lines.append(line.strip()); continue
                dn = int(t[6]); oi = 7 + dn; sn = int(t[oi + 1]); kk = oi + 2 + sn
                lines.append(" ".join(t[:kk + 1]))
        for l in sorted(lines): h.update(l.encode())
        out[re.sub(r"-ctx_0x[0-9a-f]+", "", os.path.basename(f))] = h.hexdigest()[:12]
    kl = open(glob.glob(os.path.join(d, "traces", "kernelslist_ctx_*"))[0]).read().splitlines()
    out["kernelslist"] = hashlib.md5("\n".join(re.sub(r"-ctx_0x[0-9a-f]+", "", l.split(",")[0]) + ("," + l.split(",")[2] if l.startswith("Memcpy") else "") for l in kl).encode()).hexdigest()[:12]
    return out
D = {k: digest(os.path.join(root, k)) for k in "abcd"}
keys = sorted(D["a"])
print(f"{'kernel':36s} a(old)        b(new)        c(new+SASS)   d(old again)")
for k in keys: print(f"{k[:36]:36s} {D['a'].get(k)}  {D['b'].get(k)}  {D['c'].get(k)}  {D['d'].get(k)}")
same = lambda x, y: sorted(D[x]) == sorted(D[y]) and all(D[x][k] == D[y][k] for k in D[x])
print("address-insensitive: a==d(old twice):", same("a", "d"), " a==b:", same("a", "b"), " a==c:", same("a", "c"), " b==c:", same("b", "c"))
