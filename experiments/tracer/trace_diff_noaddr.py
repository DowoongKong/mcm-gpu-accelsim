#!/usr/bin/env python3
# trace_diff_noaddr.py <a.traceg[.xz]> <b.traceg[.xz]> [--max-report 20]
# Compare two accel-sim kernel traces of the SAME kernel captured by two tracer builds (e.g. NVBit 1.7.6 vs 1.8.1),
# ignoring what legitimately differs between processes: the nvbit version line, shared/local memory base addresses, and
# the memory ADDRESSES of each instruction. Everything else must match line by line: thread-block order, warp ids,
# instruction counts, PC, active mask, registers, opcode, memory width, address-list format and address count.
# Exit 0 and "IDENTICAL (addresses ignored)" when they match.
import lzma, sys

IGNORE_HDR = ("nvbit version", "shmem base_addr", "local mem base_addr")


def opener(p):
    return lzma.open(p, "rt") if p.endswith(".xz") else open(p)


def norm(line):
    if line.startswith("-"):
        k = line.lstrip("-").split("=", 1)[0].strip()
        return None if k in IGNORE_HDR else line.rstrip("\n")
    t = line.split()
    if len(t) < 5 or not t[2].isdigit():
        return line.rstrip("\n")                       # thread block / warp / insts / comment lines: compare as is
    dn = int(t[2]); oi = 3 + dn
    try:
        sn = int(t[oi + 1]); kk = oi + 2 + sn
        width = t[kk]
    except (IndexError, ValueError):
        return line.rstrip("\n")
    if width == "0" or len(t) <= kk + 1:
        return " ".join(t)
    fmt = t[kk + 1]; act = bin(int(t[1], 16)).count("1")
    # keep everything up to the address-list format; replace the addresses by the active-lane count
    return " ".join(t[:kk + 2]) + f" <{act} addrs fmt {fmt}>"


def main():
    a, b = sys.argv[1], sys.argv[2]
    mx = int(sys.argv[sys.argv.index("--max-report") + 1]) if "--max-report" in sys.argv else 20
    n = diff = 0
    with opener(a) as fa, opener(b) as fb:
        ia = (x for x in (norm(l) for l in fa) if x is not None)
        ib = (x for x in (norm(l) for l in fb) if x is not None)
        while True:
            la = next(ia, None); lb = next(ib, None)
            if la is None and lb is None: break
            n += 1
            if la != lb:
                diff += 1
                if diff <= mx: print(f"line {n}:\n  A: {la}\n  B: {lb}")
    print(f"compared {n:,} lines, {diff:,} differ -> " + ("IDENTICAL (addresses ignored)" if diff == 0 else "DIFFERENT"))
    sys.exit(0 if diff == 0 else 1)


main()
