"""
Static ISA report over the cross-compiles written by `cisc662.analysis.xcompile` (task card AMD-1, step 3).

For every results/isa/<kernel_id>/triton_<variant>/<target>/xc-<tag>/<kernel>.compile.json:
  NVIDIA (sm_90a)  cuobjdump -res-usage (REG/SHARED/LOCAL/STACK; HPC SDK on PATH, else Triton's bundled one),
                   cuobjdump -sass -> <kernel>.sass + instruction histogram, ptxas -v (Triton's bundled ptxas, the
                   one that built the cubin) for spill bytes; occupancy from the sm_90 limits (65536 regs/SM in
                   256-register warp granules, 228 KiB smem/SM incl. 1 KiB/CTA reserved, 32 CTAs, 64 warps).
  AMD (gfx9)       '; Kernel info' comments of the .amdgcn + `llvm-readelf --notes` of the .hsaco (CCE clang 20
                   tools), instruction histogram of the .amdgcn, `llvm-objdump -d` of the base-config hsaco (MFMA
                   cross-check), MFMA cycles from the AMD Matrix Instruction Calculator (scratch run_mic.py), and
                   occupancy RECOMPUTED from the LLVM AMDGPU tables with Triton's dynamic LDS (metadata["shared"],
                   which the compiler's printed 'Occupancy' line does not see) and the workgroup size:
                     gfx908: waves/SIMD = min(10, 256 / max(align4(VGPR), align4(AGPR)))
                     gfx90a/gfx942: waves/SIMD = min(8, 512 / align8(align4(VGPR) + AGPR))
                     SGPR limit (gfx9): <=80 -> 10, <=88 -> 9, <=100 -> 8, else 7
                     workgroups/CU = min(floor(4 * waves/SIMD / num_warps), floor(64 KiB / LDS per WG),
                                         16 if num_warps > 1)       (a WG is resident whole or not at all)
                     occupancy = workgroups/CU * num_warps / 4 waves per SIMD
Writes cisc662/results/static_isa.csv (one row per compile; every row label=STATIC with compiler + tool
versions, PLAN.md R10) plus <kernel>.hist.json / .sass / .notes.txt / .hsaco.dis.s next to the dumps (sweep
configs: text files in the scratch mirror, like xcompile).

  nice -n 19 python -m cisc662.analysis.isa_report [--kernels rmsnorm,...] [--cpu-limit 1080]
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import re
import resource
import shutil
import subprocess
import sys
from collections import Counter

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS = os.path.join(REPO, "cisc662", "results")
ISA_DIR = os.path.join(RESULTS, "isa")
CSV_PATH = os.path.join(RESULTS, "static_isa.csv")
WORK = os.environ.get("CISC662_WORK", "/work/nvme/bffp/tmahmood1/cisc662")
SCRATCH = os.path.join(WORK, "scratch", "AMD-1")
MIC = os.path.join(WORK, "scratch", "amd-rocm-planning", "scripts", "run_mic.py")
MIC_CACHE = os.path.join(SCRATCH, "mic_cycles.json")
CB = "/opt/cray/pe/cce/20.0.0/cce-clang/aarch64/bin"

AMD_ARCH = {  # max waves/SIMD, VGPR budget per lane (unified on gfx90a+), alloc granule, MIC architecture
    "gfx908": dict(max_waves=10, budget=256, granule=4, unified=False, mic="cdna1"),
    "gfx90a": dict(max_waves=8, budget=512, granule=8, unified=True, mic="cdna2"),
    "gfx942": dict(max_waves=8, budget=512, granule=8, unified=True, mic="cdna3"),
}
AMD_LDS_PER_CU = 65536
AMD_MAX_WG_PER_CU = 16
SM90 = dict(regs_per_sm=65536, reg_granule_per_warp=256, smem_per_sm=233472, reserved_smem_per_cta=1024,
            max_ctas=32, max_warps=64, smsp=4)

COLUMNS = [
    "label", "kernel_id", "variant", "kernel", "target", "isa", "config", "is_base", "num_warps", "num_stages",
    "waves_per_eu", "div_hints", "threads_per_block", "vgpr", "agpr", "sgpr", "regs_for_occupancy",
    "spill_bytes", "spill_detail", "lds_smem_dynamic_bytes", "lds_smem_static_bytes", "lds_smem_per_block_bytes",
    "occ_compiler", "occ_regs_only", "occ_recomputed", "occ_max", "occ_pct", "occ_limiter",
    "occ_recomputed_differs_from_compiler", "occ_unit",
    "g_ld_128", "g_ld_96", "g_ld_64", "g_ld_32", "g_ld_16", "g_st_128", "g_st_64", "g_st_32", "g_st_16",
    "max_global_load_bits", "load_summary", "async_copy_insts", "lds_read_insts", "lds_write_insts", "lds_detail",
    "xlane_insts", "xlane_detail", "mma_insts", "mma_detail", "mma_cycles_per_tile_cu",
    "bf16_cvt_native", "bf16_cvt_emulated", "rsqrt_insts", "rcp_insts", "exp_insts", "barrier_insts",
    "wait_insts", "total_insts", "code_bytes", "compiler", "tools", "kernel_source", "source_sha", "shape",
    "verdict",
]


def _cpu():
    a, b = resource.getrusage(resource.RUSAGE_SELF), resource.getrusage(resource.RUSAGE_CHILDREN)
    return a.ru_utime + a.ru_stime + b.ru_utime + b.ru_stime


def _run(cmd, timeout=300):
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    return r.returncode, r.stdout, r.stderr


def _na(reason):
    return f"n/a ({reason})"


# ----------------------------------------------------------------------------------------------
# Tools and versions (R10)
# ----------------------------------------------------------------------------------------------
def _triton_bin(name):
    import triton
    return os.path.join(os.path.dirname(triton.__file__), "backends", "nvidia", "bin", name)


def tool_versions():
    v = {}
    cuobj = shutil.which("cuobjdump") or _triton_bin("cuobjdump")
    v["cuobjdump_path"] = cuobj
    _, o, _ = _run([cuobj, "--version"])
    mm = re.search(r"V\d+\.\d+\.\d+", o)
    v["cuobjdump"] = mm.group(0) if mm else (o.strip()[-80:] or "?")
    _, o, _ = _run([_triton_bin("ptxas"), "--version"])
    v["ptxas"] = (re.search(r"V\d+\.\d+\.\d+", o) or re.search(r".*", "?")).group(0)
    _, o, _ = _run([os.path.join(CB, "llvm-readelf"), "--version"])
    m = re.search(r"Cray LLVM ([\d.]+).*?based on LLVM version ([\d.]+)", o, re.S)
    v["cce_llvm"] = f"Cray LLVM {m.group(1)} (LLVM {m.group(2)})" if m else o.strip()[:80]
    import triton
    v["triton"] = triton.__version__
    return v


def compiler_string(backend, tv):
    if backend == "cuda":
        return (f"Triton {tv['triton']} (bundled LLVM 22.0.0git 7d5de30, NVPTX) + ptxas {tv['ptxas']} "
                f"(Triton-bundled CUDA 12.8)")
    return f"Triton {tv['triton']} (bundled LLVM 22.0.0git 7d5de30, AMDGPU backend + in-libtriton lld)"


def tools_string(backend, tv):
    if backend == "cuda":
        src = "HPC SDK 25.5" if "hpc_sdk" in tv["cuobjdump_path"] else "Triton-bundled"
        return f"cuobjdump {tv['cuobjdump']} ({src}); ptxas -v {tv['ptxas']} (Triton-bundled)"
    return (f"llvm-readelf/llvm-objdump {tv['cce_llvm']} (CCE 20.0.0); occupancy recomputed by isa_report.py; "
            f"MFMA cycles from AMD matrix_instruction_calculator")


# ----------------------------------------------------------------------------------------------
# NVIDIA
# ----------------------------------------------------------------------------------------------
_SASS_LINE = re.compile(r"^\s+/\*[0-9a-f]{4,}\*/\s+(?:@!?U?P[T0-9]+\s+)?([A-Z][A-Z0-9_.]*)", re.M)


def _width_nv(op):
    parts = op.split(".")
    if "128" in parts:
        return 128
    if "64" in parts:
        return 64
    if "U16" in parts or "S16" in parts:
        return 16
    if "U8" in parts or "S8" in parts:
        return 8
    return 32


def nv_metrics(cubin, ptx, sass_out, tv):
    m = {}
    cuobj = tv["cuobjdump_path"]
    _, o, e = _run([cuobj, "-res-usage", cubin])
    r = re.search(r"REG:(\d+)\s+STACK:(\d+)\s+SHARED:(\d+)\s+LOCAL:(\d+)", o)
    if not r:
        raise RuntimeError(f"cuobjdump -res-usage failed: {o[-300:]} {e[-300:]}")
    m.update(reg=int(r.group(1)), stack=int(r.group(2)), shared_static=int(r.group(3)), local=int(r.group(4)))
    rc, sass, e = _run([cuobj, "-sass", cubin])
    if rc != 0:
        raise RuntimeError(f"cuobjdump -sass failed: {e[-300:]}")
    os.makedirs(os.path.dirname(sass_out), exist_ok=True)
    with open(sass_out, "w") as f:
        f.write(sass)
    ops = _SASS_LINE.findall(sass)
    m["hist"] = Counter(ops)
    # spills from Triton's ptxas -v (same ptxas that produced the cubin)
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        rc, o, e = _run([_triton_bin("ptxas"), "-arch=sm_90a", "-v", ptx, "-o", os.path.join(d, "k.cubin")])
    t = o + e
    st = re.search(r"(\d+) bytes spill stores, (\d+) bytes spill loads", t)
    m["spill_st"], m["spill_ld"] = (int(st.group(1)), int(st.group(2))) if st else (None, None)
    rr = re.search(r"Used (\d+) registers", t)
    m["ptxas_regs"] = int(rr.group(1)) if rr else None
    return m


def nv_occupancy(regs, nw, smem_dyn, smem_static):
    rpw = math.ceil(max(regs, 1) * 32 / SM90["reg_granule_per_warp"]) * SM90["reg_granule_per_warp"]
    warps_regs = SM90["regs_per_sm"] // rpw
    smem_blk = smem_dyn + max(smem_static, SM90["reserved_smem_per_cta"])
    lim = {"registers": warps_regs // nw, "shared memory": SM90["smem_per_sm"] // smem_blk,
           "max CTAs/SM": SM90["max_ctas"], "max warps/SM": SM90["max_warps"] // nw}
    blocks = min(lim.values())
    limiter = " = ".join(k for k in lim if lim[k] == blocks)
    warps = blocks * nw
    regs_only = min(warps_regs, SM90["max_warps"]) / SM90["smsp"]
    return dict(occ_regs_only=regs_only, occ=warps / SM90["smsp"], occ_max=16, pct=100.0 * warps / SM90["max_warps"],
                limiter=f"{limiter} ({blocks} CTA/SM x {nw} warps)", smem_blk=smem_blk)


# ----------------------------------------------------------------------------------------------
# AMD
# ----------------------------------------------------------------------------------------------
_AMD_LINE = re.compile(r"^\s+([sv]_[a-z0-9_]+|(?:global|buffer|flat|scratch|ds)_[a-z0-9_]+)\b(.*)$", re.M)


def _kinfo(asm):
    out = {}
    for key in ("NumVgprs", "NumAgprs", "TotalNumVgprs", "TotalNumSgprs", "ScratchSize", "Occupancy",
                "codeLenInByte", "AccumOffset"):
        mm = re.search(rf"^; {key}\s*[:=]\s*(\d+)", asm, re.M)
        if mm:
            out[key] = int(mm.group(1))
    mm = re.search(r"^; LDSByteSize: (\d+)", asm, re.M)
    if mm:
        out["LDSByteSize"] = int(mm.group(1))
    return out


def _notes(hsaco):
    rc, o, e = _run([os.path.join(CB, "llvm-readelf"), "--notes", hsaco])
    if rc != 0:
        raise RuntimeError(f"llvm-readelf --notes failed: {e[-300:]}")
    keys = ("vgpr_count", "agpr_count", "sgpr_count", "vgpr_spill_count", "sgpr_spill_count",
            "group_segment_fixed_size", "private_segment_fixed_size", "max_flat_workgroup_size", "wavefront_size")
    n = {}
    for k in keys:
        mm = re.search(rf"^\s+\.{k}:\s+(\d+)", o, re.M)
        if mm:
            n[k] = int(mm.group(1))
    return n, o


def _width_amd(mn):
    if mn.endswith("dwordx4") or mn.endswith("b128"):
        return 128
    if mn.endswith("dwordx3") or mn.endswith("b96"):
        return 96
    if mn.endswith("dwordx2") or mn.endswith("b64"):
        return 64
    if mn.endswith("dword") or mn.endswith("b32"):
        return 32
    if "short" in mn or mn.endswith("b16") or "d16" in mn:
        return 16
    if "byte" in mn or mn.endswith("b8"):
        return 8
    return 32


def amd_occupancy(arch, vgpr, agpr, sgpr, nw, lds):
    a = AMD_ARCH[arch]
    al = lambda x, g: int(math.ceil(x / g) * g) if x > 0 else 0
    if a["unified"]:
        regs = al(al(vgpr, 4) + agpr, a["granule"])
    else:
        regs = max(al(vgpr, a["granule"]), al(agpr, a["granule"]))
    w_regs = min(a["max_waves"], a["budget"] // max(regs, 1))
    w_sgpr = 10 if sgpr <= 80 else 9 if sgpr <= 88 else 8 if sgpr <= 100 else 7
    w = min(w_regs, w_sgpr)
    lim = {"VGPR/AGPR (whole WGs)" if w_regs <= w_sgpr else "SGPR (whole WGs)": (4 * w) // nw,
           "LDS 64 KiB/CU": AMD_LDS_PER_CU // lds if lds > 0 else 10 ** 6}
    if nw > 1:
        lim["16 WG/CU barrier limit"] = AMD_MAX_WG_PER_CU
    wg = min(lim.values())
    limiter = " = ".join(k for k in lim if lim[k] == wg)
    occ = wg * nw / 4
    return dict(regs=regs, occ_regs_only=w, occ=occ, occ_max=a["max_waves"], pct=100.0 * occ / a["max_waves"],
                limiter=f"{limiter} ({wg} WG/CU x {nw} waves)")


def mic_cycles(arch, mnemonic, cache):
    key = f"{AMD_ARCH[arch]['mic']}:{mnemonic}"
    if key not in cache:
        try:
            rc, o, e = _run([sys.executable, "-I", MIC, "--architecture", AMD_ARCH[arch]["mic"], "--instruction",
                             mnemonic, "--detail-instruction"], timeout=120)
            mm = re.search(r"Execution cycles:\s*(\d+)", o)
            fl = re.search(r"FLOPs:\s*(\d+)", o)
            cache[key] = dict(cycles=int(mm.group(1)) if mm else None, flops=int(fl.group(1)) if fl else None,
                              err=None if mm else (o + e)[-300:])
        except Exception as ex:  # pragma: no cover
            cache[key] = dict(cycles=None, flops=None, err=repr(ex)[:300])
    return cache[key]


def amd_metrics(asm, hsaco, arch, notes_out, dis_out, mic_cache):
    m = {"kinfo": _kinfo(asm)}
    n, ntext = _notes(hsaco)
    m["notes"] = n
    os.makedirs(os.path.dirname(notes_out), exist_ok=True)
    with open(notes_out, "w") as f:
        f.write(ntext)
    hist, ops = Counter(), []
    for mn, rest in _AMD_LINE.findall(asm):
        hist[mn] += 1
        ops.append((mn, rest))
    m["hist"] = hist
    m["bf16_emul"] = sum(1 for mn, rest in ops if mn == "v_bfe_u32" and re.search(r",\s*16,\s*1\s*$", rest.split(";")[0]))
    mf = Counter({k: v for k, v in hist.items() if k.startswith("v_mfma")})
    m["mfma"] = mf
    m["mic"] = {mn: mic_cycles(arch, mn, mic_cache) for mn in mf}
    if dis_out:
        rc, o, e = _run([os.path.join(CB, "llvm-objdump"), "-d", "--triple=amdgcn-amd-amdhsa", f"--mcpu={arch}",
                         hsaco])
        with open(dis_out, "w") as f:
            f.write(o if rc == 0 else e)
        m["objdump_mfma"] = len(re.findall(r"^\s+v_mfma", o, re.M)) if rc == 0 else None
    return m


def _mfma_tile_cycles(mfma, mic, const):
    """MFMA cycles per CU for one BLOCK_M x BLOCK_N x BLOCK_K tile step: (2*BM*BN*BK / FLOPs per MFMA) MFMAs x
    cycles each, spread over the 4 SIMDs (matrix-instruction calculator numbers). The static instruction count
    in mma_insts also contains the software-pipeline prologue/epilogue copies, so it is not used here."""
    if not mfma:
        return 0
    bm, bn, bk = (const.get(k) for k in ("BLOCK_SIZE_M", "BLOCK_SIZE_N", "BLOCK_SIZE_K"))
    if not (bm and bn and bk):
        return _na("kernel has MFMAs but no BLOCK_SIZE_M/N/K tile constexprs")
    mn = max(mfma, key=mfma.get)
    c = mic.get(mn, {})
    if not c.get("cycles") or not c.get("flops"):
        return _na(f"matrix-instruction calculator has no entry for {mn}")
    n = 2 * bm * bn * bk / c["flops"]
    return int(round(n * c["cycles"] / 4))


# ----------------------------------------------------------------------------------------------
# Row building
# ----------------------------------------------------------------------------------------------
def _fmt(x):
    if isinstance(x, float):
        return f"{x:.2f}".rstrip("0").rstrip(".")
    return str(x)


def _load_summary(w):
    s = " + ".join(f"{b}b x{w[b]}" for b in (128, 96, 64, 32, 16, 8) if w.get(b))
    return s or "none"


def _file(rec, d, ext):
    """Path of <kernel>.<ext>: next to compile.json, or the absolute scratch path listed in rec['files']."""
    p = os.path.join(d, f"{rec['kernel']}.{ext}")
    if os.path.exists(p):
        return p
    for f in rec.get("files", []):
        if os.path.isabs(f) and f.endswith(f".{ext}"):
            return f
    raise FileNotFoundError(p)


def _text_dir(rec, d):
    """Where derived text files go: the dump dir for base configs, the scratch mirror for sweep configs."""
    if rec["is_base"]:
        return d
    return os.path.join(SCRATCH, "isa_ir", os.path.relpath(d, ISA_DIR))


def build_row(rec, d, tv, mic_cache):
    backend = "cuda" if rec["target"].startswith("cuda") else "hip"
    cfg, md = rec["cfg"], rec["metadata"]
    nw = cfg["num_warps"]
    row = dict(label="STATIC", kernel_id=rec["kernel_id"], variant=rec["variant"], kernel=rec["kernel"],
               target=rec["target"], isa=rec["target_dir"], config=rec["tag"], is_base=int(rec["is_base"]),
               num_warps=nw, num_stages=md.get("num_stages"),
               waves_per_eu=cfg.get("waves_per_eu", _na("AMD-only option")),
               div_hints="on" if cfg["hints"] else "off", compiler=compiler_string(backend, tv),
               tools=tools_string(backend, tv), kernel_source=rec["source"], source_sha=rec["source_sha"],
               shape=rec["shape"], verdict="base: GH200 launch options" if rec["is_base"] else "")
    td = _text_dir(rec, d)
    shared = int(md.get("shared") or 0)
    if backend == "cuda":
        warp = 32
        m = nv_metrics(_file(rec, d, "cubin"), _file(rec, d, "ptx"), os.path.join(td, f"{rec['kernel']}.sass"), tv)
        h = m["hist"]
        occ = nv_occupancy(m["reg"], nw, shared, m["shared_static"])
        ld, st, asyncw = Counter(), Counter(), 0
        for op, c in h.items():
            if op.startswith("LDGSTS"):
                asyncw += c
                ld[_width_nv(op)] += 0
            elif op.startswith("LDG") and not op.startswith("LDGDEPBAR"):   # LDGDEPBAR = cp.async commit
                ld[_width_nv(op)] += c
            elif op.startswith("STG"):
                st[_width_nv(op)] += c
        max_ld = max([b for b, c in ld.items() if c] + [_width_nv(o) for o in h if o.startswith("LDGSTS")] or [0])
        lds_r = sum(c for o, c in h.items() if o.split(".")[0] in ("LDS", "LDSM"))
        lds_w = sum(c for o, c in h.items() if o.split(".")[0] == "STS")
        shfl = Counter({o: c for o, c in h.items() if o.startswith("SHFL")})
        mma = Counter({o: c for o, c in h.items() if o.split(".")[0] in ("HGMMA", "HMMA", "IMMA", "QGMMA")})
        spill = (m["spill_st"] or 0) + (m["spill_ld"] or 0)
        row.update(
            threads_per_block=nw * warp, vgpr=m["reg"], agpr=_na("no accumulator register file on sm_90"),
            sgpr=_na("uniform registers not reported by cuobjdump -res-usage"), regs_for_occupancy=m["reg"],
            spill_bytes=spill if m["spill_st"] is not None else _na("ptxas -v printed no spill line"),
            spill_detail=f"ptxas -v: {m['spill_st']} B spill st / {m['spill_ld']} B spill ld; LOCAL {m['local']} B, "
                         f"STACK {m['stack']} B",
            lds_smem_dynamic_bytes=shared, lds_smem_static_bytes=m["shared_static"],
            lds_smem_per_block_bytes=occ["smem_blk"],
            occ_compiler=_na("ptxas prints no occupancy"), occ_regs_only=_fmt(occ["occ_regs_only"]),
            occ_recomputed=_fmt(occ["occ"]), occ_max=occ["occ_max"], occ_pct=_fmt(occ["pct"]),
            occ_limiter=occ["limiter"], occ_recomputed_differs_from_compiler=_na("no compiler occupancy on NVIDIA"),
            occ_unit="warps per SM sub-partition (scheduler), max 16",
            g_ld_128=ld[128], g_ld_96=_na("no 96-bit LDG on sm_90"), g_ld_64=ld[64], g_ld_32=ld[32], g_ld_16=ld[16],
            g_st_128=st[128], g_st_64=st[64], g_st_32=st[32], g_st_16=st[16], max_global_load_bits=max_ld,
            load_summary=(_load_summary(ld) + (f"; LDGSTS x{asyncw}" if asyncw else "")),
            async_copy_insts=asyncw, lds_read_insts=lds_r, lds_write_insts=lds_w,
            lds_detail=", ".join(f"{o} x{c}" for o, c in sorted(h.items()) if o.split(".")[0] in
                                 ("LDS", "LDSM", "STS")) or "none",
            xlane_insts=sum(shfl.values()), xlane_detail=", ".join(f"{o} x{c}" for o, c in shfl.items()) or "none",
            mma_insts=sum(mma.values()), mma_detail=", ".join(f"{o} x{c}" for o, c in mma.items()) or "none",
            mma_cycles_per_tile_cu=0 if not mma else _na("no public per-instruction HGMMA/HMMA cycle table"),
            bf16_cvt_native=sum(c for o, c in h.items() if o.startswith("F2FP.BF16")), bf16_cvt_emulated=0,
            rsqrt_insts=h.get("MUFU.RSQ", 0), rcp_insts=h.get("MUFU.RCP", 0), exp_insts=h.get("MUFU.EX2", 0),
            barrier_insts=sum(c for o, c in h.items() if o.startswith("BAR")),
            wait_insts=sum(c for o, c in h.items() if o.startswith("DEPBAR")),
            total_insts=sum(h.values()), code_bytes=16 * sum(h.values()))
        hist = dict(h)
        extra = dict(ptxas_regs=m["ptxas_regs"])
    else:
        warp = 64
        arch = rec["target_dir"]
        m = amd_metrics(open(_file(rec, d, "amdgcn")).read(), _file(rec, d, "hsaco"), arch,
                        os.path.join(td, f"{rec['kernel']}.notes.txt"),
                        os.path.join(td, f"{rec['kernel']}.hsaco.dis.s") if rec["is_base"] else None, mic_cache)
        h, k, n = m["hist"], m["kinfo"], m["notes"]
        # ArchVGPR / AGPR split from the Kernel info comments (on gfx90a+ the notes' .vgpr_count is the unified
        # total align4(VGPR)+AGPR); SGPRs from the notes (= TotalNumSgprs incl. VCC/flat-scratch/XNACK)
        vg = k.get("NumVgprs", n.get("vgpr_count"))
        ag = k.get("NumAgprs", n.get("agpr_count", 0))
        sg = n.get("sgpr_count", k.get("TotalNumSgprs"))
        lds = shared + n.get("group_segment_fixed_size", 0)
        occ = amd_occupancy(arch, vg, ag, sg, nw, lds)
        ld, st = Counter(), Counter()
        for mn, c in h.items():
            if re.match(r"(global|buffer|flat)_load_", mn):
                ld[_width_amd(mn)] += c
            elif re.match(r"(global|buffer|flat)_store_", mn):
                st[_width_amd(mn)] += c
        dsr = Counter({o: c for o, c in h.items() if o.startswith("ds_read")})
        dsw = Counter({o: c for o, c in h.items() if o.startswith("ds_write")})
        xl = Counter()
        for o, c in h.items():
            if o.endswith("_dpp"):
                xl["dpp:" + o] += c
            elif o.startswith(("ds_swizzle", "ds_bpermute", "ds_permute", "v_permlane", "v_readlane",
                               "v_writelane")):
                xl[o] += c
        n_dpp = sum(c for o, c in xl.items() if o.startswith("dpp:"))
        xl_detail = (f"DPP x{n_dpp} ({', '.join(f'{o[4:]} x{c}' for o, c in xl.items() if o.startswith('dpp:'))})"
                     if n_dpp else "DPP x0")
        for name in ("ds_swizzle_b32", "ds_bpermute_b32", "ds_permute_b32"):
            xl_detail += f"; {name} x{h.get(name, 0)}"
        others = {o: c for o, c in xl.items() if not o.startswith("dpp:") and o not in
                  ("ds_swizzle_b32", "ds_bpermute_b32", "ds_permute_b32")}
        if others:
            xl_detail += "; " + ", ".join(f"{o} x{c}" for o, c in others.items())
        spill_v, spill_s = n.get("vgpr_spill_count", 0), n.get("sgpr_spill_count", 0)
        scratch = k.get("ScratchSize", n.get("private_segment_fixed_size", 0))
        occ_c = k.get("Occupancy")
        row.update(
            threads_per_block=nw * warp, vgpr=vg, agpr=ag, sgpr=sg, regs_for_occupancy=occ["regs"],
            spill_bytes=scratch,
            spill_detail=f"ScratchSize {scratch} B/lane; vgpr_spill_count {spill_v}, sgpr_spill_count {spill_s}",
            lds_smem_dynamic_bytes=shared, lds_smem_static_bytes=n.get("group_segment_fixed_size", 0),
            lds_smem_per_block_bytes=lds,
            occ_compiler=occ_c if occ_c is not None else _na("no Occupancy line"),
            occ_regs_only=occ["occ_regs_only"], occ_recomputed=_fmt(occ["occ"]), occ_max=occ["occ_max"],
            occ_pct=_fmt(occ["pct"]), occ_limiter=occ["limiter"],
            occ_recomputed_differs_from_compiler=("yes" if occ_c is not None and abs(occ["occ"] - occ_c) > 1e-9
                                                  else "no"),
            occ_unit="waves per SIMD (EU)",
            g_ld_128=ld[128], g_ld_96=ld[96], g_ld_64=ld[64], g_ld_32=ld[32], g_ld_16=ld[16],
            g_st_128=st[128], g_st_64=st[64], g_st_32=st[32], g_st_16=st[16],
            max_global_load_bits=max([b for b, c in ld.items() if c] or [0]), load_summary=_load_summary(ld),
            async_copy_insts=sum(c for o, c in h.items() if o.startswith(("buffer_load", "global_load")) and
                                 "lds" in o),
            lds_read_insts=sum(dsr.values()), lds_write_insts=sum(dsw.values()),
            lds_detail=", ".join(f"{o} x{c}" for o, c in sorted((dsr + dsw).items())) or "none",
            xlane_insts=sum(xl.values()), xlane_detail=xl_detail,
            mma_insts=sum(m["mfma"].values()),
            mma_detail=", ".join(f"{o} x{c}" for o, c in m["mfma"].items()) or "none",
            mma_cycles_per_tile_cu=_mfma_tile_cycles(m["mfma"], m["mic"], rec["constexprs"]),
            bf16_cvt_native=h.get("v_cvt_pk_bf16_f32", 0), bf16_cvt_emulated=m["bf16_emul"],
            rsqrt_insts=sum(c for o, c in h.items() if o.startswith("v_rsq_f32")),
            rcp_insts=sum(c for o, c in h.items() if o.startswith("v_rcp_f32")),
            exp_insts=sum(c for o, c in h.items() if o.startswith("v_exp_f32")),
            barrier_insts=h.get("s_barrier", 0), wait_insts=h.get("s_waitcnt", 0), total_insts=sum(h.values()),
            code_bytes=k.get("codeLenInByte", _na("no codeLenInByte comment")))
        hist = dict(h)
        extra = dict(kernel_info=k, notes=n, objdump_mfma=m.get("objdump_mfma"),
                     accvgpr_moves=sum(c for o, c in h.items() if o.startswith("v_accvgpr")))
        row["mma_detail"] += (f"; v_accvgpr_read/write x{extra['accvgpr_moves']}" if extra["accvgpr_moves"] else "")
    with open(os.path.join(td, f"{rec['kernel']}.hist.json"), "w") as f:
        json.dump(dict(histogram=dict(sorted(hist.items(), key=lambda kv: -kv[1])), **extra,
                       row={k: row[k] for k in COLUMNS if k in row}), f, indent=1, default=str)
    return row


# ----------------------------------------------------------------------------------------------
# Static tuning-pair verdicts
# ----------------------------------------------------------------------------------------------
def _num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def verdict(row, base):
    ch = []
    for k, lab in (("num_warps", "num_warps"), ("num_stages", "num_stages"), ("waves_per_eu", "waves_per_eu"),
                   ("div_hints", "hints")):
        if str(row[k]) != str(base[k]):
            ch.append(f"{lab} {base[k]}->{row[k]}")
    head = ", ".join(ch) or "same options"
    d = []

    def diff(key, lab, unit=""):
        if str(row[key]) != str(base[key]):
            d.append(f"{lab} {base[key]}->{row[key]}{unit}")

    diff("vgpr", "VGPR" if row["isa"] != "sm_90a" else "REG")
    if row["isa"] != "sm_90a":
        diff("agpr", "AGPR")
    diff("spill_bytes", "scratch" if row["isa"] != "sm_90a" else "spill", " B")
    diff("lds_smem_dynamic_bytes", "LDS" if row["isa"] != "sm_90a" else "smem", " B")
    diff("occ_recomputed", "occupancy")
    if str(row["occ_compiler"]) != str(base["occ_compiler"]) and row["isa"] != "sm_90a":
        d.append(f"(compiler line {base['occ_compiler']}->{row['occ_compiler']})")
    if row["load_summary"] != base["load_summary"]:
        d.append(f"loads [{base['load_summary']}]->[{row['load_summary']}]")
    diff("mma_insts", "MMA")
    diff("xlane_insts", "cross-lane ops")
    diff("lds_read_insts", "LDS reads")
    if not d:
        return f"{head}: no static change"
    tail = []
    sp_b, sp_r = _num(base["spill_bytes"]) or 0, _num(row["spill_bytes"]) or 0
    oc_b, oc_r = _num(base["occ_recomputed"]) or 0, _num(row["occ_recomputed"]) or 0
    if sp_r > sp_b and oc_r > oc_b:
        tail.append("occupancy bought with spills: only hardware can settle it")
    elif sp_r > sp_b:
        tail.append("spills without an occupancy gain: worse")
    elif oc_r > oc_b:
        tail.append("higher occupancy at no spill cost")
    elif oc_r < oc_b:
        tail.append("lower occupancy")
    if (str(row["num_stages"]) == str(base["num_stages"]) and (_num(row["mma_insts"]) or 0) < (_num(base["mma_insts"]) or 0)
            and (_num(row["lds_smem_dynamic_bytes"]) or 0) < (_num(base["lds_smem_dynamic_bytes"]) or 0)):
        tail.append("multi-stage pipelining not applied (fewer static MMA copies, smaller LDS/smem ring)")
    if (_num(row["max_global_load_bits"]) or 0) < (_num(base["max_global_load_bits"]) or 0):
        tail.append(f"narrower global loads ({base['max_global_load_bits']}->{row['max_global_load_bits']} bit)")
    return f"{head}: " + "; ".join(d) + (" => " + "; ".join(tail) if tail else "")


# ----------------------------------------------------------------------------------------------
def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--kernels", default="", help="comma list of kernel_ids (default: all)")
    p.add_argument("--cpu-limit", type=float, default=1080.0)
    p.add_argument("--csv", default=CSV_PATH)
    a = p.parse_args(argv)
    kids = [k for k in a.kernels.split(",") if k]
    tv = tool_versions()
    mic_cache = {}
    if os.path.exists(MIC_CACHE):
        mic_cache = json.load(open(MIC_CACHE))
    recs = []
    for j in sorted(glob.glob(os.path.join(ISA_DIR, "*", "triton_*", "*", "xc-*", "*.compile.json"))):
        r = json.load(open(j))
        if kids and r["kernel_id"] not in kids:
            continue
        recs.append((j, r))
    # existing rows of other kernels are kept when --kernels restricts the run
    rows, failed = [], []
    if kids and os.path.exists(a.csv):
        with open(a.csv) as f:
            rows = [r for r in csv.DictReader(f) if r["kernel_id"] not in kids]
    for j, r in recs:
        if _cpu() > a.cpu_limit:
            print(f"CPU limit reached after {len(rows)} rows; rerun with --kernels for the rest", flush=True)
            break
        if not r.get("ok"):
            failed.append((r["target"], r["kernel_id"], r["variant"], r["kernel"], r["tag"], r.get("error")))
            continue
        try:
            rows.append(build_row(r, os.path.dirname(j), tv, mic_cache))
        except Exception as e:
            failed.append((r["target"], r["kernel_id"], r["variant"], r["kernel"], r["tag"], f"report: {e!r}"))
    os.makedirs(SCRATCH, exist_ok=True)
    with open(MIC_CACHE, "w") as f:
        json.dump(mic_cache, f, indent=1)
    base = {(x["kernel_id"], x["variant"], x["kernel"], x["target"]): x for x in rows if str(x["is_base"]) == "1"}
    for x in rows:
        if str(x["is_base"]) != "1":
            b = base.get((x["kernel_id"], x["variant"], x["kernel"], x["target"]))
            x["verdict"] = verdict(x, b) if b else "no base row to pair with"
    order = {"cuda:90": 0, "hip:gfx908": 1, "hip:gfx90a": 2, "hip:gfx942": 3}
    rows.sort(key=lambda x: (x["kernel_id"], x["variant"], x["kernel"], -int(x["is_base"]), order[x["target"]],
                             x["config"]))
    empty = [(x["kernel"], x["target"], x["config"], c) for x in rows for c in COLUMNS if str(x.get(c, "")) == ""]
    with open(a.csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {a.csv}: {len(rows)} rows, {len(failed)} failed, {len(empty)} empty cells; CPU {_cpu():.1f} s")
    for fl in failed:
        print("FAILED", *fl)
    for e in empty[:20]:
        print("EMPTY", *e)
    return 0


if __name__ == "__main__":
    sys.exit(main())
