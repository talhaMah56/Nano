"""
GPU-less cross-compilation of every registered Triton kernel (task card AMD-1, step 2).

Targets: cuda:90 (sm_90a, GH200), hip:gfx908 (MI100), hip:gfx90a (MI210/MI250X), hip:gfx942 (MI300).
For each (kernel spec, target, config) it runs triton.compile(ASTSource(...), target=GPUTarget(...)) with a
per-target TRITON_CACHE_DIR / TRITON_DUMP_DIR (so the target-independent .ttir of one target never
overwrites another's) and writes

  cisc662/results/isa/<kernel_id>/triton_<variant>/<target>/xc-<tag>/<kernel>.{ttgir,llir,ptx,cubin}   (cuda)
  cisc662/results/isa/<kernel_id>/triton_<variant>/<target>/xc-<tag>/<kernel>.{ttgir,llir,amdgcn,hsaco} (hip)
  ... /<kernel>.compile.json      options, Triton metadata (shared = dynamic LDS/smem), versions, CPU seconds

Sweep (non-base) configs keep only the binaries (cubin / hsaco) and compile.json in the repo; their text stages
(ttgir, llir, ptx, amdgcn: ~19 MB of amdgcn alone with debug .loc lines) go to
$CISC662_WORK/scratch/AMD-1/isa_ir/<same relative path>/ (absolute paths listed in compile.json "files").
`xc-<tag>` keeps cross-compiles apart from GPU-run dumps (`--dump-isa`) that the kernel cards write next to
them. SASS, cuobjdump -res-usage, llvm-readelf --notes and the metrics are produced by isa_report.py.

Configs ("sets"):
  base    the GH200 launch options from the registry (num_warps, num_stages), hints on, waves_per_eu default
  sweep   static tuning pairs: num_warps in {4, 8}, waves_per_eu in {default(1), 2, 4} (hip only), divisibility
          hints on/off, num_stages in {2, 3} (kernels with a pipelined loop: the matmul). "ofat" specs vary one
          factor at a time from base; "full" specs (matmul) take the full factorial.

Login-node rules (NCSA kills > 30 CPU-min): run under `nice -n 19`; a process compiles at most 100 kernels
(--count, enforced) and stops itself after --cpu-limit CPU-seconds (default 1080 = 18 min); jobs whose
compile.json already exists are skipped, so repeat the command with the next --start until "remaining 0".

  python -m cisc662.analysis.xcompile --list [--set all]
  nice -n 19 python -m cisc662.analysis.xcompile --set base --start 0 --count 100
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import resource
import sys
import time

WORK = os.environ.get("CISC662_WORK", "/work/nvme/bffp/tmahmood1/cisc662")
SCRATCH = os.path.join(WORK, "scratch", "AMD-1")
os.environ.setdefault("MAX_JOBS", "8")

import triton  # noqa: E402
from triton.backends.compiler import GPUTarget  # noqa: E402

from cisc662.analysis import kernel_registry as KR  # noqa: E402

RESULTS = os.path.join(KR.REPO, "cisc662", "results")
ISA_DIR = os.path.join(RESULTS, "isa")

# target key -> (GPUTarget, directory name)
TARGETS = {
    "cuda:90": (GPUTarget("cuda", 90, 32), "sm_90a"),
    "hip:gfx908": (GPUTarget("hip", "gfx908", 64), "gfx908"),
    "hip:gfx90a": (GPUTarget("hip", "gfx90a", 64), "gfx90a"),
    "hip:gfx942": (GPUTarget("hip", "gfx942", 64), "gfx942"),
}
STAGES = {"cuda": ("ttgir", "llir", "ptx", "cubin"), "hip": ("ttgir", "llir", "amdgcn", "hsaco")}
TEXT_STAGES = ("ttgir", "llir", "ptx", "amdgcn")
WPE_DEFAULT = 1          # HIPOptions.waves_per_eu default in triton 3.5.1 (amdgpu-waves-per-eu="1")
STAGES_DEFAULT = {"cuda": 3, "hip": 2}   # CUDAOptions / HIPOptions defaults (only matter with a pipelined loop)


def cfg_tag(cfg: dict) -> str:
    t = f"nw{cfg['num_warps']}"
    if "num_stages" in cfg:
        t += f"_st{cfg['num_stages']}"
    if "waves_per_eu" in cfg:
        t += f"_wpe{cfg['waves_per_eu']}"
    return t + ("_h1" if cfg["hints"] else "_h0")


def base_cfg(spec, backend: str) -> dict:
    c = dict(num_warps=spec.options["num_warps"], hints=True)
    if spec.has_loop:
        c["num_stages"] = spec.options.get("num_stages", STAGES_DEFAULT[backend])
    if backend == "hip":
        c["waves_per_eu"] = WPE_DEFAULT
    return c


def sweep_cfgs(spec, backend: str):
    """[(cfg, is_base)] for one spec and backend; the base config comes first."""
    base = base_cfg(spec, backend)
    axes = {"num_warps": [4, 8], "hints": [True, False]}
    if spec.has_loop:
        axes["num_stages"] = [2, 3]
    if backend == "hip":
        axes["waves_per_eu"] = [WPE_DEFAULT, 2, 4]
    out, seen = [(base, True)], {cfg_tag(base)}
    if spec.sweep == "full":
        keys = list(axes)
        cands = [dict(zip(keys, vals)) for vals in itertools.product(*(axes[k] for k in keys))]
    elif spec.sweep == "ofat":
        cands = []
        for k, vals in axes.items():
            for v in vals:
                c = dict(base)
                c[k] = v
                cands.append(c)
    else:
        cands = []
    for c in cands:
        t = cfg_tag(c)
        if t not in seen:
            seen.add(t)
            out.append((c, False))
    return out


def jobs(which: str = "all", kernel_ids=None, targets=None):
    """Deterministic job list: dicts with spec, target key, cfg, is_base, outdir."""
    out = []
    for spec in KR.specs(kernel_ids):
        for tkey, (target, tdir) in TARGETS.items():
            if targets and tkey not in targets:
                continue
            for cfg, is_base in sweep_cfgs(spec, target.backend):
                if which == "base" and not is_base:
                    continue
                if which == "sweep" and is_base:
                    continue
                d = os.path.join(ISA_DIR, spec.kernel_id, f"triton_{spec.variant}", tdir, f"xc-{cfg_tag(cfg)}")
                out.append(dict(spec=spec, tkey=tkey, cfg=cfg, is_base=is_base, outdir=d))
    return out


def _cpu_seconds() -> float:
    a = resource.getrusage(resource.RUSAGE_SELF)
    b = resource.getrusage(resource.RUSAGE_CHILDREN)
    return a.ru_utime + a.ru_stime + b.ru_utime + b.ru_stime


def _versions() -> dict:
    v = {"triton": triton.__version__, "llvm": "22.0.0git (7d5de3033187, bundled in libtriton)"}
    try:
        import torch
        v["torch"] = torch.__version__
    except Exception:  # pragma: no cover
        pass
    try:
        from triton.backends.nvidia.compiler import get_ptxas
        p = get_ptxas()
        v["ptxas"] = f"{getattr(p, 'version', '?')} ({getattr(p, 'path', p)})"
    except Exception as e:  # pragma: no cover
        v["ptxas"] = f"unknown: {e!r}"[:200]
    return v


def compile_one(job, versions) -> dict:
    spec, cfg = job["spec"], job["cfg"]
    target, tdir = TARGETS[job["tkey"]]
    os.environ["TRITON_CACHE_DIR"] = os.path.join(SCRATCH, "triton_cache", tdir)
    os.environ["TRITON_DUMP_DIR"] = os.path.join(SCRATCH, "triton_dump", tdir)
    sig, const, attrs = spec.build(hints=cfg["hints"])
    opts = {k: cfg[k] for k in ("num_warps", "num_stages", "waves_per_eu") if k in cfg}
    t0, c0 = time.time(), _cpu_seconds()
    rec = dict(kernel_id=spec.kernel_id, variant=spec.variant, kernel=spec.kname, target=job["tkey"],
               target_dir=tdir, tag=cfg_tag(cfg), is_base=job["is_base"], cfg=cfg, options=opts,
               signature=sig, constexprs={k: v for k, v in const.items()},
               divisibility_args=[spec.fn.arg_names[k[0]] for k in attrs], source=spec.source,
               source_sha=spec.source_sha(), shape=spec.shape, notes=spec.notes, versions=versions,
               mode=f"xcompile GPUTarget({target.backend!r}, {target.arch!r}, {target.warp_size}) on the login node, no GPU", label="STATIC")
    os.makedirs(job["outdir"], exist_ok=True)
    try:
        src = triton.compiler.ASTSource(fn=spec.fn, signature=sig, constexprs=const, attrs=attrs)
        ck = triton.compile(src, target=target, options=opts)
        md = ck.metadata._asdict()
        written = []
        for st in STAGES[target.backend]:
            data = ck.asm.get(st)
            if data is None:
                continue
            d = job["outdir"]
            if not job["is_base"] and st in TEXT_STAGES:
                # sweep configs: keep the repo light (binaries + json); text stages go to the scratch mirror
                d = os.path.join(SCRATCH, "isa_ir", os.path.relpath(job["outdir"], ISA_DIR))
                os.makedirs(d, exist_ok=True)
            fn = os.path.join(d, f"{spec.kname}.{st}")
            with open(fn, "wb" if isinstance(data, (bytes, bytearray)) else "w") as f:
                f.write(data)
            written.append(fn if d != job["outdir"] else os.path.basename(fn))
        rec.update(ok=True, files=written, name=ck.name, hash=ck.hash,
                   metadata={k: md.get(k) for k in ("shared", "num_warps", "num_stages", "num_ctas", "arch",
                                                    "ptx_version", "warp_size", "waves_per_eu", "cluster_dims",
                                                    "tmem_size", "global_scratch_size", "profile_scratch_size")
                             if k in md})
    except Exception as e:  # a failure is a result: record it and continue
        rec.update(ok=False, error=f"{type(e).__name__}: {e}"[:3000])
    rec.update(wall_s=round(time.time() - t0, 2), cpu_s=round(_cpu_seconds() - c0, 2))
    with open(os.path.join(job["outdir"], f"{spec.kname}.compile.json"), "w") as f:
        json.dump(rec, f, indent=1, default=str)
    return rec


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--set", default="base", choices=("base", "sweep", "all"))
    p.add_argument("--kernels", default="", help="comma list of kernel_ids (default: all)")
    p.add_argument("--targets", default="", help="comma list of target keys, e.g. hip:gfx90a,cuda:90")
    p.add_argument("--start", type=int, default=0, help="index into the pending (not yet compiled) jobs")
    p.add_argument("--count", type=int, default=100, help="max compiles in this process (<= 100)")
    p.add_argument("--cpu-limit", type=float, default=1080.0, help="stop after this many CPU-seconds")
    p.add_argument("--force", action="store_true", help="recompile even if compile.json exists")
    p.add_argument("--list", action="store_true")
    a = p.parse_args(argv)
    if a.count > 100:
        p.error("--count must be <= 100 (login-node rule: batch <= 100 compiles per process)")
    kids = [k for k in a.kernels.split(",") if k] or None
    tgts = [t for t in a.targets.split(",") if t] or None
    js = jobs(a.set, kids, tgts)
    done = lambda j: os.path.exists(os.path.join(j["outdir"], f"{j['spec'].kname}.compile.json"))
    pending = js if a.force else [j for j in js if not done(j)]
    if a.list:
        for i, j in enumerate(js):
            print(f"{i:4d} {'done' if done(j) else 'todo'} {j['tkey']:11s} {j['spec'].kernel_id:8s} "
                  f"{j['spec'].variant:22s} {j['spec'].kname:32s} {cfg_tag(j['cfg'])}{' BASE' if j['is_base'] else ''}")
        print(f"total {len(js)}, pending {len(pending)}")
        return 0
    batch = pending[a.start:a.start + a.count]
    versions = _versions()
    nok = nfail = 0
    for j in batch:
        if _cpu_seconds() > a.cpu_limit:
            print(f"CPU limit {a.cpu_limit:.0f} s reached; stopping (rerun to continue)", flush=True)
            break
        r = compile_one(j, versions)
        nok += r["ok"]
        nfail += not r["ok"]
        sh = r.get("metadata", {}).get("shared")
        print(f"{'OK  ' if r['ok'] else 'FAIL'} {r['target']:11s} {r['kernel_id']}/{r['variant']}/{r['kernel']} "
              f"{r['tag']} shared={sh} cpu={r['cpu_s']}s" + ("" if r["ok"] else f"  {r['error'][:300]}"),
              flush=True)
    remaining = len([j for j in js if not done(j)])
    print(f"batch: {nok} ok, {nfail} failed; process CPU {_cpu_seconds():.1f} s; remaining {remaining}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
