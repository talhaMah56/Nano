"""
Summarize an Nsight Compute raw CSV export (`ncu --import X.ncu-rep --csv --page raw --print-units base`)
into one record per kernel name (task card BAS-6; also used for our kernels' measured bytes, KT-2/KT-3).

Per kernel name: launches seen, and for the LAST launch (the warm instance when a driver launches every kernel
a few times; for a one-step capture the launches of a name are the same code at the same shape) plus the
median over launches: duration, DRAM bytes read/write, achieved DRAM GB/s, registers/thread, block/grid size,
theoretical and achieved occupancy, the top-2 warp stall reasons, global-load sectors per request.

  python -m cisc662.profiling.ncu_summary RAW.csv --json OUT.json [--top 15] [--match REGEX]
"""
import argparse
import csv
import json
import re
import statistics
import sys

from cisc662.analysis.inductor_fusion import parse_number

M = dict(
    time_ns="gpu__time_duration.sum",
    dram_read="dram__bytes_read.sum",
    dram_write="dram__bytes_write.sum",
    regs="launch__registers_per_thread",
    block="launch__block_size",
    grid="launch__grid_size",
    occ_theoretical="sm__maximum_warps_per_active_cycle_pct",
    occ_achieved="sm__warps_active.avg.pct_of_peak_sustained_active",
    dram_pct="gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed",
    sm_pct="sm__throughput.avg.pct_of_peak_sustained_elapsed",
    ld_sectors="l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum",
    ld_requests="l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum",
    smem_dyn="launch__shared_mem_per_block_dynamic",
    smem_static="launch__shared_mem_per_block_static",
)
STALL_RE = re.compile(r"^smsp__average_warps_issue_stalled_(\w+?)_per_issue_active\.ratio$")
SCALE = {"byte": 1.0, "kbyte": 1e3, "mbyte": 1e6, "gbyte": 1e9, "nsecond": 1.0, "usecond": 1e3, "msecond": 1e6,
         "second": 1e9}


def load(path):
    with open(path, newline="", errors="replace") as f:
        lines = f.read().splitlines()
    start = next(i for i, l in enumerate(lines) if l.startswith('"ID"'))
    rows = list(csv.reader(lines[start:]))
    header, body = rows[0], [r for r in rows[1:] if r]
    units = {}
    if body and not body[0][0].strip():
        units = {c: body[0][i].strip().lower() for i, c in enumerate(header) if i < len(body[0])}
        body = body[1:]
    stall_cols = [c for c in header if STALL_RE.match(c)]
    out = []
    for r in body:
        if len(r) < len(header) or parse_number(r[0]) is None:
            continue
        d = dict(zip(header, r))
        rec = {"id": int(parse_number(d["ID"])), "name": d.get("Kernel Name", "?")}
        for k, metric in M.items():
            v = parse_number(d.get(metric))
            if v is not None:
                v *= SCALE.get(units.get(metric, ""), 1.0)
            rec[k] = v
        stalls = {STALL_RE.match(c).group(1): parse_number(d.get(c)) for c in stall_cols}
        rec["stalls"] = {k: v for k, v in stalls.items() if v is not None}
        out.append(rec)
    return out


def summarize(recs, match=None):
    by = {}
    for r in recs:
        if match and not re.search(match, r["name"]):
            continue
        by.setdefault(r["name"], []).append(r)
    res = []
    for name, rs in by.items():
        rs.sort(key=lambda r: r["id"])
        last = rs[-1]
        times = [r["time_ns"] for r in rs if r["time_ns"]]
        rec = {"name": name, "launches": len(rs), "warm": {k: last.get(k) for k in M}}
        w = rec["warm"]
        b = (w.get("dram_read") or 0) + (w.get("dram_write") or 0)
        w["dram_bytes"] = b
        w["dram_GBps"] = b / w["time_ns"] if w.get("time_ns") else None      # bytes/ns = GB/s
        if w.get("ld_sectors") and w.get("ld_requests"):
            w["sectors_per_request"] = w["ld_sectors"] / w["ld_requests"]
        st = sorted(last["stalls"].items(), key=lambda kv: -(kv[1] or 0))
        w["top_stalls"] = [[k, round(v, 3)] for k, v in st[:2]]
        rec["median_time_ns"] = statistics.median(times) if times else None
        rec["total_time_ns"] = sum(times) if times else None
        # time-weighted aggregate over all launches of the name (a one-step capture: launches of one name can be
        # different shapes, e.g. a cuBLAS nvjet kernel serving several GEMMs, so "last launch" is not representative)
        tw = sum(r["time_ns"] or 0 for r in rs)
        agg = {"time_ns": tw, "dram_bytes": sum((r.get("dram_read") or 0) + (r.get("dram_write") or 0) for r in rs)}
        agg["dram_GBps"] = agg["dram_bytes"] / tw if tw else None
        for k in ("occ_achieved", "occ_theoretical", "dram_pct", "sm_pct"):
            vals = [(r.get(k), r["time_ns"] or 0) for r in rs if r.get(k) is not None]
            agg[k] = sum(v * t for v, t in vals) / sum(t for _, t in vals) if vals and sum(t for _, t in vals) else None
        reasons = {}
        for r in rs:
            for k, v in r["stalls"].items():
                reasons[k] = reasons.get(k, 0.0) + (v or 0) * (r["time_ns"] or 0)
        agg["top_stalls"] = [[k, round(v / tw, 3)] for k, v in sorted(reasons.items(), key=lambda kv: -kv[1])[:2]] if tw else []
        rec["aggregate"] = agg
        res.append(rec)
    res.sort(key=lambda r: -(r["total_time_ns"] or 0))
    return res


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("raw_csv")
    p.add_argument("--json", default=None)
    p.add_argument("--top", type=int, default=0)
    p.add_argument("--match", default=None)
    a = p.parse_args(argv)
    res = summarize(load(a.raw_csv), a.match)
    if a.top:
        res = res[:a.top]
    for r in res:
        w, g = r["warm"], r["aggregate"]
        print(f"{r['name'][:64]:64s} x{r['launches']:<4d} last {(w.get('time_ns') or 0) / 1e3:8.1f} us "
              f"{w['dram_GBps'] or 0:6.0f} GB/s | all {g['time_ns'] / 1e3:9.1f} us {g['dram_GBps'] or 0:6.0f} GB/s "
              f"occ {w.get('occ_theoretical')}/{g['occ_achieved'] and round(g['occ_achieved'], 1)} regs={w.get('regs')} "
              f"stalls={g['top_stalls']}")
    if a.json:
        with open(a.json, "w") as f:
            json.dump(res, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
