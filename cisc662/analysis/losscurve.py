"""
Loss-curve noise envelope and gate G4 (card BAS-2; PLAN.md section 7.3).

Two modes over nanochat base_train logs (R12-L: 1000 steps, eval every 250 steps):

  --make-envelope  A.log B.log are two stock runs with the same seed and data order (slurm/s2_noise_band.sbatch).
                   band = max over steps > 100 of |loss_a - loss_b|; val_bpb_delta = |final val bpb a - b|.
                   Writes the envelope JSON and an overlay + delta figure with the band shaded. Exit code 1 (and
                   valid=false in the JSON, which compare() refuses) if a log is truncated, has NaN/Inf, or the two
                   runs share no eval step.
  --envelope E     stock.log variant.log are compared under G4:
                     |loss_variant - loss_stock| <= max(2 x band, 0.01) at every step > 100,
                     |final val bpb delta|       <= max(0.005, 2 x val_bpb_delta),
                     no NaN/Inf in either log.
                   Prints a line starting with PASS or FAIL (with the worst step); exit code 0 on PASS, 1 on FAIL.

The logged "loss:" field is already nanochat's debiased EMA(0.9) of the train loss (base_train.py,
smooth_train_loss), so it is used as-is and never smoothed again.

Results carry the job id parsed from the run directory name (.../s2_noise_<jobid>/noise_a.log). Besides the
requested --out / --fig path, a job-stamped copy (<name>_<jobid>.json/.png) is written next to it, so a later
envelope replaces the canonical file but never another job's results.

Usage:
  python -m cisc662.analysis.losscurve cisc662/results/runs/s2_noise_<jobid>/noise_a.log \
      cisc662/results/runs/s2_noise_<jobid>/noise_b.log --make-envelope
  python -m cisc662.analysis.losscurve stock.log variant.log --envelope cisc662/results/db/loss/noise_envelope.json \
      [--out cisc662/results/db/loss/<variant>_<jobid>.json] [--fig cisc662/results/figures/loss_<variant>.png]
"""
import argparse
import datetime
import json
import math
import os
import re
import shutil
import sys

from cisc662.analysis.summarize_s1 import parse_train_log

ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "results"))
DEFAULT_ENVELOPE = os.path.join(ROOT, "db", "loss", "noise_envelope.json")
DEFAULT_FIG = os.path.join(ROOT, "figures", "loss_noise_band.png")

AFTER_STEP = 100        # G4 compares every step strictly after this one
LOSS_TOL_FLOOR = 0.01   # |delta loss| <= max(2 x band, 0.01)
VAL_TOL_FLOOR = 0.005   # |delta final val bpb| <= max(0.005, 2 x stock-stock delta)
G4_RULE = ("G4: |loss_var - loss_stock| <= max(2*band, 0.01) at every step > 100; "
           "|final val bpb delta| <= max(0.005, 2*val_bpb_delta); no NaN/Inf")

# parse_train_log's regexes only accept digits, so a "loss: nan" step or "Validation bpb: inf" line is silently
# dropped there. These looser patterns catch them (and the step count "step 00010/01000").
LOOSE_STEP_RE = re.compile(r"step (\d+)/(\d+) .*?\| loss: (\S+) ")
LOOSE_VAL_RE = re.compile(r"Step (\d+) \| Validation bpb: (\S+)")
JOB_RE = re.compile(r"_(\d+)$")


def job_id_from_path(path):
    """Job id from the run directory name (s2_noise_3401234/noise_a.log -> '3401234'), else None."""
    m = JOB_RE.search(os.path.basename(os.path.dirname(os.path.abspath(path))))
    return m.group(1) if m else None


def _float(tok):
    try:
        return float(tok)
    except ValueError:
        return float("nan")  # anything unparsable in a numeric field counts as non-finite


def load_curve(path):
    """Per-step loss and val bpb of one base_train log.
    Returns dict(path, job_id, loss={step: loss}, val={step: bpb}, num_iterations, nonfinite=[...], duplicates)."""
    steps, vals, _ = parse_train_log(path)
    loss = {}
    dups = 0
    for s in steps:
        dups += s["step"] in loss
        loss[s["step"]] = s["loss"]  # a duplicated step (e.g. a resumed run) keeps the last occurrence
    val = {v["step"]: v["val_bpb"] for v in vals}
    # second pass: non-finite values parse_train_log cannot see, plus the planned iteration count
    nonfinite, num_iter = [], None
    with open(path, errors="replace") as f:
        for line in f:
            m = LOOSE_STEP_RE.search(line)
            if m:
                num_iter = int(m.group(2))
                x = _float(m.group(3))
                if not math.isfinite(x):
                    loss[int(m.group(1))] = x
                    nonfinite.append(dict(kind="loss", step=int(m.group(1)), value=m.group(3)))
                continue
            m = LOOSE_VAL_RE.search(line)
            if m:
                x = _float(m.group(2))
                if not math.isfinite(x):
                    val[int(m.group(1))] = x
                    nonfinite.append(dict(kind="val_bpb", step=int(m.group(1)), value=m.group(2)))
    return dict(path=os.path.abspath(path), job_id=job_id_from_path(path), loss=dict(sorted(loss.items())),
                val=dict(sorted(val.items())), num_iterations=num_iter, nonfinite=nonfinite, duplicates=dups)


def loss_deltas(a, b):
    """[(step, loss_a, loss_b, loss_a - loss_b)] over the steps > AFTER_STEP present in both curves."""
    common = sorted(set(a["loss"]) & set(b["loss"]))
    return [(s, a["loss"][s], b["loss"][s], a["loss"][s] - b["loss"][s]) for s in common if s > AFTER_STEP]


def val_deltas(a, b):
    common = sorted(set(a["val"]) & set(b["val"]))
    return [dict(step=s, val_bpb_a=a["val"][s], val_bpb_b=b["val"][s], delta=a["val"][s] - b["val"][s])
            for s in common]


def make_envelope(log_a, log_b):
    a, b = load_curve(log_a), load_curve(log_b)
    warnings, invalid = [], []   # invalid: the envelope must not be used for G4 (main exits 1, compare() refuses it)
    for c in (a, b):
        if c["nonfinite"]:
            invalid.append(f"{c['path']}: non-finite values {c['nonfinite'][:5]}")
        if c["duplicates"]:
            warnings.append(f"{c['path']}: {c['duplicates']} duplicated step lines (kept the last)")
    if set(a["loss"]) != set(b["loss"]):
        warnings.append(f"step sets differ ({len(a['loss'])} vs {len(b['loss'])} steps); using the intersection")
    if a["num_iterations"] != b["num_iterations"]:
        invalid.append(f"planned step counts differ ({a['num_iterations']} vs {b['num_iterations']}): not the same run")
    for tag, c in (("A", a), ("B", b)):
        if truncated(c):
            invalid.append(f"run {tag} {truncated(c)}")
    warnings = invalid + warnings
    d = loss_deltas(a, b)
    if not d:
        raise SystemExit(f"no common steps > {AFTER_STEP} in {log_a} and {log_b}; cannot build an envelope")
    finite = [x for x in d if math.isfinite(x[3])]
    if len(finite) < len(d):
        warnings.append(f"{len(d) - len(finite)} compared steps are non-finite; the band ignores them (rerun!)")
    worst = max(finite, key=lambda x: abs(x[3])) if finite else None
    absd = sorted(abs(x[3]) for x in finite)
    vd = val_deltas(a, b)
    final = vd[-1] if vd else None
    if final is None:
        invalid.append("no common 'Validation bpb' step; val_bpb_delta is null")
        warnings.insert(0, invalid[-1])
    band = abs(worst[3]) if worst else None
    vbd = abs(final["delta"]) if final and math.isfinite(final["delta"]) else None
    env = dict(
        band=band, band_step=worst[0] if worst else None,
        val_bpb_delta=vbd, val_bpb_delta_step=final["step"] if final else None,
        steps=len(d), step_range=[d[0][0], d[-1][0]], after_step=AFTER_STEP,
        median_abs_delta=absd[len(absd) // 2] if absd else None,
        loss_tol=max(2 * band, LOSS_TOL_FLOOR) if band is not None else LOSS_TOL_FLOOR,
        val_tol=max(VAL_TOL_FLOOR, 2 * vbd) if vbd is not None else VAL_TOL_FLOOR,
        per_eval=vd, num_iterations=a["num_iterations"],
        final_loss=dict(a=a["loss"][d[-1][0]], b=b["loss"][d[-1][0]]),
        source_logs=[a["path"], b["path"]], job_id=a["job_id"], source_job_ids=[a["job_id"], b["job_id"]],
        rule=G4_RULE, loss_field="base_train debiased EMA(0.9), used as logged",
        created=datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"), warnings=warnings,
        valid=not invalid, invalid_reasons=invalid)
    return env, a, b, d


def truncated(c):
    """Why a log did not run to completion ('' if it did): the last step line must be num_iterations - 1 and, when the
    run evaluated at all, the last eval must be at step num_iterations (base_train always evaluates the final step)."""
    n = c["num_iterations"]
    if not n or not c["loss"]:
        return "has no step lines" if not c["loss"] else ""
    if max(c["loss"]) < n - 1:
        return f"stopped at step {max(c['loss'])} of {n} (truncated log?)"
    if c["val"] and max(c["val"]) != n:
        return f"has no final eval at step {n} (last eval at step {max(c['val'])}; truncated log?)"
    return ""


def compare(stock_log, variant_log, envelope):
    """Gate G4 (PLAN.md 7.3) for one variant against stock. `envelope` is the dict from make_envelope or a path
    to its JSON. Returns a result dict with 'passed' (bool), 'reasons' (why it failed) and the measured deltas."""
    if isinstance(envelope, str):
        with open(envelope) as f:
            envelope = json.load(f)
    s, v = load_curve(stock_log), load_curve(variant_log)
    band, vbd = envelope.get("band"), envelope.get("val_bpb_delta")
    loss_tol = max(2 * band, LOSS_TOL_FLOOR) if band is not None else LOSS_TOL_FLOOR
    val_tol = max(VAL_TOL_FLOOR, 2 * vbd) if vbd is not None else VAL_TOL_FLOOR
    reasons, notes = [], []
    if envelope.get("valid") is False:
        reasons.append(f"envelope is marked invalid: {'; '.join(envelope.get('invalid_reasons', []))}")
    if "band" not in envelope or "val_bpb_delta" not in envelope:
        reasons.append("envelope JSON lacks 'band' / 'val_bpb_delta' (not a noise_envelope.json?)")

    # the stock run must be complete (a truncated stock would shrink the compared window and the final-val step),
    # and both runs must plan the same step count (the LR schedule depends on it)
    if truncated(s):
        reasons.append(f"stock log {truncated(s)}")
    if s["num_iterations"] != v["num_iterations"]:
        reasons.append(f"planned step counts differ: stock {s['num_iterations']}, variant {v['num_iterations']}")

    # no NaN/Inf anywhere in either log (stock too: a broken stock run makes the comparison meaningless)
    for name, c in (("variant", v), ("stock", s)):
        if c["nonfinite"]:
            nf = c["nonfinite"][0]
            reasons.append(f"{name} has {len(c['nonfinite'])} non-finite values (first: {nf['kind']} = {nf['value']} "
                           f"at step {nf['step']})")

    # every stock step > 100 must exist in the variant and be within loss_tol
    stock_steps = [st for st in s["loss"] if st > AFTER_STEP]
    if not stock_steps:
        reasons.append(f"stock log has no steps > {AFTER_STEP}")
    missing = [st for st in stock_steps if st not in v["loss"]]
    if missing:
        reasons.append(f"variant is missing {len(missing)} of {len(stock_steps)} stock steps > {AFTER_STEP} "
                       f"(first missing: {missing[0]}; crashed or truncated?)")
    d = loss_deltas(v, s)  # variant - stock
    finite = [x for x in d if math.isfinite(x[3])]
    worst = max(finite, key=lambda x: abs(x[3])) if finite else None
    over = [x for x in finite if abs(x[3]) > loss_tol]
    if over:
        reasons.append(f"|delta loss| > {loss_tol:.4f} at {len(over)} steps (first: step {over[0][0]}, "
                       f"delta {over[0][3]:+.4f}; worst: step {worst[0]}, delta {worst[3]:+.4f})")

    # final val bpb: the stock run's last eval step must be present in the variant
    vd = val_deltas(v, s)
    final = None
    if not s["val"]:
        reasons.append("stock log has no 'Validation bpb' line")
    else:
        last = max(s["val"])
        final = next((x for x in vd if x["step"] == last), None)
        if final is None:
            reasons.append(f"variant has no val bpb at the stock's final eval step {last}")
        elif not math.isfinite(final["delta"]) or abs(final["delta"]) > val_tol:
            reasons.append(f"final val bpb |delta| {abs(final['delta']):.5f} > {val_tol:.5f} at step {last}")
    if envelope.get("num_iterations") and s["num_iterations"] != envelope["num_iterations"]:
        notes.append(f"stock num_iterations {s['num_iterations']} != envelope's {envelope['num_iterations']}")
    return dict(
        passed=not reasons, gate="G4", reasons=reasons, notes=notes,
        worst_step=worst[0] if worst else None, worst_delta=worst[3] if worst else None,
        loss_tol=loss_tol, steps_compared=len(d),
        final_val_bpb=dict(step=final["step"], stock=final["val_bpb_b"], variant=final["val_bpb_a"],
                           delta=final["delta"]) if final else None,
        val_tol=val_tol, per_eval=[dict(step=x["step"], val_bpb_stock=x["val_bpb_b"], val_bpb_variant=x["val_bpb_a"],
                                        delta=x["delta"]) for x in vd],
        envelope=dict(band=band, val_bpb_delta=vbd, job_id=envelope.get("job_id")),
        stock_log=s["path"], variant_log=v["path"], job_id=v["job_id"], stock_job_id=s["job_id"],
        gates_passed=["G4"] if not reasons else [], rule=G4_RULE,
        created=datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"))


# ---------------------------------------------------------------------------------------------- outputs
def job_copy_path(path, job_id):
    """<name>_<jobid><ext>; the path itself if its name already carries the job id."""
    if job_id and job_id in os.path.basename(path):
        return path
    root, ext = os.path.splitext(path)
    return f"{root}_{job_id or 'nojob'}{ext}"


def write_json(obj, path):
    """Writes `path` and the job-stamped copy next to it. If `path` holds another job's result whose stamped copy
    does not exist yet, that copy is made first, so replacing the canonical file never loses a job's numbers."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    if os.path.exists(path):
        try:
            with open(path) as f:
                old = json.load(f).get("job_id")
        except (json.JSONDecodeError, OSError, AttributeError):
            old = None
        keep = job_copy_path(path, old)
        if old != obj.get("job_id") and not os.path.exists(keep):
            shutil.copy2(path, keep)
            print(f"kept previous result of job {old} as {keep}")
    stamped = job_copy_path(path, obj.get("job_id"))
    for p in (stamped, path) if os.path.abspath(stamped) != os.path.abspath(path) else (path,):
        with open(p, "w") as f:
            json.dump(obj, f, indent=1)
        print("wrote", p)


def plot_curves(a, b, d, band, loss_tol, labels, title, job_id, path):
    """Top: the two loss curves; bottom: their difference with the noise band shaded and the G4 tolerance dashed."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from cisc662.analysis.make_figs_s1 import INK2, MUTED, SERIES, SURFACE, value_grid  # shared slide style

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8.2, 5.6), sharex=True, gridspec_kw=dict(height_ratios=[3, 2]))
    for i, (c, lab) in enumerate(zip((a, b), labels)):
        xs = [st for st, x in c["loss"].items() if math.isfinite(x)]
        ax1.plot(xs, [c["loss"][st] for st in xs], color=SERIES[i], linewidth=1.6 if i == 0 else 1.2,
                 linestyle="-" if i == 0 else (0, (4, 2)), label=lab)
    tail = [x for st, x in list(a["loss"].items()) + list(b["loss"].items()) if st > AFTER_STEP and math.isfinite(x)]
    if tail:  # zoom on the compared window; the first 100 steps fall from ~10 and would flatten the plot
        lo, hi = min(tail), max(tail)
        ax1.set_ylim(lo - 0.05 * (hi - lo + 1e-3), hi + 0.6 * (hi - lo + 1e-3))
    ax1.set_ylabel("train loss (logged EMA)")
    value_grid(ax1, "y")
    ax1.legend(loc="upper right")
    ax1.set_title(title)

    xs = [x[0] for x in d if math.isfinite(x[3])]
    ys = [x[3] for x in d if math.isfinite(x[3])]
    if band is not None:
        ax2.axhspan(-band, band, color=SERIES[2], alpha=0.18, linewidth=0, label=f"stock-stock band ±{band:.4f}")
    for sgn in (1, -1):
        ax2.axhline(sgn * loss_tol, color=MUTED, linewidth=1.0, linestyle=(0, (4, 3)),
                    label=f"G4 tolerance ±{loss_tol:.4f}" if sgn == 1 else None)
    ax2.axhline(0, color=INK2, linewidth=0.6)
    ax2.plot(xs, ys, color=SERIES[0], linewidth=1.2, label=f"{labels[0]} − {labels[1]}")
    ax2.axvline(AFTER_STEP, color=MUTED, linewidth=0.8)
    ax2.set_ylabel("Δ loss")
    ax2.set_xlabel("training step")
    value_grid(ax2, "y")
    ax2.legend(loc="upper right", fontsize=8)
    ax1.axvline(AFTER_STEP, color=MUTED, linewidth=0.8)
    ax1.text(AFTER_STEP, ax1.get_ylim()[1], f" gate window: steps > {AFTER_STEP}", va="top", fontsize=8, color=MUTED)
    fig.text(0.0, -0.02, f"nanochat base_train, {a['num_iterations'] or '?'} steps (R12-L: depth 12, 32k tokens/step, "
             f"bf16), same seed and data order. Loss = the logged debiased EMA(0.9). Job {job_id or 'n/a'}.",
             fontsize=8, color=MUTED)
    fig.tight_layout()
    for p in dict.fromkeys((job_copy_path(path, job_id), path)):  # one save if the name already has the job id
        os.makedirs(os.path.dirname(os.path.abspath(p)), exist_ok=True)
        fig.savefig(p, dpi=200, bbox_inches="tight")
        print("wrote", p)
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("log_a", help="--make-envelope: stock run A; --envelope: the stock run")
    ap.add_argument("log_b", help="--make-envelope: stock run B; --envelope: the variant run")
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--make-envelope", action="store_true", help="build the noise envelope from two stock runs")
    mode.add_argument("--envelope", help="noise_envelope.json to gate the variant against (G4)")
    ap.add_argument("--out", help=f"JSON output (make-envelope default: {DEFAULT_ENVELOPE}; compare: none)")
    ap.add_argument("--fig", help=f"PNG output (make-envelope default: {DEFAULT_FIG}; compare: none)")
    ap.add_argument("--variant", default="variant", help="compare mode: variant name for the figure legend")
    args = ap.parse_args(argv)

    if args.make_envelope:
        env, a, b, d = make_envelope(args.log_a, args.log_b)
        for w in env["warnings"]:
            print("warning:", w)
        write_json(env, args.out or DEFAULT_ENVELOPE)
        plot_curves(a, b, d, env["band"], env["loss_tol"], ("stock run A", "stock run B"),
                    "Run-to-run noise of the training loss: two identical stock runs", env["job_id"],
                    args.fig or DEFAULT_FIG)
        band = f"{env['band']:.6f} at step {env['band_step']}" if env["band"] is not None else "n/a"
        vbd = f"{env['val_bpb_delta']:.6f} at step {env['val_bpb_delta_step']}" if env["val_bpb_delta"] is not None \
            else "n/a"
        print(f"ENVELOPE job {env['job_id']}: band {band} over {env['steps']} steps "
              f"({env['step_range'][0]}-{env['step_range'][1]}); val bpb delta {vbd}; "
              f"G4 tolerances: loss {env['loss_tol']:.4f}, val bpb {env['val_tol']:.4f}")
        if not env["valid"]:
            print("ENVELOPE INVALID (written with valid=false; compare() will refuse it): "
                  + "; ".join(env["invalid_reasons"]))
            return 1
        return 0

    res = compare(args.log_a, args.log_b, args.envelope)
    fv = res["final_val_bpb"]
    detail = (f"worst step {res['worst_step']} (delta loss {res['worst_delta']:+.5f}, tol {res['loss_tol']:.4f}, "
              f"{res['steps_compared']} steps > {AFTER_STEP}); " if res["worst_step"] is not None else "no steps compared; ")
    detail += (f"final val bpb delta {fv['delta']:+.5f} at step {fv['step']} (tol {res['val_tol']:.4f})" if fv
               else "final val bpb n/a")
    print(("PASS" if res["passed"] else "FAIL") + f" G4 {os.path.basename(res['variant_log'])} vs "
          f"{os.path.basename(res['stock_log'])}: {detail}")
    for r in res["reasons"]:
        print("  reason:", r)
    for n in res["notes"]:
        print("  note:", n)
    if args.out:
        write_json(res, args.out)
    if args.fig:
        v, s = load_curve(args.log_b), load_curve(args.log_a)
        plot_curves(v, s, loss_deltas(v, s), res["envelope"]["band"], res["loss_tol"], (args.variant, "stock"),
                    f"G4 loss-curve gate: {args.variant} vs stock — {'PASS' if res['passed'] else 'FAIL'}",
                    res["job_id"], args.fig)
    return 0 if res["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
