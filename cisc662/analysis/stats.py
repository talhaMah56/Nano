"""
Statistics for kernel microbenchmarks (PLAN.md §7.1): trial summaries and bootstrap speedup CIs.

Timing data is always a list of trials, each trial a list of per-rep times (one process per trial,
see cisc662/bench/kbench.py). The headline statistic is the median of the trial medians; the
speedup of a variant over a baseline is base / variant (> 1 means the variant is faster), and its
95% CI comes from a bootstrap stratified by trial: inside every trial the reps are resampled with
replacement, the trial median is recomputed, and the statistic is the median of those trial medians.
Pure numpy, CPU-only.

Self-test (login node is fine):  nice -n 19 python -m cisc662.analysis.stats --selftest
"""
import argparse
import sys

import numpy as np


def _as_trials(trials):
    """Validates and converts a list of per-trial lists to a list of 1-D float arrays."""
    out = [np.asarray(t, dtype=np.float64).ravel() for t in trials]
    out = [t for t in out if t.size > 0]
    if not out:
        raise ValueError("need at least one non-empty trial")
    return out


def median_of_medians(trials):
    return float(np.median([np.median(t) for t in _as_trials(trials)]))


def summarize_trials(trials):
    """Headline numbers for one case. p10/p90/min are over all reps pooled across trials;
    cv_trials is the sample coefficient of variation of the trial medians (None with one trial)."""
    ts = _as_trials(trials)
    med = np.array([np.median(t) for t in ts])
    pooled = np.concatenate(ts)
    cv = float(np.std(med, ddof=1) / np.mean(med)) if len(med) > 1 else None
    return {
        "median": float(np.median(med)),
        "p10": float(np.percentile(pooled, 10)),
        "p90": float(np.percentile(pooled, 90)),
        "min": float(pooled.min()),
        "trial_medians": [float(m) for m in med],
        "cv_trials": cv,
        "n_trials": len(ts),
        "n_reps": int(pooled.size),
    }


def _boot_stat(trials, n_boot, rng):
    """(n_boot,) bootstrap replicates of the median of trial medians, reps resampled within each trial."""
    per_trial = []
    for t in trials:
        idx = rng.integers(0, t.size, size=(n_boot, t.size))
        per_trial.append(np.median(t[idx], axis=1))
    return np.median(np.stack(per_trial), axis=0)


def bootstrap_speedup_ci(base_trials, var_trials, n_boot=10000, seed=0, ci=0.95):
    """Speedup base/var with a stratified bootstrap CI. Returns (ratio, lo, hi).

    base_trials, var_trials: lists of per-trial lists of rep times (same unit). Base and variant are
    resampled independently (they are different kernels timed in the same processes). The point
    estimate is the ratio of the observed medians of trial medians, not the bootstrap mean.
    Caveat: with 3 trials the between-trial spread enters only through the median over trials; the
    CI mostly reflects within-trial noise. kbench also reports cv_trials so drift is visible.
    """
    b, v = _as_trials(base_trials), _as_trials(var_trials)
    rng = np.random.default_rng(seed)
    ratio = median_of_medians(b) / median_of_medians(v)
    reps = _boot_stat(b, n_boot, rng) / _boot_stat(v, n_boot, rng)
    alpha = (1.0 - ci) / 2.0
    lo, hi = np.quantile(reps, [alpha, 1.0 - alpha])
    return float(ratio), float(lo), float(hi)


def _selftest():
    """Synthetic checks: identical distributions -> CI contains 1; a 1.5x shift -> lo > 1.3."""
    rng = np.random.default_rng(123)

    def fake(n_trials, center, spread=0.02, reps=100):
        # lognormal-ish rep times around a per-trial center that drifts by ~0.5%
        return [list(center * (1 + rng.normal(0, 0.005)) * np.exp(rng.normal(0, spread, reps)))
                for _ in range(n_trials)]

    ok = True
    base = fake(3, 100.0)
    same = fake(3, 100.0)
    r, lo, hi = bootstrap_speedup_ci(base, same)
    passed = lo <= 1.0 <= hi
    ok &= passed
    print(f"identical: ratio={r:.4f} ci=[{lo:.4f}, {hi:.4f}] contains 1: {'PASS' if passed else 'FAIL'}")

    fast = fake(3, 100.0 / 1.5)
    r, lo, hi = bootstrap_speedup_ci(base, fast)
    passed = lo > 1.3 and abs(r - 1.5) < 0.05
    ok &= passed
    print(f"1.5x shift: ratio={r:.4f} ci=[{lo:.4f}, {hi:.4f}] lo>1.3: {'PASS' if passed else 'FAIL'}")

    # a noisy 1.32x shift must not be claimable as >= 1.3x (CI lower bound below 1.3)
    noisy = fake(3, 100.0 / 1.32, spread=0.10)
    r, lo, hi = bootstrap_speedup_ci(fake(3, 100.0, spread=0.10), noisy)
    passed = lo < 1.3 < hi
    ok &= passed
    print(f"noisy 1.32x: ratio={r:.4f} ci=[{lo:.4f}, {hi:.4f}] straddles 1.3: {'PASS' if passed else 'FAIL'}")

    # determinism and single-trial support
    a = bootstrap_speedup_ci(base, fast, seed=7)
    b = bootstrap_speedup_ci(base, fast, seed=7)
    passed = a == b and bootstrap_speedup_ci(base[:1], fast[:1])[1] > 1.3
    ok &= passed
    print(f"deterministic + 1-trial: {'PASS' if passed else 'FAIL'}")

    s = summarize_trials(base)
    passed = s["n_trials"] == 3 and s["n_reps"] == 300 and s["p10"] < s["median"] < s["p90"]
    ok &= passed
    print(f"summarize_trials: median={s['median']:.3f} cv={s['cv_trials']:.4f}: {'PASS' if passed else 'FAIL'}")
    return ok


def main():
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--selftest", action="store_true", help="run the synthetic-data checks")
    args = parser.parse_args()
    if args.selftest:
        sys.exit(0 if _selftest() else 1)
    parser.print_help()


if __name__ == "__main__":
    main()
