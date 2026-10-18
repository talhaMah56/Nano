## Training: seconds per 100 steps (R12, steps 10-109)

| mode | attention | reps | s/100 steps | median step (ms) | tok/s | MFU % | peak mem (GiB) |
|---|---|---|---|---|---|---|---|
| compile | fa3 | 3 | 8.046 ± 0.031 | 80.47 | 407,257 | 36.5 | 14.8 |
| compile | sdpa | 3 | 9.114 ± 0.015 | 91.19 | 359,545 | 32.2 | 14.8 |
| eager | fa3 | 3 | 15.080 ± 0.018 | 150.82 | 217,305 | 19.5 | 29.4 |
| eager | sdpa | 3 | 15.983 ± 0.008 | 159.93 | 205,022 | 18.4 | 29.4 |

GPU monitor: {'samples': 639, 'sm_clock_MHz_median_when_busy': 1665.0, 'power_W_median_when_busy': 524.45, 'frac_samples_util_ge_50': 0.5054773082942097}
