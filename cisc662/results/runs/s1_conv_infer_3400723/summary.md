## Training: seconds per 100 steps (R12, steps 10-109)

| mode | attention | reps | s/100 steps | median step (ms) | tok/s | MFU % | peak mem (GiB) |
|---|---|---|---|---|---|---|---|

## Decode latency (decode_bench, batch 1)

| run | device | threads | TTFT ms | TPOT ms | tok/s |
|---|---|---|---|---|---|
| decode_gpu_fa3 | cuda | 1 | 15.9 | 14.70 | 68.2 |

Convergence run: 2000 steps, final smoothed train loss 3.46, val bpb [(0, 3.157534), (250, 1.5512), (500, 1.300724), (750, 1.208559), (1000, 1.151667), (1250, 1.11267), (1500, 1.083761), (1750, 1.062833), (2000, 1.050529)]

## Inference (infer_bench: prompt 512, decode 256)

| attention | rep | prefill tok/s | batch | TPOT ms | tok/s | MBU % |
|---|---|---|---|---|---|---|
| fa3 | 1 | 36,188 | 1 | 11.64 | 83 | 1.8 |
| fa3 | 1 | 36,188 | 8 | 11.48 | 676 | 2.1 |
| fa3 | 1 | 36,188 | 32 | 12.71 | 2,469 | 3.0 |
| fa3 | 1 | 36,188 | 128 | 13.12 | 9,498 | 7.3 |
| fa3 | 2 | 33,744 | 1 | 12.57 | 81 | 1.6 |
| fa3 | 2 | 33,744 | 8 | 10.08 | 762 | 2.4 |
| fa3 | 2 | 33,744 | 32 | 10.42 | 3,011 | 3.7 |
| fa3 | 2 | 33,744 | 128 | 11.50 | 10,942 | 8.3 |
| sdpa | 1 | 43,752 | 1 | 10.15 | 98 | 2.0 |
| sdpa | 1 | 43,752 | 8 | 10.41 | 732 | 2.4 |
| sdpa | 1 | 43,752 | 32 | 11.27 | 2,805 | 3.4 |
| sdpa | 1 | 43,752 | 128 | 12.49 | 10,091 | 7.6 |
| sdpa | 2 | 37,048 | 1 | 12.11 | 82 | 1.7 |
| sdpa | 2 | 37,048 | 8 | 12.31 | 628 | 2.0 |
| sdpa | 2 | 37,048 | 32 | 12.28 | 2,556 | 3.1 |
| sdpa | 2 | 37,048 | 128 | 10.06 | 12,279 | 9.5 |

GPU monitor: {'samples': 639, 'sm_clock_MHz_median_when_busy': 1485.0, 'power_W_median_when_busy': 528.8, 'frac_samples_util_ge_50': 0.5586854460093896}
