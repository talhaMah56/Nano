## Training: seconds per 100 steps (R12, steps 10-109)

| mode | attention | reps | s/100 steps | median step (ms) | tok/s | MFU % | peak mem (GiB) |
|---|---|---|---|---|---|---|---|
| b1-compile | fa3 | 1 | 3.019 ± 0.000 | 29.48 | 68,290 | 6.1 | 3.7 |
| b1-eager | fa3 | 1 | 5.574 ± 0.000 | 54.93 | 36,823 | 3.3 | 3.9 |

## CPU reference (fp32 eager, batch 1 x 2048)

| run | timed steps | median step (s) | extrapolated s/100 steps | tok/s |
|---|---|---|---|---|
| cpu_t1 | 3 | 39.86 | 3985.9 | 51 |
| cpu_t72 | 5 | 2.27 | 226.9 | 902 |

## Decode latency (decode_bench, batch 1)

| run | device | threads | TTFT ms | TPOT ms | tok/s |
|---|---|---|---|---|---|
| decode_cpu_t1 | cpu | 1 | 1564.9 | 30.60 | 32.8 |
| decode_cpu_t72 | cpu | 72 | 67.3 | 10.82 | 89.2 |
| decode_gpu | cuda | 1 | 10.1 | 9.34 | 107.2 |
