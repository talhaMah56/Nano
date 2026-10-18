"""
Device-agnostic decode benchmark (CPU or GPU) for the serial-CPU-reference comparison (proposal goal i).

scripts/infer_bench.py asserts CUDA, so this mirrors its measurement (same Engine code path, same prompt
construction, median time per output token) but runs on any device. Prints one JSON line at the end.

Examples:
  CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 NANOCHAT_DTYPE=float32 python -m cisc662.bench.decode_bench --device cpu -g r12_conv
  python -m cisc662.bench.decode_bench --device cuda -g r12_conv --batch-sizes 1,8
"""
import argparse
import json
import time

import torch

from nanochat.checkpoint_manager import load_model
from nanochat.engine import Engine
from scripts.infer_bench import build_prompt


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def bench(engine, prompt, batch_size, decode_tokens, device):
    gen = engine.generate(prompt, num_samples=batch_size, max_tokens=decode_tokens, temperature=0.0)
    t0 = time.perf_counter()
    next(gen)  # prefill (batch 1) + KV replication + first token
    sync(device)
    ttft = time.perf_counter() - t0
    steps = []
    while True:
        t0 = time.perf_counter()
        try:
            next(gen)
        except StopIteration:
            break
        sync(device)
        steps.append(time.perf_counter() - t0)
    return ttft, steps


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    p.add_argument("-i", "--source", default="base")
    p.add_argument("-g", "--model-tag", required=True)
    p.add_argument("--prompt-tokens", type=int, default=512)
    p.add_argument("--decode-tokens", type=int, default=32)
    p.add_argument("--batch-sizes", default="1")
    args = p.parse_args()

    device = torch.device(args.device)
    model, tokenizer, meta = load_model(args.source, device, phase="eval", model_tag=args.model_tag)
    engine = Engine(model, tokenizer)
    prompt = build_prompt(tokenizer, args.prompt_tokens)
    out = {"device": args.device, "threads": torch.get_num_threads(), "dtype": str(next(model.parameters()).dtype),
           "model_tag": args.model_tag, "step": meta["step"], "prompt_tokens": args.prompt_tokens,
           "decode_tokens": args.decode_tokens, "sweep": []}
    bench(engine, prompt, 1, 3, device)  # warmup
    for bs in [int(b) for b in args.batch_sizes.split(",")]:
        ttft, steps = bench(engine, prompt, bs, args.decode_tokens, device)
        tpot = sorted(steps)[len(steps) // 2] if steps else float("nan")
        row = {"batch_size": bs, "ttft_sec": ttft, "tpot_sec": tpot, "decode_steps": len(steps),
               "tok_per_sec": bs * len(steps) / sum(steps) if steps else 0.0}
        print(row, flush=True)
        out["sweep"].append(row)
    print(json.dumps(out))


if __name__ == "__main__":
    main()
