#!/usr/bin/env python3
"""Bayesian parameter tuning for hilum-local-llm-engine.

Usage:
    pip install scikit-optimize
    python scripts/tune_device.py --model path/to/model.gguf --bench-bin ./build/bin/llama-bench

Outputs: tune_results_<device>.json
"""

import argparse
import json
import os
import platform
import re
import subprocess
import sys
from pathlib import Path

try:
    from skopt import gp_minimize
    from skopt.space import Categorical, Integer
    from skopt.utils import use_named_args
except ImportError:
    sys.exit("scikit-optimize is required: pip install scikit-optimize")


def get_device_name():
    node = platform.node() or "unknown"
    return re.sub(r"[^\w\-.]", "_", node)


def parse_tokens_per_sec(output: str) -> float:
    """Extract tokens/sec from llama-bench output.

    llama-bench outputs a markdown table where the last column is t/s.
    We take the best (highest) value found.
    """
    best = 0.0
    for line in output.splitlines():
        # Match lines with numeric last column (t/s)
        parts = [p.strip() for p in line.split("|") if p.strip()]
        if not parts:
            continue
        try:
            val = float(parts[-1])
            if val > best:
                best = val
        except (ValueError, IndexError):
            continue
    return best


def run_bench(bench_bin, model, n_threads, n_batch, n_gpu_layers, flash_attn,
              extra_args=None, repetitions=1):
    """Run llama-bench and return tokens/sec."""
    cmd = [
        str(bench_bin),
        "-m", str(model),
        "-t", str(n_threads),
        "-b", str(n_batch),
        "-ngl", str(n_gpu_layers),
        "-r", str(repetitions),
    ]
    if flash_attn:
        cmd.extend(["-fa", "1"])
    if extra_args:
        cmd.extend(extra_args)

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=300
        )
    except subprocess.TimeoutExpired:
        return 0.0
    except FileNotFoundError:
        sys.exit(f"Benchmark binary not found: {bench_bin}")

    tps = parse_tokens_per_sec(result.stdout)
    if tps == 0.0 and result.returncode != 0:
        print(f"  bench failed (exit {result.returncode}): {result.stderr[:200]}", file=sys.stderr)
    return tps


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="Path to GGUF model")
    parser.add_argument("--bench-bin", default="./build/bin/llama-bench",
                        help="Path to llama-bench binary")
    parser.add_argument("--n-iter", type=int, default=40,
                        help="Number of Bayesian optimization iterations (default: 40)")
    parser.add_argument("--max-threads", type=int,
                        default=os.cpu_count() or 4,
                        help="Max threads to try (default: CPU count)")
    parser.add_argument("--max-ngl", type=int, default=99,
                        help="Max GPU layers to try (default: 99)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON path (default: tune_results_<device>.json)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from checkpoint if available")
    parser.add_argument("--extra-args", nargs="*", default=None,
                        help="Extra args to pass to llama-bench")
    args = parser.parse_args()

    model = Path(args.model)
    if not model.exists():
        sys.exit(f"Model not found: {model}")

    bench_bin = Path(args.bench_bin)
    if not bench_bin.exists():
        sys.exit(f"Benchmark binary not found: {bench_bin}")

    device_name = get_device_name()
    output_path = Path(args.output) if args.output else Path(f"tune_results_{device_name}.json")
    checkpoint_path = output_path.with_suffix(".checkpoint.json")

    # Define search space
    space = [
        Integer(1, args.max_threads, name="n_threads"),
        Categorical([32, 64, 128, 256, 512], name="n_batch"),
        Integer(0, args.max_ngl, name="n_gpu_layers"),
        Categorical([False, True], name="flash_attn"),
    ]

    best_tps = 0.0
    best_params = {}
    iteration = [0]

    # Load checkpoint if resuming
    x0 = None
    y0 = None
    if args.resume and checkpoint_path.exists():
        with open(checkpoint_path) as f:
            ckpt = json.load(f)
        x0 = ckpt.get("x_iters", [])
        y0 = ckpt.get("y_iters", [])
        if x0 and y0:
            print(f"Resuming from checkpoint with {len(x0)} previous iterations")
        else:
            x0, y0 = None, None

    @use_named_args(space)
    def objective(n_threads, n_batch, n_gpu_layers, flash_attn):
        nonlocal best_tps, best_params
        iteration[0] += 1
        print(f"[{iteration[0]}/{args.n_iter}] threads={n_threads} batch={n_batch} "
              f"ngl={n_gpu_layers} fa={flash_attn}", end=" ... ", flush=True)

        tps = run_bench(bench_bin, model, n_threads, n_batch, n_gpu_layers,
                        flash_attn, extra_args=args.extra_args)
        print(f"{tps:.2f} t/s")

        if tps > best_tps:
            best_tps = tps
            best_params = {
                "n_threads": int(n_threads),
                "n_batch": int(n_batch),
                "n_gpu_layers": int(n_gpu_layers),
                "flash_attn": bool(flash_attn),
                "tokens_per_sec": tps,
            }

        return -tps  # minimize negative = maximize

    print(f"Tuning for device: {device_name}")
    print(f"Model: {model}")
    print(f"Iterations: {args.n_iter}")
    print()

    gp_kwargs = dict(
        func=objective,
        dimensions=space,
        n_calls=args.n_iter,
        random_state=42,
        verbose=False,
    )
    if x0 is not None and y0 is not None:
        gp_kwargs["x0"] = x0
        gp_kwargs["y0"] = y0

    result = gp_minimize(**gp_kwargs)

    # Save checkpoint
    checkpoint_data = {
        "x_iters": [list(x) for x in result.x_iters],
        "y_iters": [float(y) for y in result.func_vals],
    }
    with open(checkpoint_path, "w") as f:
        json.dump(checkpoint_data, f)

    # Build final output from best observed (not GP prediction)
    output = {
        "device": device_name,
        "model": str(model),
        "best_params": best_params,
        "iterations": args.n_iter,
    }

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print()
    print(f"Best: {best_tps:.2f} t/s")
    print(f"  n_threads:    {best_params['n_threads']}")
    print(f"  n_batch:      {best_params['n_batch']}")
    print(f"  n_gpu_layers: {best_params['n_gpu_layers']}")
    print(f"  flash_attn:   {best_params['flash_attn']}")
    print(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()
