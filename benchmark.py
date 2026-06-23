#!/usr/bin/env python3
"""Benchmark: Python (PyTorch) vs C++ (GGML) inference speed."""

import subprocess
import time
import os
import argparse

# Test prompts
PROMPTS = [
    "<|caption|> A dog barking loudly",
    "<|caption|> Birds chirping in a forest <|env|> morning ambience",
    "<|caption|> Thunder and rain <|music|> dramatic orchestral",
]

STEPS = 25
CFG = 3.0


def benchmark_python(prompt, steps, cfg, device="cpu"):
    """Benchmark Python/PyTorch implementation."""
    import torch
    from transformers import AutoModel

    # Load model (cached after first load)
    cache_key = f'model_{device}'
    if not hasattr(benchmark_python, cache_key):
        print(f"Loading Python model on {device}...")
        model = AutoModel.from_pretrained(
            "mispeech/Dasheng-AudioGen",
            trust_remote_code=True
        )
        model.eval()

        if device == "mps" and torch.backends.mps.is_available():
            model = model.to("mps")
        elif device == "cuda" and torch.cuda.is_available():
            model = model.cuda()
        # else stays on CPU

        setattr(benchmark_python, cache_key, model)

    model = getattr(benchmark_python, cache_key)

    # Warmup
    warmup_key = f'warmed_up_{device}'
    if not hasattr(benchmark_python, warmup_key):
        print(f"Warming up Python model ({device})...")
        with torch.no_grad():
            _ = model.generate(prompt, num_steps=5, guidance_scale=cfg)
        setattr(benchmark_python, warmup_key, True)

    # Benchmark
    if device == "cuda":
        torch.cuda.synchronize()

    start = time.perf_counter()

    with torch.no_grad():
        audio = model.generate(prompt, num_steps=steps, guidance_scale=cfg)

    if device == "cuda":
        torch.cuda.synchronize()

    elapsed = time.perf_counter() - start

    return elapsed, audio.shape[-1] if hasattr(audio, 'shape') else len(audio)


def benchmark_cpp(prompt, steps, cfg, binary="build-metal/dasheng-audiogen"):
    """Benchmark C++/GGML implementation."""
    cmd = [
        binary,
        "models/t5_encoder.gguf",
        "models/dit.gguf",
        "models/vocoder.gguf",
        "models/spiece.model",
        "--prompt", prompt,
        "--steps", str(steps),
        "--cfg", str(cfg),
        "--output", "/tmp/benchmark_output.wav"
    ]

    start = time.perf_counter()
    result = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.perf_counter() - start

    if result.returncode != 0:
        print(f"C++ error: {result.stderr}")
        return None, 0

    # Parse sample count from output
    samples = 160000  # default
    for line in result.stderr.split('\n'):
        if 'generated' in line and 'samples' in line:
            try:
                samples = int(line.split()[2])
            except:
                pass

    return elapsed, samples


def main():
    parser = argparse.ArgumentParser(description='Benchmark Python vs C++')
    parser.add_argument('--device', choices=['cpu', 'mps', 'cuda'], default='cpu',
                        help='Device for Python model (default: cpu for fair comparison)')
    parser.add_argument('--steps', type=int, default=25, help='Diffusion steps')
    parser.add_argument('--prompts', type=int, default=3, help='Number of prompts to test')
    args = parser.parse_args()

    print("=" * 70)
    print("Dasheng-AudioGen Benchmark: Python vs C++")
    print("=" * 70)
    print(f"Steps: {args.steps}, CFG: {CFG}, Python device: {args.device}")
    print()

    # Check if C++ binary exists
    cpp_binary = "build-cpu/dasheng-audiogen"
    if not os.path.exists(cpp_binary):
        cpp_binary = "build-metal/dasheng-audiogen"
    if not os.path.exists(cpp_binary):
        print("Error: No C++ binary found. Run 'make metal' or 'make cpu' first.")
        return

    print(f"C++ binary: {cpp_binary}")
    print()

    prompts = PROMPTS[:args.prompts]

    # Warmup C++
    print("Warming up C++ implementation...")
    benchmark_cpp(prompts[0], 5, CFG, cpp_binary)

    results = []

    for i, prompt in enumerate(prompts):
        print(f"\n[Prompt {i+1}/{len(prompts)}] {prompt[:50]}...")

        # Python benchmark
        print("  Python: ", end="", flush=True)
        try:
            py_time, py_samples = benchmark_python(prompt, args.steps, CFG, args.device)
            print(f"{py_time:.2f}s ({py_samples} samples)")
        except Exception as e:
            print(f"Error: {e}")
            py_time, py_samples = None, 0

        # C++ benchmark
        print("  C++:    ", end="", flush=True)
        cpp_time, cpp_samples = benchmark_cpp(prompt, args.steps, CFG, cpp_binary)
        if cpp_time:
            print(f"{cpp_time:.2f}s ({cpp_samples} samples)")

        if py_time and cpp_time:
            speedup = py_time / cpp_time
            results.append({
                'prompt': prompt[:40],
                'python': py_time,
                'cpp': cpp_time,
                'speedup': speedup
            })

    # Summary
    if results:
        print("\n" + "=" * 70)
        print("SUMMARY")
        print("=" * 70)
        print(f"{'Prompt':<42} {'Python':>10} {'C++':>10} {'Speedup':>8}")
        print("-" * 70)

        total_py = 0
        total_cpp = 0
        for r in results:
            print(f"{r['prompt']:<42} {r['python']:>9.2f}s {r['cpp']:>9.2f}s {r['speedup']:>7.2f}x")
            total_py += r['python']
            total_cpp += r['cpp']

        print("-" * 70)
        avg_speedup = total_py / total_cpp if total_cpp > 0 else 0
        print(f"{'TOTAL':<42} {total_py:>9.2f}s {total_cpp:>9.2f}s {avg_speedup:>7.2f}x")
        print()

        if avg_speedup > 1:
            print(f"Result: C++ is {avg_speedup:.1f}x FASTER than Python ({args.device})")
        else:
            print(f"Result: Python ({args.device}) is {1/avg_speedup:.1f}x faster than C++")

        # Per-step timing
        avg_py_per_step = (total_py / len(results)) / args.steps
        avg_cpp_per_step = (total_cpp / len(results)) / args.steps
        print(f"\nPer-step average: Python={avg_py_per_step*1000:.0f}ms, C++={avg_cpp_per_step*1000:.0f}ms")


if __name__ == "__main__":
    main()
