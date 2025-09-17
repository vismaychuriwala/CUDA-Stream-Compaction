#!/usr/bin/env python3
import argparse
from collections import defaultdict
import math
import sys

import matplotlib
matplotlib.use("Agg")  # Ensure non-interactive backend for headless runs
import matplotlib.pyplot as plt


def parse_timings(path: str) -> dict[int, list[float]]:
    data: dict[int, list[float]] = defaultdict(list)
    try:
        with open(path, "r", encoding="utf-8") as f:
            lines = [ln.strip() for ln in f if ln.strip()]
    except FileNotFoundError:
        print(f"Input file not found: {path}", file=sys.stderr)
        sys.exit(1)

    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("blockSize"):
            # Expect format like: "blockSize= 32"
            try:
                _, rhs = line.split("=", 1)
                block = int(rhs.strip())
            except Exception:
                i += 1
                continue
            # Next non-empty line should be the timing
            if i + 1 < len(lines):
                try:
                    t = float(lines[i + 1])
                    data[block].append(t)
                except Exception:
                    pass
                i += 2
                continue
        i += 1

    if not data:
        print("No timings parsed. Please check input format.", file=sys.stderr)
        sys.exit(1)
    return data


def compute_averages(data: dict[int, list[float]]):
    blocks = sorted(data.keys())
    avgs = [sum(data[b]) / len(data[b]) for b in blocks]
    return blocks, avgs


def plot(blocks, avgs, output_path: str | None):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(blocks, avgs, marker="o", linestyle="-", color="#1f77b4")
    ax.set_xlabel("Block Size")
    ax.set_ylabel("Average Time (ms)")
    ax.set_title("Block Size vs Average Timing")
    # Use logarithmic scale for block size (base 2 preferred)
    try:
        ax.set_xscale("log", base=2)
    except TypeError:
        # Older Matplotlib versions use 'basex'
        ax.set_xscale("log", basex=2)
    ax.grid(True, which="both", linestyle=":", linewidth=0.6, alpha=0.7)
    ax.set_xticks(blocks)
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.get_xaxis().set_minor_formatter(matplotlib.ticker.NullFormatter())
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=200)
    else:
        # Default filename
        plt.savefig("blocksize_avg.png", dpi=200)


def main():
    parser = argparse.ArgumentParser(description="Plot block size vs average timings (log-scaled x-axis).")
    parser.add_argument("--input", "-i", default="blockSize_timings.txt", help="Path to timings input file.")
    parser.add_argument("--output", "-o", default="blocksize_avg.png", help="Output PNG path for the plot.")
    args = parser.parse_args()

    data = parse_timings(args.input)
    blocks, avgs = compute_averages(data)

    # Print summary to stdout
    print("Averages (blockSize -> avg time):")
    for b, a in zip(blocks, avgs):
        print(f"{b} -> {a:.5f}")

    plot(blocks, avgs, args.output)
    print(f"Saved plot to: {args.output}")


if __name__ == "__main__":
    main()

