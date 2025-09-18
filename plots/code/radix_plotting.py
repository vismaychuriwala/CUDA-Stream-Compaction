import re
from pathlib import Path
from typing import List, Dict, Any, Tuple

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, LogLocator, LogFormatterMathtext

# Paths
BASE_DIR = Path(__file__).resolve().parent  # plots/code
PLOTS_DIR = BASE_DIR.parent                 # plots
DATA_DIR = PLOTS_DIR / "data"              # plots/data

INPUT_PATH = DATA_DIR / "RadixSort_timings.txt"
OUTPUT_ALL_RUNS_CSV = DATA_DIR / "radix_timings_all_runs_long.csv"
OUTPUT_AVG_CSV = DATA_DIR / "radix_timings_avg.csv"
OUTPUT_PLOT_LOGLOG_FULL = PLOTS_DIR / "radix_timings_loglog_full.png"
OUTPUT_PLOT_LOGX_LINEAR_GT = PLOTS_DIR / "radix_timings_log_linear_gt_2pow18.png"

# Data in RadixSort_timings.txt is in milliseconds already
Y_LABEL = "Time (ms)"
MIN_PLOT_SIZE = 2 ** 18  # for the log-x, linear-y plot

# Limit how many runs per size to include in outputs/plots
MAX_RUNS_PER_SIZE = 10

# Methods in order per SIZE block
METHODS = [
    "cpu_pow2",
    "cpu_non_pow2",
    "radixsort_pow2",
    "radixsort_non_pow2",
]
N_PER_RUN = len(METHODS)

# Style
try:
    plt.style.use("seaborn-whitegrid")
except Exception:
    pass


def _is_float_token(tok: str) -> bool:
    return bool(
        re.match(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$", tok.strip())
    )


def parse_radix_timings(
    path: Path,
    max_runs_per_size: int = MAX_RUNS_PER_SIZE,
) -> Tuple[pd.DataFrame, Dict[int, int], Dict[int, int]]:
    """
    Parse RadixSort_timings.txt blocks of the form:
      SIZE= <int>
      <cpu_pow2>
      <cpu_non_pow2>
      <radixsort_pow2>
      <radixsort_non_pow2>

    Returns:
      - DataFrame with columns: size, run, method, time_ms
      - encountered_counts: dict[size] -> total blocks seen
      - used_counts: dict[size] -> blocks included (capped)
    """
    lines = [ln.strip() for ln in path.read_text().splitlines()]

    records: List[Dict[str, Any]] = []
    i = 0
    encountered_counts: Dict[int, int] = {}
    used_counts: Dict[int, int] = {}

    while i < len(lines):
        line = lines[i]
        if not line or line.startswith("**"):
            i += 1
            continue

        if line.upper().startswith("SIZE"):
            m = re.search(r"SIZE\s*=\s*(\d+)", line, flags=re.IGNORECASE)
            if not m:
                raise ValueError(f"Bad SIZE line: {line}")
            size = int(m.group(1))

            # Collect next N_PER_RUN float values
            vals: List[float] = []
            i += 1
            while i < len(lines) and len(vals) < N_PER_RUN:
                ln = lines[i].strip()
                if ln and not ln.upper().startswith("SIZE") and not ln.startswith("**"):
                    for tok in ln.replace(",", " ").split():
                        if _is_float_token(tok):
                            vals.append(float(tok))
                            if len(vals) >= N_PER_RUN:
                                break
                i += 1

            if len(vals) != N_PER_RUN:
                raise ValueError(
                    f"Expected {N_PER_RUN} values for size {size}, got {len(vals)}"
                )

            encountered_counts[size] = encountered_counts.get(size, 0) + 1
            if encountered_counts[size] <= max_runs_per_size:
                used_counts[size] = used_counts.get(size, 0) + 1
                run_no = used_counts[size]
                for idx, t in enumerate(vals):
                    records.append(
                        {
                            "size": size,
                            "run": run_no,
                            "method": METHODS[idx],
                            "time_ms": t,
                        }
                    )
            continue

        i += 1

    df = pd.DataFrame.from_records(records)
    return df, encountered_counts, used_counts


def average_across_runs(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(["size", "method"], as_index=False)["time_ms"].mean()
    )


def label_for_method(method: str, subset: str) -> str:
    base = {
        "cpu_pow2": "CPU",
        "cpu_non_pow2": "CPU",
        "radixsort_pow2": "Radix GPU",
        "radixsort_non_pow2": "Radix GPU",
    }.get(method, method)

    if subset == "all":
        if method.endswith("non_pow2"):
            return f"{base} (non-power-of-two)"
        if method.endswith("pow2"):
            return f"{base} (power-of-two)"
    return base


def plot_full_loglog(avg_df: pd.DataFrame, out_path: Path) -> None:
    sizes = sorted(avg_df["size"].unique())
    fig, ax = plt.subplots(figsize=(7.5, 5.0), dpi=180)

    for m in METHODS:
        y = []
        for s in sizes:
            row = avg_df[(avg_df["size"] == s) & (avg_df["method"] == m)]
            y.append(row["time_ms"].iloc[0] if not row.empty else float("nan"))
        ax.plot(
            sizes,
            y,
            marker="o",
            linewidth=2.0,
            markersize=4,
            label=label_for_method(m, subset="all"),
        )

    # Axes scales and formatting
    try:
        ax.set_xscale("log", base=10)
        ax.set_yscale("log", base=10)
    except TypeError:
        ax.set_xscale("log")
        ax.set_yscale("log")

    ax.xaxis.set_major_locator(LogLocator(base=10.0))
    ax.xaxis.set_major_formatter(LogFormatterMathtext(base=10.0))
    ax.yaxis.set_major_locator(LogLocator(base=10.0))
    ax.yaxis.set_major_formatter(LogFormatterMathtext(base=10.0))

    ax.set_xlabel("Array Size")
    ax.set_ylabel(Y_LABEL + " (log)")
    ax.set_title("Radix sort timings — full range [log-log]")
    ax.legend(fontsize=9)
    # ax.grid(True, which="both", linestyle="--", alpha=0.4)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_logx_linear_gt(avg_df: pd.DataFrame, min_size: int, out_path: Path) -> None:
    df = avg_df[avg_df["size"] > min_size].copy()
    sizes = sorted(df["size"].unique())
    fig, ax = plt.subplots(figsize=(7.5, 5.0), dpi=180)

    for m in METHODS:
        y = []
        for s in sizes:
            row = df[(df["size"] == s) & (df["method"] == m)]
            y.append(row["time_ms"].iloc[0] if not row.empty else float("nan"))
        ax.plot(
            sizes,
            y,
            marker="o",
            linewidth=2.0,
            markersize=4,
            label=label_for_method(m, subset="all"),
        )

    try:
        ax.set_xscale("log", base=10)
    except TypeError:
        ax.set_xscale("log")

    ax.xaxis.set_major_locator(LogLocator(base=10.0))
    ax.xaxis.set_major_formatter(LogFormatterMathtext(base=10.0))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))

    ax.set_xlabel("Array Size")
    ax.set_ylabel(Y_LABEL)
    ax.set_title(f"Radix sort timings — sizes > 256k [log-x]")
    ax.legend(fontsize=9)
    # ax.grid(True, which="both", linestyle="--", alpha=0.4)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    if not INPUT_PATH.exists():
        raise SystemExit(f"Input file not found: {INPUT_PATH.resolve()}")

    df, encountered_counts, used_counts = parse_radix_timings(INPUT_PATH)

    # Save all runs (long format)
    df.to_csv(OUTPUT_ALL_RUNS_CSV, index=False)

    # Average across runs
    avg = average_across_runs(df)
    avg.to_csv(OUTPUT_AVG_CSV, index=False)

    # Plots
    plot_full_loglog(avg, OUTPUT_PLOT_LOGLOG_FULL)
    plot_logx_linear_gt(avg, MIN_PLOT_SIZE, OUTPUT_PLOT_LOGX_LINEAR_GT)

    print(f"Wrote: {OUTPUT_ALL_RUNS_CSV}")
    print(f"Wrote: {OUTPUT_AVG_CSV}")
    print(f"Wrote: {OUTPUT_PLOT_LOGLOG_FULL}")
    print(f"Wrote: {OUTPUT_PLOT_LOGX_LINEAR_GT}")
    print("Runs encountered per size:")
    for size in sorted(encountered_counts.keys()):
        enc = encountered_counts[size]
        used = used_counts.get(size, 0)
        print(f"  size={size}: encountered={enc}, used={used}")


if __name__ == "__main__":
    main()

