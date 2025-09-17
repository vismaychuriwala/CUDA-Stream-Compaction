import re
from pathlib import Path
from typing import List, Dict, Any, Tuple

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, LogLocator, FixedLocator, LogFormatterMathtext

# Configuration (resolve paths relative to this file for robustness)
BASE_DIR = Path(__file__).resolve().parent  # plots/code
PLOTS_DIR = BASE_DIR.parent                 # plots
DATA_DIR = PLOTS_DIR / "data"              # plots/data

INPUT_PATH = DATA_DIR / "timings.txt"
OUTPUT_ALL_RUNS_CSV = DATA_DIR / "timings_all_runs_long.csv"
OUTPUT_AVG_CSV = DATA_DIR / "timings_avg.csv"
OUTPUT_PLOT = DATA_DIR / "timings_plot.png"
OUTPUT_PLOT_POW2 = PLOTS_DIR / "timings_plot_pow2.png"
OUTPUT_PLOT_NONPOW2 = PLOTS_DIR / "timings_plot_nonpow2.png"
OUTPUT_PLOT_BOTH = PLOTS_DIR / "timings_plot_both.png"
OUTPUT_PLOT_BOTH_LOG = PLOTS_DIR / "timings_plot_both_loglog.png"
OUTPUT_PLOT_POW2_LOG = PLOTS_DIR / "timings_plot_pow2_loglog.png"
OUTPUT_PLOT_NONPOW2_LOG = PLOTS_DIR / "timings_plot_nonpow2_loglog.png"
OUTPUT_PLOT_BOTH_LOG_FULL = PLOTS_DIR / "timings_plot_both_loglog_full.png"

# If your numbers are in seconds (very likely), set this to True to convert
# to milliseconds for plotting convenience.
CONVERT_TO_MS = True
Y_LABEL = "Time (ms)" if CONVERT_TO_MS else "Time (s)"
MIN_PLOT_SIZE = 2 ** 18  # start plots from 2^18

# Limit how many runs per size to include in outputs/plots
MAX_RUNS_PER_SIZE = 10

# Expected order in the file (exactly 13 numbers per block):
SCAN_METHODS = [
    "scan_cpu_pow2",
    "scan_cpu_non_pow2",
    "scan_naive_pow2",
    "scan_naive_non_pow2",
    "scan_work_efficient_pow2",
    "scan_work_efficient_non_pow2",
    "scan_thrust_pow2",
    "scan_thrust_non_pow2",
]
COMPACT_METHODS = [
    "compact_cpu_without_scan_pow2",
    "compact_cpu_without_scan_non_pow2",
    "compact_cpu_with_scan",
    "compact_work_efficient_pow2",
    "compact_work_efficient_non_pow2",
]
N_SCAN = len(SCAN_METHODS)
N_COMPACT = len(COMPACT_METHODS)
N_PER_RUN = N_SCAN + N_COMPACT

# Plot style for better readability
try:
    plt.style.use("seaborn-whitegrid")
except Exception:
    pass


def _is_float_token(tok: str) -> bool:
    return bool(
        re.match(
            r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$",
            tok.strip(),
        )
    )


def parse_timings(
    path: Path,
    max_runs_per_size: int = MAX_RUNS_PER_SIZE,
) -> Tuple[pd.DataFrame, Dict[int, int], Dict[int, int]]:
    """
    Parse timings in the simplified format present in timings.txt:
      Optional header lines (e.g., "** SCAN TESTS **")
      SIZE= <int>
      <8 scan times> (possibly one per line; blanks ignored)
      <5 compaction times> (possibly one per line; blanks ignored)
      ... repeated blocks for more runs and/or sizes

    Notes:
      - There are no explicit RUN lines. We infer run numbers per size
        by counting consecutive blocks encountered for that size.

    Returns:
      - DataFrame (long format) with columns: size, run, suite, method, time_s
      - encountered_counts: dict[size] -> total blocks encountered for that size
      - used_counts: dict[size] -> blocks included (capped at max_runs_per_size)
    """
    lines = [ln.strip() for ln in path.read_text().splitlines()]

    records: List[Dict[str, Any]] = []
    i = 0
    # Track how many blocks (runs) we've seen per size
    encountered_counts: Dict[int, int] = {}
    used_counts: Dict[int, int] = {}

    while i < len(lines):
        line = lines[i]
        if not line or line.startswith("**"):
            i += 1
            continue

        # Parse a SIZE block
        if line.upper().startswith("SIZE"):
            m = re.search(r"SIZE\s*=\s*(\d+)", line, flags=re.IGNORECASE)
            if not m:
                raise ValueError(f"Bad SIZE line: {line}")
            size = int(m.group(1))

            # Gather the next N_PER_RUN float values (ignore blanks and headers)
            times: List[float] = []
            i += 1
            while i < len(lines) and len(times) < N_PER_RUN:
                ln = lines[i].strip()
                if ln and not ln.upper().startswith("SIZE") and not ln.startswith("**"):
                    for tok in ln.replace(",", " ").split():
                        if _is_float_token(tok):
                            times.append(float(tok))
                            if len(times) >= N_PER_RUN:
                                break
                i += 1

            if len(times) != N_PER_RUN:
                raise ValueError(
                    f"Expected {N_PER_RUN} times for size {size}, got {len(times)}"
                )

            # Count this block for the size
            encountered_counts[size] = encountered_counts.get(size, 0) + 1

            # Include only the first `max_runs_per_size` blocks per size
            if encountered_counts[size] <= max_runs_per_size:
                used_counts[size] = used_counts.get(size, 0) + 1
                run_no = used_counts[size]

                # Emit records for scan
                for idx, t in enumerate(times[:N_SCAN]):
                    records.append(
                        {
                            "size": size,
                            "run": run_no,
                            "suite": "scan",
                            "method": SCAN_METHODS[idx],
                            "time_s": t,
                        }
                    )

                # Emit records for compaction
                for jdx, t in enumerate(times[N_SCAN:]):
                    records.append(
                        {
                            "size": size,
                            "run": run_no,
                            "suite": "compact",
                            "method": COMPACT_METHODS[jdx],
                            "time_s": t,
                        }
                    )
            continue

        # Skip any other lines
        i += 1

    df = pd.DataFrame.from_records(records)
    return df, encountered_counts, used_counts


def average_across_runs(df: pd.DataFrame) -> pd.DataFrame:
    avg = (
        df.groupby(["size", "suite", "method"], as_index=False)["time_s"]
        .mean()
    )
    return avg


def _is_pow2(x: int) -> bool:
    return x > 0 and (x & (x - 1)) == 0


def _format_size_ticks(sizes):
    # Kept for compatibility; not used when using log10 axis with automatic ticks
    return [f"{s:,}" for s in sizes]


def plot_by_suite(
    avg_df: pd.DataFrame,
    out_path: Path,
    to_ms: bool = True,
    runs_cap: int | None = None,
    log_x: bool = True,
    log_y: bool = False,
    method_subset: str = "all",  # one of: 'all', 'pow2', 'nonpow2'
    min_size: int | None = MIN_PLOT_SIZE,
) -> None:
    # Helper: map internal method keys to plain-English legend labels
    def label_for_method(method: str, suite: str, subset: str) -> str:
        # Determine base label by method family
        base = method
        if suite == "scan":
            if method.startswith("scan_cpu"):
                base = "CPU"
            elif method.startswith("scan_naive"):
                base = "Naive GPU"
            elif method.startswith("scan_work_efficient"):
                base = "Work-efficient GPU"
            elif method.startswith("scan_thrust"):
                base = "Thrust"
            # Append operation for clarity
            base = f"{base} scan"
        else:  # compact
            if method == "compact_cpu_with_scan":
                base = "CPU (with scan)"
            elif method.startswith("compact_cpu_without_scan"):
                base = "CPU (no scan)"
            elif method.startswith("compact_work_efficient"):
                base = "Work-efficient GPU"
            else:
                base = method.replace("_", " ")
            base = f"{base}"

        # Determine suffix only when plotting both pow2 and nonpow2
        if subset == "all":
            # Important: check the non-power-of-two suffix first because
            # strings like "non_pow2" also end with "_pow2".
            if method.endswith("_non_pow2"):
                return f"{base} (non-power-of-two)"
            if method.endswith("_pow2"):
                return f"{base} (power-of-two)"
        # In pow2-only or nonpow2-only plots, suffix is redundant
        return base

    # Prepare values
    plot_df = avg_df.copy()
    if to_ms:
        plot_df["time_val"] = plot_df["time_s"] * 1000.0
    else:
        plot_df["time_val"] = plot_df["time_s"]

    # Optional filter on minimum size for plotting (strictly greater than)
    if min_size is not None:
        plot_df = plot_df[plot_df["size"] > min_size]

    # Use all sizes; split by method family (pow2/nonpow2)
    sizes = sorted(plot_df["size"].unique())

    # Keep method ordering consistent with the lists above
    def select_methods(all_methods: List[str]) -> List[str]:
        if method_subset == "pow2":
            sel = [m for m in all_methods if m.endswith("_pow2")]
            # Include methods without explicit suffix (apply to both)
            sel += [m for m in all_methods if ("_pow2" not in m and "_non_pow2" not in m)]
            return [m for m in all_methods if m in sel]
        elif method_subset == "nonpow2":
            sel = [m for m in all_methods if m.endswith("_non_pow2")]
            sel += [m for m in all_methods if ("_pow2" not in m and "_non_pow2" not in m)]
            return [m for m in all_methods if m in sel]
        else:
            return all_methods

    scan_methods = select_methods(SCAN_METHODS)
    compact_methods = select_methods(COMPACT_METHODS)

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(13, 5.5), dpi=180)
    plt.tight_layout(pad=2.0)

    # Scan subplot
    ax = axes[0]
    scan_df = plot_df[plot_df["suite"] == "scan"]
    for m in scan_methods:
        y = []
        for s in sizes:
            row = scan_df[(scan_df["size"] == s) & (scan_df["method"] == m)]
            y.append(row["time_val"].iloc[0] if not row.empty else float("nan"))
        ax.plot(
            sizes,
            y,
            marker="o",
            linewidth=2.0,
            markersize=4,
            label=label_for_method(m, "scan", method_subset),
        )
    ax.set_title("Scan")
    ax.set_xlabel("Array Size")
    ax.set_ylabel(Y_LABEL + (" (log)" if log_y else ""))
    if log_x:
        try:
            ax.set_xscale("log", base=10)
        except TypeError:
            ax.set_xscale("log")
    if log_y:
        try:
            ax.set_yscale("log", base=10)
        except TypeError:
            ax.set_yscale("log")
    # Prefer 10** major ticks for x; don't label every point
    if log_x:
        ax.xaxis.set_major_locator(LogLocator(base=10.0))
        ax.xaxis.set_major_formatter(LogFormatterMathtext(base=10.0))
    # Thousands separators for y values (linear)
    if not log_y:
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
    if log_y:
        ax.yaxis.set_major_locator(LogLocator(base=10.0))
        ax.yaxis.set_major_formatter(LogFormatterMathtext(base=10.0))
    if min_size is not None and len(sizes) > 0:
        ax.set_xlim(left=min_size)
    ax.legend(fontsize=8, ncol=2, loc="upper left", bbox_to_anchor=(0, 1.02))
    ax.grid(True, which="both", linestyle="--", alpha=0.4)

    # Compaction subplot
    ax = axes[1]
    comp_df = plot_df[plot_df["suite"] == "compact"]
    for m in compact_methods:
        y = []
        for s in sizes:
            row = comp_df[(comp_df["size"] == s) & (comp_df["method"] == m)]
            y.append(row["time_val"].iloc[0] if not row.empty else float("nan"))
        ax.plot(
            sizes,
            y,
            marker="o",
            linewidth=2.0,
            markersize=4,
            label=label_for_method(m, "compact", method_subset),
        )
    ax.set_title("Stream compaction")
    ax.set_xlabel("Array Size")
    ax.set_ylabel(Y_LABEL + (" (log)" if log_y else ""))
    if log_x:
        try:
            ax.set_xscale("log", base=10)
        except TypeError:
            ax.set_xscale("log")
    if log_y:
        try:
            ax.set_yscale("log", base=10)
        except TypeError:
            ax.set_yscale("log")
    if log_x:
        ax.xaxis.set_major_locator(LogLocator(base=10.0))
        ax.xaxis.set_major_formatter(LogFormatterMathtext(base=10.0))
    if not log_y:
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
    if log_y:
        ax.yaxis.set_major_locator(LogLocator(base=10.0))
        ax.yaxis.set_major_formatter(LogFormatterMathtext(base=10.0))
    if min_size is not None and len(sizes) > 0:
        ax.set_xlim(left=min_size)
    ax.legend(fontsize=8, ncol=2, loc="upper left", bbox_to_anchor=(0, 1.02))
    ax.grid(True, which="both", linestyle="--", alpha=0.4)

    # Build a clear, accurate title segment about the size subset.
    # If a minimum size filter excludes smaller inputs, reflect that using
    # compact binary units (e.g., "256k" instead of 2^18).
    overall_min_size = int(avg_df["size"].min()) if not avg_df.empty else None
    def _fmt_min(s: int) -> str:
        if s is None:
            return ""
        # Prefer binary multiples: k, M, G
        for unit, factor in (("k", 1024), ("M", 1024**2), ("G", 1024**3)):
            if s % factor == 0 and s >= factor:
                val = s // factor
                # Keep it simple (e.g., 256k, 1M, 4G)
                return f"{val}{unit}"
        return f"{s:,}"

    filtered = (
        min_size is not None
        and overall_min_size is not None
        and min_size > overall_min_size
    )
    base_subset = {
        "all": "All sizes" if not filtered else "Sizes",
        "pow2": "Power-of-two sizes",
        "nonpow2": "Non-power-of-two sizes",
    }.get(method_subset, "All sizes" if not filtered else "Sizes")
    subset_note = (
        f"{base_subset} > {_fmt_min(min_size)}" if filtered else base_subset
    )
    y_unit = "ms" if to_ms else "s"
    if log_x and log_y:
        scale_note = " [log-x, log-y]"
    elif log_x:
        scale_note = " [log-x]"
    elif log_y:
        scale_note = " [log-y]"
    else:
        scale_note = ""
    # English title with clarity about metric
    # Example: "Average time (lower is better) — Power-of-two sizes [log-x]"
    fig.suptitle(
        f"Average time (lower is better) — {subset_note}{scale_note}\n"
        f"{runs_cap} runs",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    if not INPUT_PATH.exists():
        raise SystemExit(
            f"Input file not found: {INPUT_PATH.resolve()}"
        )

    df, encountered_counts, used_counts = parse_timings(INPUT_PATH)
    # Save all runs (long format)
    df_out = df.copy()
    df_out["time_ms"] = df_out["time_s"] * 1000.0
    df_out.to_csv(OUTPUT_ALL_RUNS_CSV, index=False)

    # Average across runs
    avg = average_across_runs(df)
    avg_out = avg.copy()
    avg_out["time_ms"] = avg_out["time_s"] * 1000.0
    avg_out.to_csv(OUTPUT_AVG_CSV, index=False)

    # Plot
    # Plots: both, pow2-only, nonpow2-only
    plot_by_suite(
        avg,
        OUTPUT_PLOT_BOTH,
        to_ms=CONVERT_TO_MS,
        runs_cap=MAX_RUNS_PER_SIZE,
        log_x=True,
        log_y=False,
        method_subset="all",
        min_size=MIN_PLOT_SIZE,
    )
    plot_by_suite(
        avg,
        OUTPUT_PLOT_POW2,
        to_ms=CONVERT_TO_MS,
        runs_cap=MAX_RUNS_PER_SIZE,
        log_x=True,
        log_y=False,
        method_subset="pow2",
        min_size=MIN_PLOT_SIZE,
    )
    plot_by_suite(
        avg,
        OUTPUT_PLOT_NONPOW2,
        to_ms=CONVERT_TO_MS,
        runs_cap=MAX_RUNS_PER_SIZE,
        log_x=True,
        log_y=False,
        method_subset="nonpow2",
        min_size=MIN_PLOT_SIZE,
    )

    # Log-Log versions (separate files)
    plot_by_suite(
        avg,
        OUTPUT_PLOT_BOTH_LOG,
        to_ms=CONVERT_TO_MS,
        runs_cap=MAX_RUNS_PER_SIZE,
        log_x=True,
        log_y=True,
        method_subset="all",
        min_size=MIN_PLOT_SIZE,
    )
    plot_by_suite(
        avg,
        OUTPUT_PLOT_POW2_LOG,
        to_ms=CONVERT_TO_MS,
        runs_cap=MAX_RUNS_PER_SIZE,
        log_x=True,
        log_y=True,
        method_subset="pow2",
        min_size=MIN_PLOT_SIZE,
    )
    plot_by_suite(
        avg,
        OUTPUT_PLOT_NONPOW2_LOG,
        to_ms=CONVERT_TO_MS,
        runs_cap=MAX_RUNS_PER_SIZE,
        log_x=True,
        log_y=True,
        method_subset="nonpow2",
        min_size=MIN_PLOT_SIZE,
    )

    # Log-Log with full range (no min size) for both
    plot_by_suite(
        avg,
        OUTPUT_PLOT_BOTH_LOG_FULL,
        to_ms=CONVERT_TO_MS,
        runs_cap=MAX_RUNS_PER_SIZE,
        log_x=True,
        log_y=True,
        method_subset="all",
        min_size=None,
    )

    print(f"Wrote: {OUTPUT_ALL_RUNS_CSV}")
    print(f"Wrote: {OUTPUT_AVG_CSV}")
    print(f"Wrote: {OUTPUT_PLOT_BOTH}")
    print(f"Wrote: {OUTPUT_PLOT_POW2}")
    print(f"Wrote: {OUTPUT_PLOT_NONPOW2}")
    print(f"Wrote: {OUTPUT_PLOT_BOTH_LOG}")
    print(f"Wrote: {OUTPUT_PLOT_POW2_LOG}")
    print(f"Wrote: {OUTPUT_PLOT_NONPOW2_LOG}")
    print(f"Wrote: {OUTPUT_PLOT_BOTH_LOG_FULL}")
    # Print encountered vs used runs per size
    print("Runs encountered per size:")
    for size in sorted(encountered_counts.keys()):
        enc = encountered_counts[size]
        used = used_counts.get(size, 0)
        print(f"  size={size}: encountered={enc}, used={used}")


if __name__ == "__main__":
    main()
