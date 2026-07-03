#!/usr/bin/env python3
"""Analyze Nsight Systems SQLite exports.

Reads an nsys-exported SQLite database and produces summary tables for:
- Top GPU kernels by total time
- Memory transfer breakdown (H2D, D2H, D2D)
- NVTX region timing
- GPU idle time analysis
- Per-GPU load balance

Usage:
    python nsys_analyze.py profile.sqlite [--top 20] [--plot] [--output-dir ./results]

Generate the SQLite from an nsys-rep:
    nsys export --type=sqlite profile.nsys-rep
"""

import argparse
import sqlite3
import sys
from pathlib import Path

COPY_KINDS = {1: "H2D", 2: "D2H", 8: "D2D", 10: "P2P"}


def analyze_kernels(conn: sqlite3.Connection, top_n: int = 20) -> str:
    """Top GPU kernels by total execution time."""
    try:
        df = _query(
            conn,
            f"""
            SELECT
                COALESCE(s.value, k.shortName) AS kernel,
                COUNT(*) AS count,
                ROUND(SUM(k.end - k.start) / 1e6, 2) AS total_ms,
                ROUND(AVG(k.end - k.start) / 1e6, 3) AS avg_ms,
                ROUND(MIN(k.end - k.start) / 1e6, 3) AS min_ms,
                ROUND(MAX(k.end - k.start) / 1e6, 3) AS max_ms
            FROM CUPTI_ACTIVITY_KIND_KERNEL k
            LEFT JOIN StringIds s ON k.demangledName = s.id
            GROUP BY kernel
            ORDER BY total_ms DESC
            LIMIT {top_n}
        """,
        )
        return f"=== Top {top_n} GPU Kernels by Total Time ===\n{df}\n"
    except Exception as e:
        return f"=== Kernel analysis skipped: {e} ===\n"


def analyze_memcpy(conn: sqlite3.Connection) -> str:
    """Memory transfer summary by direction."""
    try:
        df = _query(
            conn,
            """
            SELECT
                copyKind,
                COUNT(*) AS count,
                ROUND(SUM(bytes) / 1e9, 3) AS total_gb,
                ROUND(SUM(end - start) / 1e6, 2) AS total_ms,
                ROUND(AVG(bytes / NULLIF((end - start) / 1e9, 0)) / 1e9, 2) AS avg_bw_gbps
            FROM CUPTI_ACTIVITY_KIND_MEMCPY
            GROUP BY copyKind
        """,
        )
        # Map copy kinds to human-readable names
        lines = ["=== Memory Transfer Summary ==="]
        for _, row in df.iterrows():
            kind = COPY_KINDS.get(int(row["copyKind"]), f"Unknown({int(row['copyKind'])})")
            lines.append(
                f"  {kind:>4s}: {int(row['count']):>6d} transfers, "
                f"{row['total_gb']:.3f} GB, {row['total_ms']:.1f} ms, "
                f"{row['avg_bw_gbps']:.1f} GB/s avg"
            )
        return "\n".join(lines) + "\n"
    except Exception as e:
        return f"=== Memory analysis skipped: {e} ===\n"


def analyze_nvtx(conn: sqlite3.Connection) -> str:
    """NVTX annotated region timing."""
    try:
        df = _query(
            conn,
            """
            SELECT
                s.value AS region,
                COUNT(*) AS count,
                ROUND(SUM(e.endTimestamp - e.startTimestamp) / 1e6, 2) AS total_ms,
                ROUND(AVG(e.endTimestamp - e.startTimestamp) / 1e6, 3) AS avg_ms
            FROM NVTX_EVENTS e
            JOIN StringIds s ON e.textId = s.id
            WHERE e.eventType = 59
            GROUP BY region
            ORDER BY total_ms DESC
        """,
        )
        if df.empty:
            return "=== No NVTX regions found (add cp.cuda.nvtx.RangePush/Pop to your code) ===\n"
        return f"=== NVTX Region Timing ===\n{df}\n"
    except Exception as e:
        return f"=== NVTX analysis skipped: {e} ===\n"


def analyze_gpu_idle(conn: sqlite3.Connection) -> str:
    """Idle time between GPU kernels per device."""
    try:
        df = _query(
            conn,
            """
            SELECT start, end, deviceId
            FROM CUPTI_ACTIVITY_KIND_KERNEL
            ORDER BY deviceId, start
        """,
        )
        lines = ["=== GPU Idle Time Analysis ==="]
        for dev in sorted(df["deviceId"].unique()):
            dev_df = df[df["deviceId"] == dev].sort_values("start")
            if len(dev_df) < 2:
                continue
            starts = dev_df["start"].values[1:]
            ends = dev_df["end"].values[:-1]
            gaps_ns = starts - ends
            gaps_ms = gaps_ns[gaps_ns > 0] / 1e6
            total_kernel_ms = (dev_df["end"] - dev_df["start"]).sum() / 1e6
            lines.append(
                f"  GPU {dev}: {len(gaps_ms)} gaps, "
                f"idle {gaps_ms.sum():.1f} ms, "
                f"max gap {gaps_ms.max():.1f} ms, "
                f"compute {total_kernel_ms:.1f} ms, "
                f"utilization {total_kernel_ms / (total_kernel_ms + gaps_ms.sum()) * 100:.1f}%"
            )
        return "\n".join(lines) + "\n"
    except Exception as e:
        return f"=== Idle time analysis skipped: {e} ===\n"


def analyze_gpu_balance(conn: sqlite3.Connection) -> str:
    """Per-GPU kernel count and time (detect load imbalance)."""
    try:
        df = _query(
            conn,
            """
            SELECT
                deviceId,
                COUNT(*) AS kernel_count,
                ROUND(SUM(end - start) / 1e6, 1) AS total_ms
            FROM CUPTI_ACTIVITY_KIND_KERNEL
            GROUP BY deviceId
        """,
        )
        if len(df) <= 1:
            return ""
        lines = ["=== Multi-GPU Load Balance ==="]
        for _, row in df.iterrows():
            lines.append(f"  GPU {int(row['deviceId'])}: {int(row['kernel_count'])} kernels, {row['total_ms']} ms")
        max_ms = df["total_ms"].max()
        min_ms = df["total_ms"].min()
        if max_ms > 0:
            imbalance = (max_ms - min_ms) / max_ms * 100
            lines.append(f"  Imbalance: {imbalance:.1f}% (0% = perfect balance)")
        return "\n".join(lines) + "\n"
    except Exception:
        return ""


def plot_results(conn: sqlite3.Connection, output_dir: Path):
    """Generate matplotlib plots from the profile data."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping plots")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    # Kernel duration distribution
    try:
        df = _query(
            conn,
            """
            SELECT (end - start) / 1e3 AS duration_us
            FROM CUPTI_ACTIVITY_KIND_KERNEL
        """,
        )
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.hist(df["duration_us"], bins=100, edgecolor="black", alpha=0.7)
        ax.set_xlabel("Kernel Duration (us)")
        ax.set_ylabel("Count")
        ax.set_title("GPU Kernel Duration Distribution")
        ax.set_yscale("log")
        fig.savefig(output_dir / "kernel_durations.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {output_dir / 'kernel_durations.png'}")
    except Exception as e:
        print(f"  Kernel plot skipped: {e}")

    # GPU timeline (top kernels)
    try:
        df = _query(
            conn,
            """
            SELECT
                COALESCE(s.value, k.shortName) AS kernel,
                ROUND(SUM(k.end - k.start) / 1e6, 1) AS total_ms
            FROM CUPTI_ACTIVITY_KIND_KERNEL k
            LEFT JOIN StringIds s ON k.demangledName = s.id
            GROUP BY kernel
            ORDER BY total_ms DESC
            LIMIT 15
        """,
        )
        fig, ax = plt.subplots(figsize=(12, 5))
        # Truncate long kernel names
        names = [n[:60] + "..." if len(n) > 60 else n for n in df["kernel"]]
        ax.barh(range(len(names)), df["total_ms"], color="steelblue")
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, fontsize=8)
        ax.set_xlabel("Total Time (ms)")
        ax.set_title("Top 15 GPU Kernels by Total Time")
        ax.invert_yaxis()
        fig.savefig(output_dir / "top_kernels.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {output_dir / 'top_kernels.png'}")
    except Exception as e:
        print(f"  Top kernels plot skipped: {e}")


def _query(conn, sql):
    """Execute SQL and return pandas DataFrame."""
    import pandas as pd

    return pd.read_sql_query(sql, conn)


def main():
    parser = argparse.ArgumentParser(description="Analyze Nsight Systems SQLite export")
    parser.add_argument("sqlite_file", help="Path to .sqlite file from nsys export")
    parser.add_argument("--top", type=int, default=20, help="Top N kernels to show")
    parser.add_argument("--plot", action="store_true", help="Generate matplotlib plots")
    parser.add_argument("--output-dir", type=str, default="./nsys_results", help="Directory for plot output")
    args = parser.parse_args()

    path = Path(args.sqlite_file)
    if not path.exists():
        print(f"Error: {path} not found")
        sys.exit(1)

    conn = sqlite3.connect(str(path))

    print(analyze_kernels(conn, args.top))
    print(analyze_memcpy(conn))
    print(analyze_nvtx(conn))
    print(analyze_gpu_idle(conn))
    print(analyze_gpu_balance(conn))

    if args.plot:
        print("Generating plots...")
        plot_results(conn, Path(args.output_dir))

    conn.close()


if __name__ == "__main__":
    main()
