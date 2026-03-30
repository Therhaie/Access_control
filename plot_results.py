"""
plot_results.py
===============
Visualise evaluation runs stored in  results/eval_runs.jsonl
and per-sample data in              results/eval_samples.jsonl

Usage
-----
python plot_results.py                          # all plots
python plot_results.py --plot trace_trend       # single plot
python plot_results.py --output plots/          # custom output dir

Plots produced
--------------
1. trace_trend       — TRACe metric means across runs (line chart)
2. metric_radar      — Radar / spider chart for the latest run
3. timing_breakdown  — Stacked bar: retrieve / rerank / generate per run
4. reranker_compare  — Side-by-side bar: reranker ON vs OFF (if both exist)
5. f1_distribution   — Histogram of per-sample answer F1 scores
6. sample_heatmap    — Per-question pass/fail heatmap for TRACe metrics

All figures are saved as PNG files and also displayed interactively
if a display is available.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# ── Optional deps guard ───────────────────────────────────────────────────────
try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import numpy as np
    import pandas as pd
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Install with:  pip install matplotlib numpy pandas")
    sys.exit(1)

RESULTS_DIR  = Path("results")
RUNS_FILE    = RESULTS_DIR / "eval_runs.jsonl"
SAMPLES_FILE = RESULTS_DIR / "eval_samples.jsonl"

TRACE_COLS = [
    "context_relevance_mean",
    "answer_faithfulness_mean",
    "context_utilization_mean",
    "answer_completeness_mean",
]
TRACE_LABELS = ["Context\nRelevance", "Answer\nFaithfulness",
                "Context\nUtilization", "Answer\nCompleteness"]
TRACE_COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]

SAMPLE_TRACE_COLS = [
    "context_relevance",
    "answer_faithfulness",
    "context_utilization",
    "answer_completeness",
]
SAMPLE_TRACE_LABELS = ["Context\nRelevance", "Answer\nFaithfulness",
                       "Context\nUtilization", "Answer\nCompleteness"]


# ── Loaders ───────────────────────────────────────────────────────────────────

def load_runs() -> pd.DataFrame:
    if not RUNS_FILE.exists():
        print(f"No runs file found at {RUNS_FILE}. Run trace_eval.py first.")
        sys.exit(1)
    rows = [json.loads(line) for line in RUNS_FILE.read_text().splitlines() if line.strip()]
    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    df["run_label"] = df["run_id"].str[:8] + "\n" + df["timestamp"].dt.strftime("%m-%d %H:%M")
    return df


def load_samples() -> pd.DataFrame:
    if not SAMPLES_FILE.exists():
        print(f"No samples file found at {SAMPLES_FILE}.")
        return pd.DataFrame()
    rows = [json.loads(line) for line in SAMPLES_FILE.read_text().splitlines() if line.strip()]
    return pd.DataFrame(rows)


# ── Individual plots ──────────────────────────────────────────────────────────

def plot_trace_trend(df: pd.DataFrame, out_dir: Path) -> Path:
    """Line chart of each TRACe metric mean across runs."""
    fig, ax = plt.subplots(figsize=(max(8, len(df) * 1.2), 5))
    x = range(len(df))
    for col, label, color in zip(TRACE_COLS, TRACE_LABELS, TRACE_COLORS):
        vals = pd.to_numeric(df[col], errors="coerce")
        ax.plot(x, vals, marker="o", label=label.replace("\n", " "), color=color, linewidth=2)

    if "trace_mean" in df.columns:
        vals = pd.to_numeric(df["trace_mean"], errors="coerce")
        ax.plot(x, vals, marker="s", label="TRACe mean", color="black",
                linewidth=2.5, linestyle="--")

    ax.set_xticks(list(x))
    ax.set_xticklabels(df["run_label"], fontsize=8)
    ax.set_ylim(-0.05, 1.05)
    ax.set_ylabel("Score (mean over samples)")
    ax.set_title("TRACe Metrics Across Evaluation Runs")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.2f}"))

    out = out_dir / "trace_trend.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  ✔  {out}")
    return out


def plot_metric_radar(df: pd.DataFrame, out_dir: Path) -> Path:
    """Radar chart for the most recent run."""
    latest = df.iloc[-1]
    values = [pd.to_numeric(latest[c], errors="coerce") or 0.0 for c in TRACE_COLS]
    labels = [l.replace("\n", " ") for l in TRACE_LABELS]

    N = len(labels)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]
    values += values[:1]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0.25", "0.50", "0.75", "1.00"], fontsize=8)
    ax.plot(angles, values, "o-", linewidth=2, color="#4C72B0")
    ax.fill(angles, values, alpha=0.25, color="#4C72B0")
    ax.set_title(
        f"TRACe Radar — run {latest['run_id'][:8]}\n"
        f"{'Reranker ON' if latest.get('use_reranker') else 'Reranker OFF'}",
        fontsize=12, pad=20,
    )

    out = out_dir / "metric_radar.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  ✔  {out}")
    return out


def plot_timing_breakdown(df: pd.DataFrame, out_dir: Path) -> Path:
    """Stacked bar chart: retrieve / rerank / generate time per run."""
    fig, ax = plt.subplots(figsize=(max(8, len(df) * 1.2), 5))
    x     = np.arange(len(df))
    width = 0.5

    retrieve = pd.to_numeric(df["t_retrieve_mean"], errors="coerce").fillna(0)
    rerank   = pd.to_numeric(df["t_rerank_mean"],   errors="coerce").fillna(0)
    generate = pd.to_numeric(df["t_generate_mean"], errors="coerce").fillna(0)

    b1 = ax.bar(x, retrieve, width, label="Retrieve",  color="#4C72B0")
    b2 = ax.bar(x, rerank,   width, bottom=retrieve,   label="Rerank",   color="#DD8452")
    b3 = ax.bar(x, generate, width, bottom=retrieve+rerank, label="Generate", color="#55A868")

    ax.set_xticks(x)
    ax.set_xticklabels(df["run_label"], fontsize=8)
    ax.set_ylabel("Mean latency (seconds)")
    ax.set_title("Pipeline Latency Breakdown per Run")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    out = out_dir / "timing_breakdown.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  ✔  {out}")
    return out


def plot_reranker_compare(df: pd.DataFrame, out_dir: Path) -> Path | None:
    """Side-by-side bar comparing reranker ON vs OFF.  Skipped if only one condition."""
    if "use_reranker" not in df.columns:
        return None
    on  = df[df["use_reranker"] == True]
    off = df[df["use_reranker"] == False]
    if on.empty or off.empty:
        print("  ⚠  Reranker comparison skipped — need runs with both ON and OFF.")
        return None

    # Use means across all runs for each condition
    def means(sub: pd.DataFrame) -> list[float]:
        return [pd.to_numeric(sub[c], errors="coerce").mean() for c in TRACE_COLS]

    on_vals  = means(on)
    off_vals = means(off)

    x     = np.arange(len(TRACE_LABELS))
    width = 0.35
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - width/2, on_vals,  width, label="Reranker ON",  color="#4C72B0")
    ax.bar(x + width/2, off_vals, width, label="Reranker OFF", color="#DD8452")
    ax.set_xticks(x)
    ax.set_xticklabels([l.replace("\n", " ") for l in TRACE_LABELS])
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Mean score")
    ax.set_title("TRACe Metrics: Reranker ON vs OFF")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    out = out_dir / "reranker_compare.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  ✔  {out}")
    return out


def plot_f1_distribution(samples: pd.DataFrame, out_dir: Path) -> Path | None:
    """Histogram of per-sample answer F1 (token overlap vs ground truth)."""
    if samples.empty or "answer_correctness_f1" not in samples.columns:
        return None

    fig, ax = plt.subplots(figsize=(8, 4))
    vals = pd.to_numeric(samples["answer_correctness_f1"], errors="coerce").dropna()
    ax.hist(vals, bins=20, color="#4C72B0", edgecolor="white", alpha=0.85)
    ax.axvline(vals.mean(), color="red", linestyle="--", label=f"Mean {vals.mean():.3f}")
    ax.set_xlabel("Answer F1 (token overlap vs ground truth)")
    ax.set_ylabel("Number of samples")
    ax.set_title("Distribution of Answer Correctness F1")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    out = out_dir / "f1_distribution.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  ✔  {out}")
    return out


def plot_sample_heatmap(samples: pd.DataFrame, out_dir: Path, max_samples: int = 50) -> Path | None:
    """
    Heatmap of pass(1) / fail(0) for each TRACe metric, one row per sample.
    Capped at max_samples rows to stay readable.
    """
    if samples.empty:
        return None

    # Use only the most recent run if many samples
    latest_run = samples["run_id"].iloc[-1] if "run_id" in samples.columns else None
    if latest_run:
        sub = samples[samples["run_id"] == latest_run].copy()
    else:
        sub = samples.copy()

    sub = sub.head(max_samples)
    data = sub[SAMPLE_TRACE_COLS].apply(pd.to_numeric, errors="coerce")

    fig, ax = plt.subplots(figsize=(7, max(4, len(sub) * 0.25)))
    cmap = plt.cm.RdYlGn   # red=fail, green=pass
    im = ax.imshow(data.values, aspect="auto", cmap=cmap, vmin=0, vmax=1,
                   interpolation="nearest")

    ax.set_xticks(range(len(SAMPLE_TRACE_COLS)))
    ax.set_xticklabels([l.replace("\n", " ") for l in SAMPLE_TRACE_LABELS], fontsize=9)
    ax.set_yticks(range(len(sub)))
    ax.set_yticklabels(
        [q[:40] + "…" if len(str(q)) > 40 else str(q)
         for q in sub["question"].values],
        fontsize=7,
    )
    ax.set_title(
        f"Per-sample TRACe Scores — run {(latest_run or '')[:8]}\n"
        f"(green=pass, red=fail, grey=judge error)",
        fontsize=10,
    )
    plt.colorbar(im, ax=ax, fraction=0.03, pad=0.04)

    out = out_dir / "sample_heatmap.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✔  {out}")
    return out


# ── Main ──────────────────────────────────────────────────────────────────────

ALL_PLOTS = [
    "trace_trend",
    "metric_radar",
    "timing_breakdown",
    "reranker_compare",
    "f1_distribution",
    "sample_heatmap",
]


def main(plot: str = "all", output: str = "plots") -> None:
    out_dir = Path(output)
    out_dir.mkdir(parents=True, exist_ok=True)

    df      = load_runs()
    samples = load_samples()

    wanted = ALL_PLOTS if plot == "all" else [plot]
    print(f"\nGenerating {len(wanted)} plot(s) → {out_dir}/\n")

    if "trace_trend"      in wanted:  plot_trace_trend(df, out_dir)
    if "metric_radar"     in wanted:  plot_metric_radar(df, out_dir)
    if "timing_breakdown" in wanted:  plot_timing_breakdown(df, out_dir)
    if "reranker_compare" in wanted:  plot_reranker_compare(df, out_dir)
    if "f1_distribution"  in wanted:  plot_f1_distribution(samples, out_dir)
    if "sample_heatmap"   in wanted:  plot_sample_heatmap(samples, out_dir)

    print(f"\nAll done.  Open {out_dir}/ to view the plots.\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot TRACE evaluation results")
    parser.add_argument(
        "--plot", "-p",
        default="all",
        choices=["all"] + ALL_PLOTS,
        help="Which plot to generate (default: all)",
    )
    parser.add_argument(
        "--output", "-o",
        default="plots",
        help="Output directory for PNG files (default: plots/)",
    )
    args = parser.parse_args()
    main(plot=args.plot, output=args.output)
