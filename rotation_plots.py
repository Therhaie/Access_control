"""
rotation_plots.py
=================
All visualisations for the rotation experiment.

Reads
-----
  results/rotation_results.json   — produced by rotation_experiment.py
  results/rotation_registry.json  — produced by rotation_experiment.py

Plots produced
--------------
1.  sim_four_way          Per-chunk box-plots of the four cosine-similarity
                          measurements (a/b/c/d) across all queries.

2.  sim_delta_histogram   Histogram of Δ = sim(orig_q, rot_chunk) − sim(orig_q, orig_chunk)
                          Shows the distribution of the alignment penalty from rotating only the chunk.

3.  orthogonality_check   sim(orig_q, orig_chunk) vs sim(rot_q, rot_chunk) scatter.
                          Perfect orthogonal rotation → points lie on y=x line.

4.  topk_overlap          Bar chart: fraction of original top-K that survive in
                          rotated top-K, per query.

5.  cross_query_delta     For the cross-query experiment: distribution of
                          Δ = sim(rot_fq, rot_chunk) − sim(fq, rot_chunk)
                          "Did the rotation accidentally pull a foreign query closer?"

6.  cross_query_heatmap   Heatmap: for each (foreign_query, chunk_group) pair,
                          colour = similarity delta.  Reveals which rotations
                          are most "leaky" to unrelated queries.

7.  group_rotation_impact Per-group mean similarity delta bar chart, sorted.
                          Identifies which (triplet, doc) groups are most
                          disrupted by their own rotation.

Usage
-----
python rotation_plots.py                         # all plots
python rotation_plots.py --plot sim_delta_histogram
python rotation_plots.py --output my_plots/
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

try:
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    import numpy as np
    import pandas as pd
except ImportError as exc:
    print(f"Missing dependency: {exc}")
    print("pip install matplotlib numpy pandas")
    sys.exit(1)

RESULTS_DIR      = Path("results")
ROTATION_RESULTS = RESULTS_DIR / "rotation_results.json"
PLOTS_DIR        = Path("plots")

# Colour palette
C_ORIG  = "#4C72B0"   # blue  — original
C_ROT   = "#DD8452"   # orange — rotated
C_BOTH  = "#55A868"   # green  — both rotated
C_CROSS = "#C44E52"   # red    — cross-query


# ═══════════════════════════════════════════════════════════════════════════════
# Loader
# ═══════════════════════════════════════════════════════════════════════════════

def load_results() -> tuple[list[dict], pd.DataFrame, pd.DataFrame]:
    """
    Returns
    -------
    raw          : raw list of query experiment dicts
    df_chunks    : flat DataFrame, one row per (query, chunk) similarity record
    df_cross     : flat DataFrame, one row per cross-query result
    """
    if not ROTATION_RESULTS.exists():
        print(f"No results file found at {ROTATION_RESULTS}. Run rotation_experiment.py first.")
        sys.exit(1)

    with open(ROTATION_RESULTS, encoding="utf-8") as fh:
        raw: list[dict] = json.load(fh)

    chunk_rows = []
    cross_rows = []
    for qr in raw:
        qid = qr["query_id"]
        q   = qr["question"]
        for cs in qr.get("chunk_similarities", []):
            chunk_rows.append({
                "query_id":   qid,
                "question":   q[:60],
                "chunk_key":  cs["chunk_key"],
                "group_key":  cs["group_key"],
                "seed":       cs["rotation_seed"],
                # four similarity values
                "sim_aa":     cs["sim_orig_query_orig_chunk"],
                "sim_bb":     cs["sim_rot_query_rot_chunk"],
                "sim_ab":     cs["sim_orig_query_rot_chunk"],
                "sim_ba":     cs["sim_rot_query_orig_chunk"],
                # deltas
                "delta_c":    cs["sim_orig_query_rot_chunk"] - cs["sim_orig_query_orig_chunk"],
                "delta_d":    cs["sim_rot_query_orig_chunk"] - cs["sim_orig_query_orig_chunk"],
                "orth_error": cs["sim_rot_query_rot_chunk"]  - cs["sim_orig_query_orig_chunk"],
            })
        for cr in qr.get("cross_query_results", []):
            cross_rows.append({
                "query_id":        qid,
                "foreign_query_id":cr["foreign_query_id"],
                "foreign_question":cr["foreign_question"][:60],
                "group_key":       cr["group_key"],
                "chunk_key":       cr["chunk_key"],
                "sim_before":      cr["sim_foreign_orig_vs_rot_chunk"],
                "sim_after":       cr["sim_foreign_rot_vs_rot_chunk"],
                "delta":           cr["sim_foreign_rot_vs_rot_chunk"] - cr["sim_foreign_orig_vs_rot_chunk"],
            })

    df_chunks = pd.DataFrame(chunk_rows)
    df_cross  = pd.DataFrame(cross_rows)

    # Also build per-query overlap table
    for qr in raw:
        qr["overlap_fraction"] = qr.get("overlap_fraction", 0.0)

    return raw, df_chunks, df_cross


# ═══════════════════════════════════════════════════════════════════════════════
# Individual plots
# ═══════════════════════════════════════════════════════════════════════════════

def plot_sim_four_way(df: pd.DataFrame, out_dir: Path) -> Path:
    """
    Box-plots for all four similarity measurements side by side.
    Visually answers: "how different are the four conditions?"
    """
    data = [
        df["sim_aa"].dropna().values,
        df["sim_bb"].dropna().values,
        df["sim_ab"].dropna().values,
        df["sim_ba"].dropna().values,
    ]
    labels = [
        "( a )\norig_q ↔ orig_c\n(baseline)",
        "( b )\nrot_q ↔ rot_c\n(both rotated)",
        "( c )\norig_q ↔ rot_c\n(chunk rotated)",
        "( d )\nrot_q ↔ orig_c\n(query rotated)",
    ]
    colors = [C_ORIG, C_BOTH, C_ROT, C_CROSS]

    fig, ax = plt.subplots(figsize=(10, 5))
    bp = ax.boxplot(data, patch_artist=True, widths=0.5,
                    medianprops=dict(color="black", linewidth=2))
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)

    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Cosine Similarity")
    ax.set_title("Four-Way Similarity Comparison\n"
                 "(b) should equal (a) if rotation is perfectly orthogonal; "
                 "(c) and (d) show alignment penalty")
    ax.grid(axis="y", alpha=0.3)
    ax.axhline(df["sim_aa"].mean(), color=C_ORIG, linestyle="--",
               alpha=0.6, label=f"baseline mean {df['sim_aa'].mean():.3f}")
    ax.legend(fontsize=8)

    out = out_dir / "sim_four_way.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  ✔  {out}")
    return out


def plot_sim_delta_histogram(df: pd.DataFrame, out_dir: Path) -> Path:
    """
    Histogram of delta_c = sim(orig_q, rot_chunk) − sim(orig_q, orig_chunk).
    The bulk of this distribution should be negative (rotation hurts alignment).
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    for ax, col, color, label in [
        (axes[0], "delta_c", C_ROT,
         "Δ(c−a): only chunk rotated\nsim(orig_q, rot_c) − sim(orig_q, orig_c)"),
        (axes[1], "delta_d", C_CROSS,
         "Δ(d−a): only query rotated\nsim(rot_q, orig_c) − sim(orig_q, orig_c)"),
    ]:
        vals = df[col].dropna()
        ax.hist(vals, bins=40, color=color, edgecolor="white", alpha=0.85)
        ax.axvline(0,          color="black", linewidth=1.5, linestyle="-")
        ax.axvline(vals.mean(), color="black", linewidth=1.5, linestyle="--",
                   label=f"mean={vals.mean():+.4f}")
        ax.set_xlabel("Δ cosine similarity")
        ax.set_ylabel("Count")
        ax.set_title(label)
        ax.legend(fontsize=9)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Similarity Delta Distributions — Impact of Rotating Only One Side",
                 fontsize=11)
    out = out_dir / "sim_delta_histogram.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  ✔  {out}")
    return out


def plot_orthogonality_check(df: pd.DataFrame, out_dir: Path) -> Path:
    """
    Scatter: sim(orig_q, orig_chunk)  vs  sim(rot_q, rot_chunk).
    Perfect orthogonality → points lie on y = x.
    Spread quantifies floating-point deviation.
    """
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(df["sim_aa"], df["sim_bb"], alpha=0.35, s=18, color=C_BOTH)
    lim = [min(df["sim_aa"].min(), df["sim_bb"].min()) - 0.02,
           max(df["sim_aa"].max(), df["sim_bb"].max()) + 0.02]
    ax.plot(lim, lim, "k--", linewidth=1, label="y = x (perfect orthogonality)")
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("sim(orig_q, orig_chunk)  — baseline")
    ax.set_ylabel("sim(rot_q, rot_chunk)  — both rotated")
    ax.set_title("Orthogonality Check\nPoints on y=x mean rotation preserves cosine similarity exactly")
    ax.legend(fontsize=9)

    err = (df["sim_bb"] - df["sim_aa"]).abs()
    ax.text(0.05, 0.92, f"Mean |error| = {err.mean():.2e}\nMax |error| = {err.max():.2e}",
            transform=ax.transAxes, fontsize=9,
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    out = out_dir / "orthogonality_check.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  ✔  {out}")
    return out


def plot_topk_overlap(raw: list[dict], out_dir: Path) -> Path:
    """
    Bar chart: per-query overlap fraction between original and rotated top-K.
    """
    query_ids  = [r["query_id"]        for r in raw]
    overlaps   = [r.get("overlap_fraction", 0) for r in raw]
    short_ids  = [qid[-12:] for qid in query_ids]

    fig, ax = plt.subplots(figsize=(max(8, len(raw) * 0.7), 4))
    bars = ax.bar(short_ids, overlaps, color=C_ORIG, alpha=0.8)
    ax.axhline(np.mean(overlaps), color="red", linestyle="--",
               label=f"mean = {np.mean(overlaps):.2%}")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Overlap fraction")
    ax.set_xlabel("Query")
    ax.set_title(f"Top-K Retrieval Overlap: Original vs Rotated Collection")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.xticks(rotation=45, ha="right", fontsize=8)

    out = out_dir / "topk_overlap.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  ✔  {out}")
    return out


def plot_cross_query_delta(df_cross: pd.DataFrame, out_dir: Path) -> Path | None:
    """
    Distribution of similarity delta for foreign queries against rotated chunks.
    Positive delta = rotation accidentally pulled a foreign query closer (bad).
    Negative delta = rotation pushed foreign query further (neutral/good).
    """
    if df_cross.empty:
        print("  ⚠  No cross-query data — skipping cross_query_delta.")
        return None

    fig, ax = plt.subplots(figsize=(9, 4))
    vals = df_cross["delta"].dropna()
    ax.hist(vals, bins=40, color=C_CROSS, edgecolor="white", alpha=0.85)
    ax.axvline(0,          color="black", linewidth=1.5)
    ax.axvline(vals.mean(), color="black", linewidth=1.5, linestyle="--",
               label=f"mean={vals.mean():+.4f}")
    pct_positive = (vals > 0).mean() * 100
    ax.text(0.65, 0.88,
            f"{pct_positive:.1f}% of cases the rotation\npulled the foreign query CLOSER",
            transform=ax.transAxes, fontsize=9,
            bbox=dict(boxstyle="round", facecolor="#ffcccc", alpha=0.7))
    ax.set_xlabel("Δ sim(rot_foreign_q, rot_chunk) − sim(foreign_q, rot_chunk)")
    ax.set_ylabel("Count")
    ax.set_title("Cross-Query Experiment\n"
                 "Does rotating a query with a 'foreign' rotation change its proximity to those chunks?")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    out = out_dir / "cross_query_delta.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  ✔  {out}")
    return out


def plot_cross_query_heatmap(df_cross: pd.DataFrame, out_dir: Path) -> Path | None:
    """
    Heatmap: rows = foreign query IDs, cols = group_keys.
    Cell = mean delta for that (foreign_query, group) pair.
    Reveals which rotation groups are most "leaky".
    """
    if df_cross.empty:
        print("  ⚠  No cross-query data — skipping cross_query_heatmap.")
        return None

    pivot = (
        df_cross.groupby(["foreign_query_id", "group_key"])["delta"]
        .mean()
        .unstack(fill_value=0.0)
    )

    fig, ax = plt.subplots(figsize=(max(8, pivot.shape[1] * 0.6),
                                    max(4, pivot.shape[0] * 0.5)))
    cmap = plt.cm.RdBu_r
    im   = ax.imshow(pivot.values, aspect="auto", cmap=cmap,
                     vmin=-abs(pivot.values).max(),
                     vmax= abs(pivot.values).max())
    plt.colorbar(im, ax=ax, label="Mean Δ similarity")

    ax.set_xticks(range(pivot.shape[1]))
    ax.set_xticklabels(pivot.columns, rotation=60, ha="right", fontsize=7)
    ax.set_yticks(range(pivot.shape[0]))
    ax.set_yticklabels(pivot.index, fontsize=8)
    ax.set_title("Cross-Query Rotation Leakage Heatmap\n"
                 "Red = foreign query pulled closer (bad), Blue = pushed further (good)")

    out = out_dir / "cross_query_heatmap.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✔  {out}")
    return out


def plot_group_rotation_impact(df: pd.DataFrame, out_dir: Path) -> Path:
    """
    Per rotation-group: mean delta_c (how much cosine drops when only the chunk
    is rotated).  Sorted ascending — leftmost groups are most disrupted.
    """
    group_delta = (
        df.groupby("group_key")["delta_c"]
        .agg(["mean", "std", "count"])
        .rename(columns={"mean": "mean_delta", "std": "std_delta", "count": "n"})
        .sort_values("mean_delta")
    )

    fig, ax = plt.subplots(figsize=(max(10, len(group_delta) * 0.5), 5))
    x = range(len(group_delta))
    ax.bar(x, group_delta["mean_delta"], color=C_ROT, alpha=0.8,
           yerr=group_delta["std_delta"], capsize=3, error_kw={"linewidth": 0.8})
    ax.axhline(0, color="black", linewidth=1)
    ax.axhline(group_delta["mean_delta"].mean(), color="black", linewidth=1.5,
               linestyle="--", label=f"global mean Δ = {group_delta['mean_delta'].mean():+.4f}")
    ax.set_xticks(list(x))
    ax.set_xticklabels(group_delta.index, rotation=60, ha="right", fontsize=7)
    ax.set_ylabel("Mean Δ cosine similarity (orig_q ↔ rot_chunk − baseline)")
    ax.set_title("Per-Group Rotation Impact\n"
                 "How much each group's rotation disrupts alignment with its own query")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    out = out_dir / "group_rotation_impact.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✔  {out}")
    return out


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

ALL_PLOTS = [
    "sim_four_way",
    "sim_delta_histogram",
    "orthogonality_check",
    "topk_overlap",
    "cross_query_delta",
    "cross_query_heatmap",
    "group_rotation_impact",
]


def main(plot: str = "all", output: str = "plots") -> None:
    out_dir = Path(output)
    out_dir.mkdir(parents=True, exist_ok=True)

    raw, df_chunks, df_cross = load_results()
    wanted = ALL_PLOTS if plot == "all" else [plot]

    print(f"\nGenerating {len(wanted)} plot(s) → {out_dir}/\n")

    if "sim_four_way"          in wanted: plot_sim_four_way(df_chunks, out_dir)
    if "sim_delta_histogram"   in wanted: plot_sim_delta_histogram(df_chunks, out_dir)
    if "orthogonality_check"   in wanted: plot_orthogonality_check(df_chunks, out_dir)
    if "topk_overlap"          in wanted: plot_topk_overlap(raw, out_dir)
    if "cross_query_delta"     in wanted: plot_cross_query_delta(df_cross, out_dir)
    if "cross_query_heatmap"   in wanted: plot_cross_query_heatmap(df_cross, out_dir)
    if "group_rotation_impact" in wanted: plot_group_rotation_impact(df_chunks, out_dir)

    print(f"\nAll done — plots saved to {out_dir}/\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot rotation experiment results")
    parser.add_argument("--plot",   "-p", default="all",
                        choices=["all"] + ALL_PLOTS)
    parser.add_argument("--output", "-o", default="plots")
    args = parser.parse_args()
    main(plot=args.plot, output=args.output)
