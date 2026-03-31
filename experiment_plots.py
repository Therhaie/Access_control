"""
experiment_plots.py
===================
All visualisations for:
  A.  Extra-dimension security experiment   (dim_results.json)
  B.  TRACE conditions comparison           (trace_conditions_results.json)
  C.  Retrieval latency benchmark           (latency_benchmark.json / latency_summary.json)

Plots produced
--------------
Dim experiment
  d1  security_delta_vs_weight   — mean security delta vs weight scalar, per normalisation combo
  d2  auth_vs_unauth_scatter     — scatter: sim(auth) vs sim(unauth) per restricted chunk
  d3  restricted_recovery_bar    — bar: % restricted chunks appearing in auth vs unauth top-K
  d4  collateral_damage          — how much does augmentation change open-chunk similarity?
  d5  normalize_comparison       — box: baseline vs norm_before vs norm_after vs both

TRACE conditions
  t1  trace_radar_all_conditions — radar chart with one polygon per condition
  t2  trace_bar_grouped          — grouped bar: 4 TRACe metrics × 4 conditions
  t3  trace_answer_f1_bar        — answer F1 per condition
  t4  trace_retrieve_time_bar    — mean retrieval time per condition

Latency benchmark
  l1  latency_boxplot            — box-plot of retrieval latency per strategy
  l2  latency_breakdown          — stacked bar: embed + retrieve time
  l3  latency_cdf                — CDF of retrieval latencies per strategy
  l4  latency_overhead_ratio     — ratio of each strategy's latency vs baseline

Usage
-----
python experiment_plots.py                       # all plots
python experiment_plots.py --group dim           # only dim plots
python experiment_plots.py --plot d1 t2 l1
python experiment_plots.py --output my_plots/
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import numpy as np
    import pandas as pd
except ImportError as exc:
    print(f"Missing dependency: {exc}")
    print("pip install matplotlib numpy pandas")
    sys.exit(1)

RESULTS_DIR = Path("results")

# Colour scheme (consistent across all plots)
COND_COLORS = {
    "baseline":           "#4C72B0",
    "rotation":           "#DD8452",
    "extra_dim":          "#55A868",
    "metadata_filter":    "#C44E52",
    "metadata_prefilter": "#C44E52",
    "metadata_postfilter":"#8172B2",
}
TRACE_COLORS  = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]
TRACE_METRICS = ["context_relevance_mean", "answer_faithfulness_mean",
                 "context_utilization_mean", "answer_completeness_mean"]
TRACE_LABELS  = ["Context\nRelevance", "Answer\nFaithfulness",
                 "Context\nUtilization", "Answer\nCompleteness"]


# ═══════════════════════════════════════════════════════════════════════════════
# Loaders
# ═══════════════════════════════════════════════════════════════════════════════

def _load_json(path: Path) -> list | dict | None:
    if not path.exists():
        print(f"  ⚠  {path} not found — skipping related plots.")
        return None
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def load_dim_results() -> pd.DataFrame | None:
    raw = _load_json(RESULTS_DIR / "dim_results.json")
    if raw is None:
        return None
    rows = []
    for qr in raw:
        for cs in qr.get("chunk_sims", []):
            rows.append({
                "query_id":            qr["query_id"],
                "config_id":           qr["config_id"],
                "chunk_key":           cs["chunk_key"],
                "is_restricted":       cs["is_restricted"],
                "sim_baseline":        cs["sim_baseline"],
                "sim_auth_restricted": cs["sim_auth_restricted"],
                "sim_unauth_restricted":cs["sim_unauth_restricted"],
                "sim_auth_open":       cs["sim_auth_open"],
                "sim_unauth_open":     cs["sim_unauth_open"],
                "security_delta":      cs["sim_unauth_restricted"] - cs["sim_auth_restricted"],
                "collateral_delta":    cs["sim_auth_open"] - cs["sim_baseline"],
            })
        rows_topk = {
            "query_id":           qr["query_id"],
            "config_id":          qr["config_id"],
            "restricted_in_auth": qr.get("restricted_in_auth",   0),
            "restricted_in_unauth":qr.get("restricted_in_unauth", 0),
            "n_restricted":       qr.get("n_restricted",          0),
        }
    return pd.DataFrame(rows)


def load_dim_topk() -> pd.DataFrame | None:
    raw = _load_json(RESULTS_DIR / "dim_results.json")
    if raw is None:
        return None
    rows = [
        {
            "query_id":            r["query_id"],
            "config_id":           r["config_id"],
            "restricted_in_auth":  r.get("restricted_in_auth",   0),
            "restricted_in_unauth":r.get("restricted_in_unauth", 0),
            "n_restricted":        r.get("n_restricted",          1),
        }
        for r in raw
    ]
    df = pd.DataFrame(rows)
    df["pct_auth"]   = df["restricted_in_auth"]   / df["n_restricted"].clip(lower=1)
    df["pct_unauth"] = df["restricted_in_unauth"] / df["n_restricted"].clip(lower=1)
    return df


def load_trace_conditions() -> tuple[pd.DataFrame | None, dict | None]:
    raw     = _load_json(RESULTS_DIR / "trace_conditions_results.json")
    summary = _load_json(RESULTS_DIR / "trace_conditions_summary.json")
    if raw is None:
        return None, summary
    rows = []
    for s in raw:
        for cond, cr in s.get("conditions", {}).items():
            t = cr.get("trace", {})
            rows.append({
                "question":                s["question"][:50],
                "condition":               cond,
                "context_relevance":       t.get("context_relevance"),
                "answer_faithfulness":     t.get("answer_faithfulness"),
                "context_utilization":     t.get("context_utilization"),
                "answer_completeness":     t.get("answer_completeness"),
                "answer_f1":               t.get("answer_f1", 0),
                "trace_mean":              cr.get("trace_mean", 0),
                "t_retrieve_s":            cr.get("t_retrieve_s", 0),
                "n_chunks":                cr.get("n_chunks", 0),
            })
    return pd.DataFrame(rows), summary


def load_latency() -> tuple[pd.DataFrame | None, dict | None]:
    raw     = _load_json(RESULTS_DIR / "latency_benchmark.json")
    summary = _load_json(RESULTS_DIR / "latency_summary.json")
    if raw is None:
        return None, summary
    rows = []
    for qr in raw:
        for strategy, tlist in qr.get("timings", {}).items():
            for t in tlist:
                rows.append({
                    "query_id":    qr["query_id"],
                    "strategy":    strategy,
                    "rep":         t["rep"],
                    "t_embed_s":   t["t_embed_s"],
                    "t_retrieve_s":t["t_retrieve_s"],
                    "t_total_s":   t["t_total_s"],
                })
    return pd.DataFrame(rows), summary


# ═══════════════════════════════════════════════════════════════════════════════
# A.  Dim experiment plots
# ═══════════════════════════════════════════════════════════════════════════════

def plot_d1_security_delta_vs_weight(df: pd.DataFrame, out_dir: Path) -> Path:
    """Mean security delta vs weight scalar, one line per normalisation combo."""
    def _weight_from_cfg(cfg_id: str) -> float:
        # config_id format: d4_w0.50_nb0_na0
        try:
            return float(cfg_id.split("_w")[1].split("_")[0])
        except Exception:
            return 0.0

    def _norm_label(cfg_id: str) -> str:
        nb = "norm_before" if "_nb1_" in cfg_id else ""
        na = "norm_after"  if "_na1"  in cfg_id else ""
        return " + ".join(filter(None, [nb, na])) or "no normalisation"

    df = df[df["is_restricted"]].copy()
    df["weight"]     = df["config_id"].apply(_weight_from_cfg)
    df["norm_label"] = df["config_id"].apply(_norm_label)

    fig, ax = plt.subplots(figsize=(9, 5))
    for label, grp in df.groupby("norm_label"):
        agg = grp.groupby("weight")["security_delta"].mean().reset_index()
        ax.plot(agg["weight"], agg["security_delta"], marker="o", label=label)

    ax.axhline(0, color="black", linewidth=1)
    ax.set_xlabel("Weight scalar applied to extra dimensions")
    ax.set_ylabel("Mean security delta\n(sim_unauth − sim_auth)")
    ax.set_title("Security Delta vs Weight — Effect of Normalisation")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    out = out_dir / "d1_security_delta_vs_weight.png"
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)
    print(f"  ✔  {out}")
    return out


def plot_d2_auth_vs_unauth_scatter(df: pd.DataFrame, out_dir: Path) -> Path:
    """Scatter: sim(auth) vs sim(unauth) per restricted chunk, one subplot per config."""
    restricted = df[df["is_restricted"]].copy()
    configs    = restricted["config_id"].unique()[:6]   # limit to 6 for readability
    ncols = 3
    nrows = int(np.ceil(len(configs) / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 4*nrows))
    axes = np.array(axes).flatten()

    for ax, cfg in zip(axes, configs):
        sub = restricted[restricted["config_id"] == cfg]
        ax.scatter(sub["sim_auth_restricted"], sub["sim_unauth_restricted"],
                   alpha=0.4, s=15, color="#C44E52")
        lim = [min(sub["sim_auth_restricted"].min(), sub["sim_unauth_restricted"].min()) - 0.02,
               max(sub["sim_auth_restricted"].max(), sub["sim_unauth_restricted"].max()) + 0.02]
        ax.plot(lim, lim, "k--", linewidth=1, label="y=x (no effect)")
        ax.set_xlim(lim); ax.set_ylim(lim)
        ax.set_xlabel("sim(auth_q, restricted_chunk)")
        ax.set_ylabel("sim(unauth_q, restricted_chunk)")
        ax.set_title(cfg, fontsize=8)
        ax.legend(fontsize=7)

    for ax in axes[len(configs):]:
        ax.set_visible(False)

    fig.suptitle("Auth vs Unauth Similarity per Restricted Chunk\n"
                 "Points below y=x mean restriction is working", fontsize=11)
    out = out_dir / "d2_auth_vs_unauth_scatter.png"
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)
    print(f"  ✔  {out}")
    return out


def plot_d3_restricted_recovery(df_topk: pd.DataFrame, out_dir: Path) -> Path:
    """% restricted chunks appearing in auth vs unauth top-K retrieval."""
    agg = df_topk.groupby("config_id")[["pct_auth", "pct_unauth"]].mean().reset_index()
    x   = np.arange(len(agg))
    w   = 0.35

    fig, ax = plt.subplots(figsize=(max(8, len(agg)*0.8), 5))
    ax.bar(x - w/2, agg["pct_auth"],   w, label="Authorised query",   color="#55A868", alpha=0.85)
    ax.bar(x + w/2, agg["pct_unauth"], w, label="Unauthorised query", color="#C44E52", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(agg["config_id"], rotation=45, ha="right", fontsize=8)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Fraction of restricted chunks in top-K")
    ax.set_title("Restricted Chunk Recovery: Auth vs Unauth Queries\n"
                 "Goal: auth=high, unauth=low")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    out = out_dir / "d3_restricted_recovery.png"
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)
    print(f"  ✔  {out}")
    return out


def plot_d4_collateral_damage(df: pd.DataFrame, out_dir: Path) -> Path:
    """Distribution of collateral_delta for open chunks across configs."""
    open_df  = df[~df["is_restricted"]]
    if open_df.empty:
        print("  ⚠  No open-chunk data for collateral damage plot.")
        return None

    fig, ax = plt.subplots(figsize=(9, 4))
    for cfg_id, grp in open_df.groupby("config_id"):
        vals = grp["collateral_delta"].dropna()
        ax.hist(vals, bins=20, alpha=0.5, label=cfg_id)
    ax.axvline(0, color="black", linewidth=1.5)
    ax.set_xlabel("Δ similarity for open chunks (auth_query − baseline)")
    ax.set_ylabel("Count")
    ax.set_title("Collateral Damage — How Much Does Augmentation Change Open Chunks?")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(axis="y", alpha=0.3)
    out = out_dir / "d4_collateral_damage.png"
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)
    print(f"  ✔  {out}")
    return out


def plot_d5_normalise_comparison(df: pd.DataFrame, out_dir: Path) -> Path:
    """Box: security_delta split by normalisation strategy."""
    restricted = df[df["is_restricted"]].copy()

    def _norm_label(cfg_id: str) -> str:
        nb = cfg_id.count("nb1") > 0
        na = cfg_id.count("na1") > 0
        return {(False,False):"none",(True,False):"before",(False,True):"after",(True,True):"both"}[(nb,na)]

    restricted["norm"] = restricted["config_id"].apply(_norm_label)
    groups = ["none", "before", "after", "both"]
    data   = [restricted[restricted["norm"]==g]["security_delta"].dropna().values for g in groups]

    fig, ax = plt.subplots(figsize=(8, 5))
    bp = ax.boxplot(data, patch_artist=True, widths=0.5,
                    medianprops=dict(color="black", linewidth=2))
    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color); patch.set_alpha(0.75)
    ax.set_xticklabels(groups)
    ax.axhline(0, color="black", linewidth=1)
    ax.set_xlabel("Normalisation strategy")
    ax.set_ylabel("Security delta (sim_unauth − sim_auth)")
    ax.set_title("Effect of Normalisation on Security Delta\n"
                 "More negative = stronger restriction signal")
    ax.grid(axis="y", alpha=0.3)
    out = out_dir / "d5_normalise_comparison.png"
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)
    print(f"  ✔  {out}")
    return out


# ═══════════════════════════════════════════════════════════════════════════════
# B.  TRACE conditions plots
# ═══════════════════════════════════════════════════════════════════════════════

def plot_t1_trace_radar(summary: dict, out_dir: Path) -> Path | None:
    if summary is None:
        return None
    cond_summary = summary.get("summary", summary)
    conditions   = list(cond_summary.keys())
    N            = len(TRACE_METRICS)
    angles       = [n / float(N) * 2 * np.pi for n in range(N)]
    angles      += angles[:1]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([l.replace("\n", " ") for l in TRACE_LABELS], fontsize=10)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])

    for cond in conditions:
        vals = [
            float(cond_summary[cond].get(m) or 0)
            for m in TRACE_METRICS
        ]
        vals += vals[:1]
        color = COND_COLORS.get(cond, "#888888")
        ax.plot(angles, vals, "o-", linewidth=2, label=cond, color=color)
        ax.fill(angles, vals, alpha=0.08, color=color)

    ax.set_title("TRACe Metrics Radar — All Conditions", fontsize=12, pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), fontsize=9)
    out = out_dir / "t1_trace_radar_all_conditions.png"
    fig.tight_layout(); fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  ✔  {out}")
    return out


def plot_t2_trace_bar_grouped(summary: dict, out_dir: Path) -> Path | None:
    if summary is None:
        return None
    cond_summary = summary.get("summary", summary)
    conditions   = list(cond_summary.keys())
    x            = np.arange(len(TRACE_METRICS))
    w            = 0.8 / max(len(conditions), 1)

    fig, ax = plt.subplots(figsize=(11, 5))
    for i, cond in enumerate(conditions):
        vals   = [float(cond_summary[cond].get(m) or 0) for m in TRACE_METRICS]
        offset = (i - len(conditions)/2 + 0.5) * w
        color  = COND_COLORS.get(cond, f"C{i}")
        ax.bar(x + offset, vals, w, label=cond, color=color, alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels([l.replace("\n", " ") for l in TRACE_LABELS])
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Mean score")
    ax.set_title("TRACe Metrics — Grouped by Condition")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    out = out_dir / "t2_trace_bar_grouped.png"
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)
    print(f"  ✔  {out}")
    return out


def plot_t3_answer_f1(summary: dict, out_dir: Path) -> Path | None:
    if summary is None:
        return None
    cond_summary = summary.get("summary", summary)
    conditions   = list(cond_summary.keys())
    vals         = [float(cond_summary[c].get("answer_f1_mean") or 0) for c in conditions]
    colors       = [COND_COLORS.get(c, "#888888") for c in conditions]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(conditions, vals, color=colors, alpha=0.85)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Answer F1 (token overlap vs ground truth)")
    ax.set_title("Answer Correctness F1 per Condition")
    ax.grid(axis="y", alpha=0.3)
    out = out_dir / "t3_answer_f1.png"
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)
    print(f"  ✔  {out}")
    return out


def plot_t4_retrieve_time(summary: dict, out_dir: Path) -> Path | None:
    if summary is None:
        return None
    cond_summary = summary.get("summary", summary)
    conditions   = list(cond_summary.keys())
    vals         = [float(cond_summary[c].get("t_retrieve_mean") or 0) for c in conditions]
    colors       = [COND_COLORS.get(c, "#888888") for c in conditions]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(conditions, [v * 1000 for v in vals], color=colors, alpha=0.85)
    ax.set_ylabel("Mean retrieval time (ms)")
    ax.set_title("Retrieval Latency per Condition (from TRACE eval)")
    ax.grid(axis="y", alpha=0.3)
    out = out_dir / "t4_retrieve_time.png"
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)
    print(f"  ✔  {out}")
    return out


# ═══════════════════════════════════════════════════════════════════════════════
# C.  Latency benchmark plots
# ═══════════════════════════════════════════════════════════════════════════════

def plot_l1_latency_boxplot(df: pd.DataFrame, out_dir: Path) -> Path:
    strategies = df["strategy"].unique()
    data = [df[df["strategy"]==s]["t_retrieve_s"].dropna().values * 1000 for s in strategies]
    colors = [COND_COLORS.get(s, "#888888") for s in strategies]

    fig, ax = plt.subplots(figsize=(10, 5))
    bp = ax.boxplot(data, patch_artist=True, widths=0.5,
                    medianprops=dict(color="black", linewidth=2))
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color); patch.set_alpha(0.75)
    ax.set_xticklabels(strategies, rotation=20, ha="right")
    ax.set_ylabel("Retrieval latency (ms)")
    ax.set_title("Retrieval Latency Distribution per Strategy")
    ax.grid(axis="y", alpha=0.3)
    out = out_dir / "l1_latency_boxplot.png"
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)
    print(f"  ✔  {out}")
    return out


def plot_l2_latency_breakdown(df: pd.DataFrame, out_dir: Path) -> Path:
    agg = df.groupby("strategy").agg(
        t_embed_mean   = ("t_embed_s",    "mean"),
        t_retrieve_mean= ("t_retrieve_s", "mean"),
    ).reset_index()
    x = np.arange(len(agg))
    w = 0.5
    emb_ms = agg["t_embed_mean"].values   * 1000
    ret_ms = agg["t_retrieve_mean"].values * 1000

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x, emb_ms, w, label="Embed",    color="#4C72B0", alpha=0.85)
    ax.bar(x, ret_ms, w, bottom=emb_ms, label="Retrieve", color="#DD8452", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(agg["strategy"], rotation=20, ha="right")
    ax.set_ylabel("Time (ms)")
    ax.set_title("Latency Breakdown: Embed vs Retrieve per Strategy")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    out = out_dir / "l2_latency_breakdown.png"
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)
    print(f"  ✔  {out}")
    return out


def plot_l3_latency_cdf(df: pd.DataFrame, out_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(9, 5))
    for strategy in df["strategy"].unique():
        vals = np.sort(df[df["strategy"]==strategy]["t_retrieve_s"].dropna().values) * 1000
        cdf  = np.arange(1, len(vals)+1) / len(vals)
        color = COND_COLORS.get(strategy, "#888888")
        ax.plot(vals, cdf, label=strategy, color=color, linewidth=2)
    ax.set_xlabel("Retrieval latency (ms)")
    ax.set_ylabel("Cumulative fraction")
    ax.set_title("CDF of Retrieval Latency per Strategy")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    out = out_dir / "l3_latency_cdf.png"
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)
    print(f"  ✔  {out}")
    return out


def plot_l4_overhead_ratio(df: pd.DataFrame, out_dir: Path) -> Path:
    agg = df.groupby("strategy")["t_retrieve_s"].mean()
    if "baseline" not in agg.index:
        print("  ⚠  No baseline in latency data — skipping overhead ratio.")
        return None
    baseline = agg["baseline"]
    ratios   = (agg / baseline).drop("baseline")
    colors   = [COND_COLORS.get(s, "#888888") for s in ratios.index]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(ratios.index, ratios.values, color=colors, alpha=0.85)
    ax.axhline(1.0, color="black", linestyle="--", label="baseline (1×)")
    ax.set_ylabel("Retrieval time / baseline time")
    ax.set_title("Overhead Ratio vs Baseline Retrieval")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    out = out_dir / "l4_overhead_ratio.png"
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)
    print(f"  ✔  {out}")
    return out


# ═══════════════════════════════════════════════════════════════════════════════
# Main dispatcher
# ═══════════════════════════════════════════════════════════════════════════════

DIM_PLOTS   = ["d1","d2","d3","d4","d5"]
TRACE_PLOTS = ["t1","t2","t3","t4"]
LAT_PLOTS   = ["l1","l2","l3","l4"]
ALL_PLOTS   = DIM_PLOTS + TRACE_PLOTS + LAT_PLOTS


def main(plots: list[str] = None, output: str = "plots") -> None:
    out_dir = Path(output)
    out_dir.mkdir(parents=True, exist_ok=True)
    wanted = set(plots or ALL_PLOTS)
    print(f"\nGenerating plots → {out_dir}/\n")

    # ── Load data lazily ──────────────────────────────────────────────────────
    df_dim     = load_dim_results()    if wanted & set(DIM_PLOTS)   else None
    df_topk    = load_dim_topk()       if "d3" in wanted            else None
    df_trace, trace_summary = load_trace_conditions() if wanted & set(TRACE_PLOTS) else (None, None)
    df_lat, _  = load_latency()        if wanted & set(LAT_PLOTS)   else (None, None)

    if df_dim is not None:
        if "d1" in wanted: plot_d1_security_delta_vs_weight(df_dim, out_dir)
        if "d2" in wanted: plot_d2_auth_vs_unauth_scatter(df_dim, out_dir)
        if "d3" in wanted and df_topk is not None:
            plot_d3_restricted_recovery(df_topk, out_dir)
        if "d4" in wanted: plot_d4_collateral_damage(df_dim, out_dir)
        if "d5" in wanted: plot_d5_normalise_comparison(df_dim, out_dir)

    if trace_summary is not None:
        if "t1" in wanted: plot_t1_trace_radar(trace_summary, out_dir)
        if "t2" in wanted: plot_t2_trace_bar_grouped(trace_summary, out_dir)
        if "t3" in wanted: plot_t3_answer_f1(trace_summary, out_dir)
        if "t4" in wanted: plot_t4_retrieve_time(trace_summary, out_dir)

    if df_lat is not None:
        if "l1" in wanted: plot_l1_latency_boxplot(df_lat, out_dir)
        if "l2" in wanted: plot_l2_latency_breakdown(df_lat, out_dir)
        if "l3" in wanted: plot_l3_latency_cdf(df_lat, out_dir)
        if "l4" in wanted: plot_l4_overhead_ratio(df_lat, out_dir)

    print(f"\nDone. All plots saved to {out_dir}/\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot all experiment results")
    parser.add_argument("--plot",   "-p", nargs="+",
                        choices=["all","dim","trace","latency"] + ALL_PLOTS,
                        default=["all"])
    parser.add_argument("--output", "-o", default="plots")
    args = parser.parse_args()

    # Expand group aliases
    wanted: list[str] = []
    for p in args.plot:
        if p == "all":    wanted += ALL_PLOTS
        elif p == "dim":    wanted += DIM_PLOTS
        elif p == "trace":  wanted += TRACE_PLOTS
        elif p == "latency":wanted += LAT_PLOTS
        else:               wanted.append(p)

    main(plots=list(set(wanted)), output=args.output)
