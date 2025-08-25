# scripts/plot_tuning_summary.py
# Create comparison charts for Prophet hyperparameter tuning results.

import math
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
CSV  = ROOT / "tuning_results.csv"
OUT  = ROOT / "reports"
OUT.mkdir(exist_ok=True)

def load_results():
    df = pd.read_csv(CSV)
    # Ensure correct dtypes
    num_cols = [
        "changepoint_prior_scale",
        "seasonality_prior_scale",
        "holidays_prior_scale",
        "changepoint_range",
        "weekly_mape",
    ]
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df[df["status"] == "ok"].dropna(subset=["weekly_mape"])
    return df

def find_default_row(df):
    mask = (
        (df["changepoint_prior_scale"] == 0.05) &
        (df["seasonality_mode"] == "additive") &
        (df["seasonality_prior_scale"] == 10) &
        (df["holidays_prior_scale"] == 10) &
        (df["changepoint_range"] == 0.8)
    )
    d = df[mask].copy()
    return d.iloc[0] if len(d) else None

def save_default_vs_best(df):
    best = df.sort_values("weekly_mape").iloc[0]
    default = find_default_row(df)

    labels, values = [], []
    labels.append("Best tuned")
    values.append(best["weekly_mape"])
    if default is not None:
        labels.append("Default")
        values.append(default["weekly_mape"])

    fig, ax = plt.subplots(figsize=(5, 4), dpi=150)
    ax.bar(labels, values)
    ax.set_ylabel("Weekly MAPE (%)")
    ax.set_title("Next-week MAPE: Best vs Default")
    for i, v in enumerate(values):
        ax.text(i, v, f"{v:.2f}%", ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(OUT / "default_vs_best.png")
    plt.close(fig)

    # Save quick stats for README
    stats = {"best_mape": float(best["weekly_mape"])}
    if default is not None:
        stats["default_mape"] = float(default["weekly_mape"])
        rel = (stats["default_mape"] - stats["best_mape"]) / stats["default_mape"] * 100.0
        stats["improvement_pct"] = float(rel)
    pd.Series(stats).to_json(OUT / "summary_stats.json")
    return stats

def scatter_cps_by_mode_faceted_crange(df):
    cr_values = sorted(df["changepoint_range"].unique())
    cols = min(4, len(cr_values))
    rows = math.ceil(len(cr_values)/cols)

    fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows), dpi=150)
    if rows * cols == 1:
        axes = np.array([[axes]])
    axes = axes.flatten()

    for ax, cr in zip(axes, cr_values):
        sub = df[df["changepoint_range"] == cr]
        # jitter x positions slightly so overlapping points are visible
        x = sub["changepoint_prior_scale"].values.astype(float)
        jitter = (np.random.rand(len(x)) - 0.5) * 0.01
        xj = x + jitter

        # plot each seasonality_mode separately
        for mode in sub["seasonality_mode"].unique():
            ssub = sub[sub["seasonality_mode"] == mode]
            ax.scatter(
                ssub["changepoint_prior_scale"].values,
                ssub["weekly_mape"].values,
                label=mode,
                alpha=0.8,
            )
        ax.set_title(f"changepoint_range = {cr}")
        ax.set_xlabel("changepoint_prior_scale")
        ax.set_ylabel("Weekly MAPE (%)")
        ax.grid(True)
        ax.legend()

    # Hide any unused axes
    for i in range(len(cr_values), len(axes)):
        fig.delaxes(axes[i])

    fig.suptitle("Weekly MAPE vs changepoint_prior_scale (colored by seasonality_mode, faceted by changepoint_range)")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(OUT / "mape_vs_cps_by_mode_facet_crange.png")
    plt.close(fig)

def boxplot_by_param(df, col, filename, title=None):
    order = sorted(df[col].unique(), key=lambda v: (isinstance(v, str), v))
    data = [df[df[col]==v]["weekly_mape"].values for v in order]

    fig, ax = plt.subplots(figsize=(6, 4), dpi=150)
    ax.boxplot(data, labels=[str(v) for v in order], showmeans=True)
    ax.set_xlabel(col)
    ax.set_ylabel("Weekly MAPE (%)")
    ax.set_title(title or f"Weekly MAPE by {col}")
    ax.grid(True, axis="y", linestyle=":", alpha=0.5)
    fig.tight_layout()
    fig.savefig(OUT / filename)
    plt.close(fig)

def heatmap_sps_hps_at_best_cps_crange(df):
    # Pick the best (median MAPE) cps and crange, then show how SPS x HPS behaves there
    pivot_candidates = (
        df.groupby(["changepoint_prior_scale", "changepoint_range"])["weekly_mape"]
          .median()
          .reset_index()
          .sort_values("weekly_mape")
    )
    if pivot_candidates.empty:
        return
    best_pair = pivot_candidates.iloc[0][["changepoint_prior_scale", "changepoint_range"]].to_dict()
    sub = df[
        (df["changepoint_prior_scale"] == best_pair["changepoint_prior_scale"]) &
        (df["changepoint_range"] == best_pair["changepoint_range"])
    ].copy()

    if sub.empty:
        return

    piv = sub.pivot_table(
        index="seasonality_prior_scale",
        columns="holidays_prior_scale",
        values="weekly_mape",
        aggfunc="median",
    ).sort_index().sort_index(axis=1)

    fig, ax = plt.subplots(figsize=(6, 5), dpi=150)
    im = ax.imshow(piv.values, aspect="auto")
    ax.set_xticks(range(piv.shape[1]))
    ax.set_xticklabels([str(c) for c in piv.columns])
    ax.set_yticks(range(piv.shape[0]))
    ax.set_yticklabels([str(r) for r in piv.index])
    ax.set_xlabel("holidays_prior_scale")
    ax.set_ylabel("seasonality_prior_scale")
    ax.set_title(
        f"Median Weekly MAPE heatmap @ cps={best_pair['changepoint_prior_scale']}, crange={best_pair['changepoint_range']}"
    )
    # annotate cells
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            val = piv.values[i, j]
            ax.text(j, i, f"{val:.2f}", ha="center", va="center")
    fig.colorbar(im, ax=ax, label="Weekly MAPE (%)")
    fig.tight_layout()
    fig.savefig(OUT / "heatmap_sps_hps_at_best_cps_crange.png")
    plt.close(fig)

def main():
    df = load_results()
    if df.empty:
        raise SystemExit("No successful rows found in tuning_results.csv (status=='ok').")

    # Save top10 table for quick reference
    top10 = df.sort_values("weekly_mape").head(10)
    top10.to_csv(OUT / "top10_table.csv", index=False)

    # 1) default vs best
    stats = save_default_vs_best(df)

    # 2) CPS vs MAPE, colored by mode, faceted by crange
    scatter_cps_by_mode_faceted_crange(df)

    # 3) Boxplots for each param
    boxplot_by_param(df, "changepoint_prior_scale", "box_by_cps.png", "Weekly MAPE by changepoint_prior_scale")
    boxplot_by_param(df, "seasonality_mode",      "box_by_mode.png", "Weekly MAPE by seasonality_mode")
    boxplot_by_param(df, "seasonality_prior_scale","box_by_sps.png", "Weekly MAPE by seasonality_prior_scale")
    boxplot_by_param(df, "holidays_prior_scale",   "box_by_hps.png", "Weekly MAPE by holidays_prior_scale")
    boxplot_by_param(df, "changepoint_range",      "box_by_crange.png", "Weekly MAPE by changepoint_range")

    # 4) Heatmap SPS×HPS at best cps/crange
    heatmap_sps_hps_at_best_cps_crange(df)

    # 5) Print a README-ready markdown block with your numbers
    best = df.sort_values("weekly_mape").iloc[0]
    default = find_default_row(df)
    lines = []
    lines.append("## 📈 Hyperparameter Tuning Summary")
    lines.append("")
    if default is not None:
        imp = (default['weekly_mape'] - best['weekly_mape']) / default['weekly_mape'] * 100.0
        lines.append(f"**Next-week MAPE improved from {default['weekly_mape']:.2f}% → {best['weekly_mape']:.2f}%** "
                     f"(**{imp:.1f}%** relative improvement) by tuning trend/seasonality/holiday priors.")
    else:
        lines.append(f"**Best next-week MAPE: {best['weekly_mape']:.2f}%**")
    lines.append("")
    lines.append("**Best config:**")
    lines.append(f"- `changepoint_prior_scale`: **{best['changepoint_prior_scale']}**")
    lines.append(f"- `seasonality_mode`: **{best['seasonality_mode']}**")
    lines.append(f"- `seasonality_prior_scale`: **{best['seasonality_prior_scale']}**")
    lines.append(f"- `holidays_prior_scale`: **{best['holidays_prior_scale']}**")
    lines.append(f"- `changepoint_range`: **{best['changepoint_range']}**")
    lines.append("")
    lines.append("**Charts:**")
    lines.append("![Best vs Default](reports/default_vs_best.png)")
    lines.append("")
    lines.append("![MAPE vs CPS by mode (faceted by changepoint_range)](reports/mape_vs_cps_by_mode_facet_crange.png)")
    lines.append("")
    lines.append("<details><summary>More comparisons</summary>")
    lines.append("")
    lines.append("![Box by cps](reports/box_by_cps.png)")
    lines.append("![Box by mode](reports/box_by_mode.png)")
    lines.append("![Box by seasonality_prior_scale](reports/box_by_sps.png)")
    lines.append("![Box by holidays_prior_scale](reports/box_by_hps.png)")
    lines.append("![Box by changepoint_range](reports/box_by_crange.png)")
    lines.append("![Heatmap SPS×HPS @ best cps & crange](reports/heatmap_sps_hps_at_best_cps_crange.png)")
    lines.append("")
    lines.append("</details>")
    md = "\n".join(lines)

    md_path = OUT / "README_snippet.md"
    with open(md_path, "w") as f:
        f.write(md)
    print("\n--- README snippet (also saved to reports/README_snippet.md) ---\n")
    print(md)
    print("\n----------------------------------------------------------------")

if __name__ == "__main__":
    main()
