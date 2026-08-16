"""Figure 4 - splitting bilingual states by home language.

a  PC1/PC2 with Canada split into CDE/CDF and Kazakhstan into KZK/KZR
b  the twelve societies closest to each Kazakh split
c  the twelve societies closest to each Canadian split

The comparison the text makes is between the top of two ranked lists, so the
panels show a ranked dot plot of the leading dozen rather than all 67 bars.
Confidence intervals are the delta-method interval for a distance between two
sample mean vectors, using each society's own sample size.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from adjustText import adjust_text
from matplotlib.gridspec import GridSpec

import wvs_panels as wpan
import wvs_pipeline as wp
import wvs_style as st
from wvs_meta import COUNTRY_INFO

OUT = Path(__file__).resolve().parent / "out"
OUT.mkdir(exist_ok=True)

SPLITS = {"CDE": "Canada, English at home", "CDF": "Canada, French at home",
          "KZK": "Kazakhstan, Kazakh at home", "KZR": "Kazakhstan, Russian at home"}
PAIRS = [("KZK", "KZR"), ("CDE", "CDF")]


def panel_pca(ax, cohort):
    scores, _, ev = cohort.pca(2)
    fam = pd.Series({c: st._canon(COUNTRY_INFO[c][3]) for c in scores.index})
    texts = []

    for f in st.FAMILY_ORDER:
        sel = scores[fam == f]
        if sel.empty:
            continue
        split = sel.index.isin(SPLITS)
        ax.scatter(sel.loc[~split, "PC1"], sel.loc[~split, "PC2"], s=13,
                   marker=st.family_marker(f), facecolor=st.family_color(f),
                   edgecolor="white", linewidth=0.35, zorder=3, alpha=0.55)
        if split.any():
            ax.scatter(sel.loc[split, "PC1"], sel.loc[split, "PC2"], s=44,
                       marker=st.family_marker(f), facecolor=st.family_color(f),
                       edgecolor=st.INK, linewidth=0.8, zorder=5)
        for code, row in sel.iterrows():
            big = code in SPLITS
            texts.append(ax.text(row["PC1"], row["PC2"], code,
                                 fontsize=6.6 if big else 4.6,
                                 fontweight=700 if big else 400,
                                 color=st.INK if big else st.INK_MUTED,
                                 zorder=6, path_effects=st.halo(1.6 if big else 1.2)))

    # join the two halves of each split state so the reader sees the gap
    for a, b in PAIRS:
        ax.annotate("", xy=(scores.loc[b, "PC1"], scores.loc[b, "PC2"]),
                    xytext=(scores.loc[a, "PC1"], scores.loc[a, "PC2"]),
                    arrowprops=dict(arrowstyle="-", color=st.INK, lw=1.0,
                                    linestyle=(0, (2.5, 1.8)), alpha=0.8),
                    zorder=4)
        mx = (scores.loc[a, "PC1"] + scores.loc[b, "PC1"]) / 2
        my = (scores.loc[a, "PC2"] + scores.loc[b, "PC2"]) / 2
        gap = float(np.hypot(scores.loc[a, "PC1"] - scores.loc[b, "PC1"],
                             scores.loc[a, "PC2"] - scores.loc[b, "PC2"]))
        ax.text(mx, my - 0.55, f"{gap:.1f}", fontsize=5.8, color=st.INK_MUTED,
                ha="center", va="top", path_effects=st.halo(1.6))

    adjust_text(texts, ax=ax, expand=(1.08, 1.14), force_text=(0.14, 0.22),
                arrowprops=dict(arrowstyle="-", color=st.INK_FAINT, lw=0.3,
                                shrinkA=0.5, shrinkB=1.5))

    ax.axhline(0, color=st.RULE, lw=0.5, zorder=0)
    ax.axvline(0, color=st.RULE, lw=0.5, zorder=0)
    ax.set_xlabel(f"PC1  ({ev[0] * 100:.1f}% of variance)")
    ax.set_ylabel(f"PC2  ({ev[1] * 100:.1f}%)")
    st.tidy(ax, grid="both", spines=("left", "bottom"))
    st.panel_title(ax, "Two bilingual states, split by the language spoken at home",
                   sub="Bold marks: the four split populations; "
                       "dashed line and number give the gap between halves",
                   pad=11)


def main():
    st.use_style()
    cohort = wp.cohorts()["cankaz"]

    fig = plt.figure(figsize=(st.PAGE_WIDTH, 7.4))
    gs = GridSpec(3, 2, figure=fig, height_ratios=[1.58, 1.0, 1.0],
                  left=0.085, right=0.985, top=0.952, bottom=0.052,
                  hspace=0.60, wspace=0.34)

    ax_pca = fig.add_subplot(gs[0, :])
    axes = {code: fig.add_subplot(gs[1 + i, j])
            for i, (a, b) in enumerate(PAIRS)
            for j, code in enumerate((a, b))}

    panel_pca(ax_pca, cohort)

    for (a, b) in PAIRS:
        for code in (a, b):
            wpan.nearest_neighbors(
                axes[code], cohort, code, k=12,
                title=SPLITS[code],
                sub=f"12 closest societies  (n = "
                    f"{int(cohort.counts.median(axis=1)[code]):,})",
                highlight={a, b} - {code})

    # a shared x range per pair makes the two lists directly comparable
    for (a, b) in PAIRS:
        lo = min(axes[a].get_xlim()[0], axes[b].get_xlim()[0])
        hi = max(axes[a].get_xlim()[1], axes[b].get_xlim()[1])
        axes[a].set_xlim(lo, hi)
        axes[b].set_xlim(lo, hi)

    fig.text(0.006, 0.975, "A", fontsize=9.5, fontweight=700, color=st.INK)
    fig.text(0.006, 0.545, "B", fontsize=9.5, fontweight=700, color=st.INK)
    fig.text(0.006, 0.268, "C", fontsize=9.5, fontweight=700, color=st.INK)

    st.family_legend(ax_pca, ncol=2, loc="lower right", bbox=(1.0, 0.005),
                     fontsize=5.7)
    st.save_caption_notes(OUT / "Figure4.png", [
        "World Values Survey wave 7, v6.0.",
        "Bars are 95% delta-method intervals for the distance between two "
        "sample mean vectors, using each society's own sample size.",
        "Hollow marks flag societies with fewer than 200 respondents.",
    ])

    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"Figure4.{ext}")
    print("wrote", OUT / "Figure4.png")


if __name__ == "__main__":
    main()
