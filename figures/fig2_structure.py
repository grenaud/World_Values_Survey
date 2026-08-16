"""Figure 2 - the structure that language family picks out.

a  hierarchical clustering of the 66 societies, leaves colored by family
b  PC1/PC2 of country mean responses
c  the items that load most heavily on PC1

Panel c replaces the two circular loading histograms of the earlier draft.
A rose diagram encodes loading angle as sector direction and count as radius,
which asks the reader to compare wedge areas; naming the items that actually
drive PC1 answers the same question directly.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform
from adjustText import adjust_text

import wvs_pipeline as wp
import wvs_style as st
from wvs_meta import COUNTRY_INFO, QUESTIONS

OUT = Path(__file__).resolve().parent / "out"
OUT.mkdir(exist_ok=True)


def short_question(code):
    """Trim the codebook wording to something that fits a tick label."""
    q = QUESTIONS.get(code, {}).get("question", code)
    q = q.replace("Important child qualities: ", "Child: ")
    q = q.replace("Important in life: ", "Life: ")
    q = q.replace("Justifiable: ", "Justifiable: ")
    q = q.replace("Neighbors: ", "Neighbors: ")
    q = q.replace("Confidence: ", "Confidence: ")
    return q if len(q) <= 54 else q[:52].rstrip(" ,;:") + "…"


def panel_dendrogram(ax, cohort):
    d = cohort.distances()
    Z = linkage(squareform(d.values, checks=False), method="average")
    dn = dendrogram(Z, labels=list(d.index), orientation="left",
                    no_plot=True, distance_sort="descending")

    # redraw by hand: scipy's own colors and linewidths do not follow the style
    icoord = np.array(dn["dcoord"])          # x = distance, y = leaf position
    dcoord = np.array(dn["icoord"])
    for xs, ys in zip(icoord, dcoord):
        ax.plot(xs, ys, color="#b9b9c2", lw=0.55, solid_joinstyle="miter",
                zorder=1)

    leaves = dn["ivl"]
    ypos = np.arange(len(leaves)) * 10 + 5
    ax.set_yticks(ypos)
    ax.set_yticklabels(leaves, fontsize=5.3)
    for tick, code in zip(ax.get_yticklabels(), leaves):
        tick.set_color(st.family_color(COUNTRY_INFO[code][3]))
        tick.set_fontweight(600)
    ax.tick_params(axis="y", length=0, pad=1.5)
    ax.set_ylim(0, len(leaves) * 10)
    ax.invert_xaxis()
    ax.set_xlabel("Distance between mean response profiles")
    st.tidy(ax, grid="none", spines=("bottom",))
    ax.xaxis.grid(True, lw=0.5, color=st.RULE)
    st.panel_title(ax, "Societies cluster by language family",
                   sub="Average linkage on Euclidean distance", pad=11)


def panel_pca(ax, cohort):
    scores, loadings, ev = cohort.pca(2)
    fam = pd.Series({c: COUNTRY_INFO[c][3] for c in scores.index})
    texts = []

    for f in st.FAMILY_ORDER:
        sel = scores[fam.reindex(scores.index).map(st._canon) == f]
        if sel.empty:
            continue
        ax.scatter(sel["PC1"], sel["PC2"], s=17, marker=st.family_marker(f),
                   facecolor=st.family_color(f), edgecolor="white",
                   linewidth=0.45, zorder=3, label=st.family_label(f))
        for code, row in sel.iterrows():
            texts.append(ax.text(row["PC1"], row["PC2"], code, fontsize=5.0,
                                 color=st.INK, zorder=4,
                                 path_effects=st.halo(1.3)))

    adjust_text(texts, ax=ax, expand=(1.10, 1.18), force_text=(0.18, 0.28),
                arrowprops=dict(arrowstyle="-", color=st.INK_FAINT, lw=0.35,
                                shrinkA=0.5, shrinkB=1.5))

    ax.axhline(0, color=st.RULE, lw=0.5, zorder=0)
    ax.axvline(0, color=st.RULE, lw=0.5, zorder=0)
    ax.set_xlabel(f"PC1  ({ev[0] * 100:.1f}% of variance)")
    ax.set_ylabel(f"PC2  ({ev[1] * 100:.1f}%)")
    st.tidy(ax, grid="both", spines=("left", "bottom"))
    st.panel_title(ax, "Principal components of country mean responses",
                   sub="199 items, standardized; each point is one society",
                   pad=11)
    return loadings, scores


def panel_loadings(ax, cohort, loadings, k=9):
    """The k items at each end of PC1, as a signed bar with the wording shown."""
    pc1 = loadings["PC1"].sort_values()
    top = pd.concat([pc1[:k], pc1[-k:]])

    y = np.arange(len(top))
    colors = [st.THEME_COLOR.get(QUESTIONS.get(q, {}).get("category"), st.INK_FAINT)
              for q in top.index]
    ax.barh(y, top.values, height=0.66, color=colors, edgecolor="white",
            linewidth=0.5)
    ax.axvline(0, color=st.INK_FAINT, lw=0.7, zorder=4)

    ax.set_yticks(y)
    ax.set_yticklabels([short_question(q) for q in top.index], fontsize=5.5,
                       color=st.INK)
    ax.tick_params(axis="y", length=0, pad=2)
    ax.set_ylim(-0.8, len(top) - 0.2)
    ax.set_xlabel("Loading on PC1   "
                  "(negative: traditional and religious;  "
                  "positive: secular and permissive)")
    lim = np.abs(top.values).max() * 1.10
    ax.set_xlim(-lim, lim)
    st.tidy(ax, grid="x", spines=("bottom",))

    handles = [plt.Line2D([], [], marker="s", linestyle="none", markersize=3.6,
                          markerfacecolor=st.THEME_COLOR[t],
                          markeredgecolor="none", label=st.theme_label(t))
               for t in st.THEME_ORDER
               if t in {QUESTIONS.get(q, {}).get("category") for q in top.index}]
    leg = ax.legend(handles=handles, loc="lower right", ncol=len(handles),
                    frameon=False, fontsize=5.9, handletextpad=0.4,
                    columnspacing=1.0, bbox_to_anchor=(1.0, -0.245))
    for t in leg.get_texts():
        t.set_color(st.INK)
    st.panel_title(ax, "What separates societies along PC1",
                   sub=f"The {k} items at each end of the component", pad=11)


def main():
    st.use_style()
    cohort = wp.cohorts()["filtered"]

    fig = plt.figure(figsize=(st.PAGE_WIDTH, 8.1))
    gs = GridSpec(2, 2, figure=fig, width_ratios=[0.80, 1.30],
                  height_ratios=[1.62, 1.0],
                  left=0.062, right=0.985, top=0.955, bottom=0.052,
                  hspace=0.30, wspace=0.14)

    ax_dend = fig.add_subplot(gs[0, 0])
    ax_pca = fig.add_subplot(gs[0, 1])
    ax_load = fig.add_subplot(gs[1, :])
    ax_load.set_position([0.335, 0.070, 0.650, 0.278])

    panel_dendrogram(ax_dend, cohort)
    loadings, _ = panel_pca(ax_pca, cohort)
    panel_loadings(ax_load, cohort, loadings)

    fig.text(0.006, 0.980, "A", fontsize=9.5, fontweight=700, color=st.INK)
    fig.text(0.362, 0.980, "B", fontsize=9.5, fontweight=700, color=st.INK)
    fig.text(0.312, 0.405, "C", fontsize=9.5, fontweight=700, color=st.INK)

    st.family_legend(ax_pca, ncol=2, loc="lower right",
                     bbox=(1.0, 0.005), fontsize=5.8)
    st.save_caption_notes(OUT / "Figure2.png", [
        "World Values Survey wave 7, v6.0; native-born and home-language "
        "sample (n = 83,770 respondents across 66 societies).",
    ])

    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"Figure2.{ext}")
    print("wrote", OUT / "Figure2.png")


if __name__ == "__main__":
    main()
