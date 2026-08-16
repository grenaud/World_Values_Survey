"""Figure 3 - non-negative matrix factorization recovers the same axis.

a  two-component NMF: each society's share of component 1, by family
b  three-component NMF in barycentric coordinates
c  NMF component 1 against PC1

Panel a replaces the stacked two-color bar of the earlier draft.  With two
components that sum to one, the second bar segment carries no information the
first does not; laying societies out by family on a single 0-1 axis shows the
grouping the panel is meant to demonstrate.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from adjustText import adjust_text
from matplotlib.gridspec import GridSpec
from scipy import stats

import wvs_pipeline as wp
import wvs_style as st
from wvs_meta import COUNTRY_INFO

OUT = Path(__file__).resolve().parent / "out"
OUT.mkdir(exist_ok=True)
rng = np.random.default_rng(0)


def panel_k2(ax, cohort):
    w = cohort.nmf(2)[0]["C1"]
    fam = pd.Series({c: st._canon(COUNTRY_INFO[c][3]) for c in w.index})
    order = [f for f in st.FAMILY_ORDER if f in set(fam)]
    texts = []

    for i, f in enumerate(order):
        yi = len(order) - 1 - i
        vals = w[fam[fam == f].index].sort_values()
        jitter = np.linspace(-0.14, 0.14, len(vals)) if len(vals) > 1 else [0.0]
        ax.plot([vals.min(), vals.max()], [yi, yi], color=st.RULE, lw=0.8,
                zorder=1, solid_capstyle="round")
        ax.scatter(vals.values, yi + np.array(jitter), s=15,
                   marker=st.family_marker(f), facecolor=st.family_color(f),
                   edgecolor="white", linewidth=0.4, zorder=3)
        for code, v, j in zip(vals.index, vals.values, jitter):
            texts.append(ax.text(v, yi + j + 0.18, code, fontsize=4.4,
                                 ha="center", va="bottom", color=st.INK_MUTED))

    adjust_text(texts, ax=ax, only_move={"text": "x", "static": "x"},
                expand=(1.06, 1.02), force_text=(0.08, 0.0))

    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([st.family_label(f) for f in order[::-1]], fontsize=6.3,
                       color=st.INK)
    for tick, f in zip(ax.get_yticklabels(), order[::-1]):
        tick.set_color(st.family_color(f))
        tick.set_fontweight(600)
    ax.tick_params(axis="y", length=0, pad=3)
    ax.set_ylim(-0.6, len(order) - 0.4)
    ax.set_xlim(-0.02, 1.02)
    ax.set_xlabel("Weight on NMF component 1  (component 2 is the remainder)")
    st.tidy(ax, grid="x", spines=("bottom",))
    st.panel_title(ax, "Two-component factorization separates families",
                   sub="One mark per society; families ordered as elsewhere",
                   pad=11)


def _tern(w):
    """Barycentric (a, b, c) -> Cartesian, unit-height triangle."""
    a, b, c = w[:, 0], w[:, 1], w[:, 2]
    return 0.5 * (2 * b + c) / (a + b + c), (np.sqrt(3) / 2) * c / (a + b + c)


def panel_ternary(ax, cohort):
    w = cohort.nmf(3)[0]
    x, y = _tern(w.values)
    fam = pd.Series({c: st._canon(COUNTRY_INFO[c][3]) for c in w.index})

    tri = np.array([[0, 0], [1, 0], [0.5, np.sqrt(3) / 2], [0, 0]])
    ax.plot(tri[:, 0], tri[:, 1], color=st.RULE, lw=0.8, zorder=1)
    for frac in (0.25, 0.5, 0.75):            # light interior guides
        w_ = np.array([[1 - frac, frac, 0], [1 - frac, 0, frac],
                       [0, 1 - frac, frac], [frac, 1 - frac, 0],
                       [frac, 0, 1 - frac], [0, frac, 1 - frac]], float)
        gx, gy = _tern(w_)
        for i in (0, 2, 4):
            ax.plot(gx[i:i + 2], gy[i:i + 2], color=st.RULE, lw=0.4,
                    alpha=0.7, zorder=0)

    texts = []
    for f in st.FAMILY_ORDER:
        sel = fam[fam == f].index
        if not len(sel):
            continue
        idx = [w.index.get_loc(c) for c in sel]
        ax.scatter(x[idx], y[idx], s=15, marker=st.family_marker(f),
                   facecolor=st.family_color(f), edgecolor="white",
                   linewidth=0.4, zorder=3)
        for c, i in zip(sel, idx):
            texts.append(ax.text(x[i], y[i], c, fontsize=4.4, color=st.INK,
                                 zorder=4, path_effects=st.halo(1.2)))
    adjust_text(texts, ax=ax, expand=(1.05, 1.12), force_text=(0.10, 0.18))

    for (px, py, lab, ha, va) in [(0, -0.035, "component 1", "center", "top"),
                                  (1, -0.035, "component 2", "center", "top"),
                                  (0.5, np.sqrt(3) / 2 + 0.03, "component 3",
                                   "center", "bottom")]:
        ax.text(px, py, lab, fontsize=5.9, color=st.INK_MUTED, ha=ha, va=va)

    ax.set_xlim(-0.10, 1.10)
    ax.set_ylim(-0.10, np.sqrt(3) / 2 + 0.10)
    ax.set_aspect("equal")
    ax.set_axis_off()
    st.panel_title(ax, "Three components, barycentric",
                   sub="Position = mix of the three latent value profiles",
                   pad=11)


def panel_pc1_vs_nmf(ax, cohort):
    w = cohort.nmf(2)[0]["C1"]
    scores, _, ev = cohort.pca(2)
    df = pd.DataFrame({"nmf": w, "pc1": scores["PC1"]}).dropna()
    fam = pd.Series({c: st._canon(COUNTRY_INFO[c][3]) for c in df.index})

    r, p = stats.pearsonr(df["nmf"], df["pc1"])
    b, a = np.polyfit(df["nmf"], df["pc1"], 1)
    xs = np.linspace(df["nmf"].min(), df["nmf"].max(), 50)
    ax.plot(xs, a + b * xs, color=st.INK_FAINT, lw=0.9, zorder=2,
            dash_capstyle="round", linestyle=(0, (4, 2)))

    for f in st.FAMILY_ORDER:
        sel = df[fam == f]
        if sel.empty:
            continue
        ax.scatter(sel["nmf"], sel["pc1"], s=15, marker=st.family_marker(f),
                   facecolor=st.family_color(f), edgecolor="white",
                   linewidth=0.4, zorder=3)

    ax.set_xlabel("Weight on NMF component 1")
    ax.set_ylabel(f"PC1  ({ev[0] * 100:.1f}% of variance)")
    st.tidy(ax, grid="both", spines=("left", "bottom"))
    ptxt = "p < 0.001" if p < 1e-3 else f"p = {p:.3f}"
    ax.text(0.03, 0.95, f"r = {r:.2f}   {ptxt}", transform=ax.transAxes,
            fontsize=6.4, color=st.INK, va="top", fontweight=600)
    st.panel_title(ax, "The first NMF component is PC1",
                   sub="Each mark is one society", pad=11)


def main():
    st.use_style()
    cohort = wp.cohorts()["filtered"]

    fig = plt.figure(figsize=(st.PAGE_WIDTH, 7.0))
    gs = GridSpec(2, 2, figure=fig, height_ratios=[1.05, 1.0],
                  width_ratios=[1.28, 1.0],
                  left=0.115, right=0.985, top=0.945, bottom=0.105,
                  hspace=0.38, wspace=0.20)

    ax_a = fig.add_subplot(gs[0, :])
    ax_b = fig.add_subplot(gs[1, 0])
    ax_c = fig.add_subplot(gs[1, 1])

    panel_k2(ax_a, cohort)
    panel_ternary(ax_b, cohort)
    panel_pc1_vs_nmf(ax_c, cohort)

    fig.text(0.008, 0.972, "A", fontsize=9.5, fontweight=700, color=st.INK)
    fig.text(0.008, 0.455, "B", fontsize=9.5, fontweight=700, color=st.INK)
    fig.text(0.600, 0.455, "C", fontsize=9.5, fontweight=700, color=st.INK)

    st.family_legend(fig, ncol=5, loc="lower center", bbox=(0.53, 0.016),
                     fontsize=7.0)

    st.save_caption_notes(OUT / "Figure3.png", [
        "World Values Survey wave 7, v6.0; native-born and home-language "
        "sample.",
        "NMF weights are row-normalized, so each society's components sum "
        "to one.",
    ])

    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"Figure3.{ext}")
    print("wrote", OUT / "Figure3.png")


if __name__ == "__main__":
    main()
