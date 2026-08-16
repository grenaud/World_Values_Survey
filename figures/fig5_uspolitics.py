"""Figure 5 - the United States, split by party identification.

a  PC1/PC2 with the USA split into Democrat- and Republican-identifying
   respondents and Canada into English- and French-speaking
b  the twelve societies closest to USD
c  the twelve societies closest to USR
d  how English Canada's nearest neighbors change between age bands

Panel d carries a caution the earlier draft did not: several comparison
societies fall below 200 respondents once the sample is cut by age, and their
codes are marked accordingly.
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

SPLITS = {"USD": "USA, Democrat-identifying", "USR": "USA, Republican-identifying",
          "CDE": "Canada, English at home", "CDF": "Canada, French at home"}


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
                                 zorder=6,
                                 path_effects=st.halo(1.6 if big else 1.2)))

    for a, b in (("USD", "USR"), ("CDE", "CDF")):
        ax.annotate("", xy=(scores.loc[b, "PC1"], scores.loc[b, "PC2"]),
                    xytext=(scores.loc[a, "PC1"], scores.loc[a, "PC2"]),
                    arrowprops=dict(arrowstyle="-", color=st.INK, lw=1.0,
                                    linestyle=(0, (2.5, 1.8)), alpha=0.8),
                    zorder=4)

    adjust_text(texts, ax=ax, expand=(1.08, 1.14), force_text=(0.14, 0.22),
                arrowprops=dict(arrowstyle="-", color=st.INK_FAINT, lw=0.3,
                                shrinkA=0.5, shrinkB=1.5))

    ax.axhline(0, color=st.RULE, lw=0.5, zorder=0)
    ax.axvline(0, color=st.RULE, lw=0.5, zorder=0)
    ax.set_xlabel(f"PC1  ({ev[0] * 100:.1f}% of variance)")
    ax.set_ylabel(f"PC2  ({ev[1] * 100:.1f}%)")
    st.tidy(ax, grid="both", spines=("left", "bottom"))
    st.panel_title(ax, "The United States separates by party identification",
                   sub="Bold marks: split populations; dashed line joins the "
                       "two halves of each state", pad=11)


def panel_age_shift(ax, young, old, focus="CDE", k=10, sub=None):
    wpan.rank_shift(ax, young, old, focus, k=k,
                    labels=("age 16-29", "age 50+"))
    st.panel_title(ax, f"Who {focus} sits closest to, by age band",
                   sub=sub or "Rank among all other societies", pad=11)


def main():
    st.use_style()
    cs = wp.cohorts()
    cohort = cs["uspolitics"]

    fig = plt.figure(figsize=(st.PAGE_WIDTH, 7.5))
    gs = GridSpec(3, 2, figure=fig, height_ratios=[1.52, 1.0, 1.06],
                  left=0.085, right=0.985, top=0.952, bottom=0.055,
                  hspace=0.56, wspace=0.34)

    ax_pca = fig.add_subplot(gs[0, :])
    ax_usd = fig.add_subplot(gs[1, 0])
    ax_usr = fig.add_subplot(gs[1, 1])
    ax_age = fig.add_subplot(gs[2, 0])
    ax_age_f = fig.add_subplot(gs[2, 1])

    panel_pca(ax_pca, cohort)

    n = cohort.counts.median(axis=1)
    for code, ax in (("USD", ax_usd), ("USR", ax_usr)):
        wpan.nearest_neighbors(ax, cohort, code, k=12, title=SPLITS[code],
                               sub=f"12 closest societies  (n = {int(n[code]):,})",
                               highlight={"USD", "USR"} - {code})
    lo = min(ax_usd.get_xlim()[0], ax_usr.get_xlim()[0])
    hi = max(ax_usd.get_xlim()[1], ax_usr.get_xlim()[1])
    ax_usd.set_xlim(lo, hi)
    ax_usr.set_xlim(lo, hi)

    panel_age_shift(ax_age, cs["cde_16_29"], cs["cde_50plus"], "CDE",
                    sub="English Canada: USD leads among the young, "
                        "GBR among the old")
    panel_age_shift(ax_age_f, cs["cde_16_29"], cs["cde_50plus"], "CDF",
                    sub="French Canada, for contrast")

    n_y = cs["cde_16_29"].counts.median(axis=1)
    n_o = cs["cde_50plus"].counts.median(axis=1)
    small = sorted({c for c in set(cs["cde_16_29"].countries)
                    & set(cs["cde_50plus"].countries)
                    if min(n_y.get(c, 0), n_o.get(c, 0)) < wpan.SMALL_N})
    small_note = ("Age-split samples are small for several societies: "
                  + ", ".join(f"{c} n = {int(min(n_y[c], n_o[c]))}"
                              for c in small[:8])
                  + (", and others; panels D and E should be read as "
                     "indicative only." if len(small) > 8 else "."))

    fig.text(0.006, 0.975, "A", fontsize=9.5, fontweight=700, color=st.INK)
    fig.text(0.006, 0.545, "B", fontsize=9.5, fontweight=700, color=st.INK)
    fig.text(0.520, 0.545, "C", fontsize=9.5, fontweight=700, color=st.INK)
    fig.text(0.006, 0.268, "D", fontsize=9.5, fontweight=700, color=st.INK)
    fig.text(0.520, 0.268, "E", fontsize=9.5, fontweight=700, color=st.INK)

    st.family_legend(ax_pca, ncol=2, loc="lower right", bbox=(1.0, 0.005),
                     fontsize=5.7)
    st.save_caption_notes(OUT / "Figure5.png", [
        "World Values Survey wave 7, v6.0.",
        "Bars are 95% delta-method intervals for the distance between two "
        "sample mean vectors, using each society's own sample size.",
        "Hollow marks flag societies with fewer than 200 respondents.",
        small_note,
    ])

    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"Figure5.{ext}")
    print("wrote", OUT / "Figure5.png")


if __name__ == "__main__":
    main()
