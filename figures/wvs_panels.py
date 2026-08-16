"""Reusable panel builders shared by more than one figure."""

from __future__ import annotations

import numpy as np
import pandas as pd

import wvs_style as st
from wvs_meta import COUNTRY_INFO

#: Below this many respondents a country mean over 199 items is too noisy to
#: read as a point estimate; such societies are drawn hollow and flagged.
SMALL_N = 200


def nearest_neighbors(ax, cohort, focus, k=12, se_mode="delta", show_n=True,
                      title=None, sub=None, xlabel=True, highlight=()):
    """Ranked dot plot of the k societies closest to ``focus``.

    Replaces the earlier draft's 67-bar chart.  All but the leading dozen bars
    were visually identical there, and the bars carried a viridis fill that
    re-encoded the y value a second time; a ranked dot plot with a confidence
    interval shows the comparison the text actually makes.
    """
    d, lo, hi = cohort.distance_ci(se_mode=se_mode)
    row = d.loc[focus].drop(focus).sort_values()[:k]
    lo_r, hi_r = lo.loc[focus, row.index], hi.loc[focus, row.index]
    n = cohort.counts.median(axis=1)

    y = np.arange(len(row))[::-1]
    for yi, code in zip(y, row.index):
        color = st.family_color(COUNTRY_INFO[code][3])
        small = n.get(code, np.inf) < SMALL_N
        ax.plot([lo_r[code], hi_r[code]], [yi, yi], color=color, lw=1.5,
                alpha=0.45, solid_capstyle="round", zorder=2)
        ax.plot([row[code]], [yi], marker=st.family_marker(COUNTRY_INFO[code][3]),
                markersize=4.2, markerfacecolor="white" if small else color,
                markeredgecolor=color, markeredgewidth=0.9, zorder=3,
                linestyle="none")
        if code in highlight:
            ax.axhspan(yi - 0.5, yi + 0.5, color=color, alpha=0.10, zorder=0)

    labels = []
    for code in row.index:
        lab = code
        if show_n and n.get(code, np.inf) < SMALL_N:
            lab += f"  (n={int(n[code])})"
        labels.append(lab)

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=5.8)
    for tick, code in zip(ax.get_yticklabels(), row.index):
        tick.set_color(st.family_color(COUNTRY_INFO[code][3]))
        tick.set_fontweight(700 if code in highlight else 600)
    ax.tick_params(axis="y", length=0, pad=2)
    ax.set_ylim(-0.7, len(row) - 0.3)

    if xlabel:
        ax.set_xlabel("Euclidean distance from " + focus)
    st.tidy(ax, grid="x", spines=("bottom",))
    if title:
        st.panel_title(ax, title, sub=sub, pad=11 if sub else 4)
    return row


def rank_shift(ax, cohort_a, cohort_b, focus, k=10, labels=("A", "B")):
    """Slope chart of a society's nearest-neighbor ranking in two cohorts.

    Shows whether the societies closest to ``focus`` change between two
    populations (here, age bands) -- the comparison the text makes in words.
    """
    da = cohort_a.distances().loc[focus].drop(focus).sort_values()
    db = cohort_b.distances().loc[focus].drop(focus).sort_values()
    keep = list(dict.fromkeys(list(da.index[:k]) + list(db.index[:k])))

    ra = pd.Series(np.arange(1, len(da) + 1), index=da.index)
    rb = pd.Series(np.arange(1, len(db) + 1), index=db.index)

    for code in keep:
        color = st.family_color(COUNTRY_INFO[code][3])
        a, b = ra[code], rb[code]
        if a > k + 0.5 and b > k + 0.5:
            continue
        ax.plot([0, 1], [a, b], color=color, lw=1.0, alpha=0.75, zorder=2,
                solid_capstyle="round")
        ax.plot([0, 1], [a, b], marker="o", markersize=3.0, linestyle="none",
                markerfacecolor=color, markeredgecolor="white",
                markeredgewidth=0.5, zorder=3)
        if a <= k + 0.5:
            ax.text(-0.035, a, code, ha="right", va="center", fontsize=5.8,
                    color=color, fontweight=600)
        if b <= k + 0.5:
            ax.text(1.035, b, code, ha="left", va="center", fontsize=5.8,
                    color=color, fontweight=600)

    ax.set_xlim(-0.30, 1.30)
    ax.set_ylim(k + 1.1, 0.3)
    ax.set_yticks(range(1, k + 1))
    ax.set_ylabel("Rank by closeness to " + focus)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(labels, fontsize=6.4, color=st.INK)
    ax.tick_params(axis="x", length=0, pad=4)
    st.tidy(ax, grid="y", spines=("left",))
    ax.xaxis.grid(False)
    return ra, rb
