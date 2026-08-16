"""Figure 1 - what the data are.

a  world map of the 66 surveyed societies, colored by language family
b  analyzed sample: societies and respondents per family
c  the 199 value items, by thematic category

Replaces the two pie charts of the earlier draft.  A pie with six or ten
wedges asks the reader to compare angles; the same numbers as a sorted bar
are read directly, and the bar can carry the second measure (respondents)
that the pie had no room for.
"""

from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec

import wvs_pipeline as wp
import wvs_style as st
from wvs_meta import COUNTRY_INFO, QUESTIONS

OUT = Path(__file__).resolve().parent / "out"
OUT.mkdir(exist_ok=True)

# Countries that are analytical splits of a parent society, not separate
# surveys; they must not be drawn twice on the map.
SPLIT_CODES = {"USS", "USN", "USD", "USR", "KZK", "KZR", "CDE", "CDF",
               "CANCHN", "USASPN", "GRBPLS", "AUSCHN", "NZLCHN", "DEUTRK",
               "NLDENG"}


def load():
    cohort = wp.cohorts()["filtered"]
    n_by_country = cohort.counts.max(axis=1)          # respondents per society
    fam = pd.Series({c: st._canon(COUNTRY_INFO[c][3]) for c in cohort.countries})
    return cohort, n_by_country, fam


def panel_map(ax, fam):
    world = gpd.read_file(Path(__file__).resolve().parent / "cache" / "world.gpkg")
    iso = next(c for c in ("ADM0_A3", "ISO_A3", "SOV_A3") if c in world.columns)
    world = world.to_crs("+proj=robin")               # equal-ish area, no polar blowup

    surveyed = fam[~fam.index.isin(SPLIT_CODES)]
    world["fam"] = world[iso].map(surveyed)

    world[world["fam"].isna()].plot(ax=ax, color="#f0f0f2",
                                    edgecolor="white", linewidth=0.25)
    covered = world[world["fam"].notna()]
    covered.plot(ax=ax, color=[st.family_color(f) for f in covered["fam"]],
                 edgecolor="white", linewidth=0.3)

    # Small states disappear at world scale; mark them with a leader dot so the
    # map is an honest inventory of what was surveyed.
    tiny = covered[covered.geometry.area < 6.0e10]
    if len(tiny):
        pts = tiny.geometry.representative_point()
        ax.scatter(pts.x, pts.y, s=9,
                   c=[st.family_color(f) for f in tiny["fam"]],
                   edgecolor="white", linewidth=0.45, zorder=5)
    missing = sorted(set(fam[~fam.index.isin(SPLIT_CODES)].index)
                     - set(world.loc[world["fam"].notna(), iso]))
    if missing:
        ax.text(0.995, 0.02, "not shown: " + ", ".join(missing),
                transform=ax.transAxes, ha="right", va="bottom",
                fontsize=5.6, color=st.INK_FAINT)

    ax.set_axis_off()
    ax.set_xlim(-1.70e7, 1.70e7)
    ax.set_ylim(-6.0e6, 8.6e6)                        # trim Antarctica
    ax.set_aspect("equal")
    ax.margins(0)


def panel_sample(ax, n_by_country, fam):
    surveyed = fam[~fam.index.isin(SPLIT_CODES)]
    n = n_by_country[surveyed.index]
    tab = pd.DataFrame({"fam": surveyed, "n": n})
    agg = tab.groupby("fam").agg(societies=("n", "size"), respondents=("n", "sum"))
    agg = agg.reindex([f for f in st.FAMILY_ORDER if f in agg.index])

    y = np.arange(len(agg))[::-1]
    colors = [st.family_color(f) for f in agg.index]
    ax.barh(y, agg["respondents"] / 1000, height=0.6, color=colors,
            edgecolor="white", linewidth=0.6)

    for yi, (_, row) in zip(y, agg.iterrows()):
        ax.text(row["respondents"] / 1000 + 0.4, yi,
                f"{row['respondents'] / 1000:,.1f}k",
                va="center", ha="left", fontsize=6.0, color=st.INK_MUTED)

    ax.set_yticks(y)
    ax.set_yticklabels([f"{st.family_label(f)}  ({int(agg.loc[f, 'societies'])})"
                        for f in agg.index], fontsize=6.4, color=st.INK)
    ax.set_xlabel("Respondents analyzed (thousands)")
    ax.set_xlim(0, (agg["respondents"].max() / 1000) * 1.20)
    ax.tick_params(axis="y", length=0)
    st.tidy(ax, grid="x", spines=("bottom",))
    st.panel_title(ax, "Analyzed sample by family",
                   sub="Societies per family in parentheses")


def panel_items(ax, cohort):
    items = [q for q in cohort.means_imp.columns if q in QUESTIONS]
    themes = pd.Series([QUESTIONS[q]["category"] for q in items]).value_counts()
    themes = themes.reindex([t for t in st.THEME_ORDER if t in themes.index])

    y = np.arange(len(themes))[::-1]
    ax.barh(y, themes.values, height=0.62,
            color=[st.THEME_COLOR[t] for t in themes.index],
            edgecolor="white", linewidth=0.6)
    for yi, (theme, v) in zip(y, themes.items()):
        ax.text(v + 1.2, yi, f"{v}", va="center", ha="left", fontsize=6.2,
                color=st.INK_MUTED)

    ax.set_yticks(y)
    ax.set_yticklabels([st.theme_label(t) for t in themes.index], fontsize=6.5,
                       color=st.INK)
    ax.set_xlabel("Survey items")
    ax.set_xlim(0, themes.max() * 1.16)
    ax.tick_params(axis="y", length=0)
    st.tidy(ax, grid="x", spines=("bottom",))
    st.panel_title(ax, f"{len(items)} value items by theme",
                   sub="Demographic and country-level indicators excluded")


def main():
    st.use_style()
    cohort, n_by_country, fam = load()

    fig = plt.figure(figsize=(st.PAGE_WIDTH, 5.55))
    gs = GridSpec(2, 2, figure=fig,
                  height_ratios=[1.52, 1.0], width_ratios=[1.0, 0.92],
                  left=0.155, right=0.985, top=0.910, bottom=0.055,
                  hspace=0.30, wspace=0.42)

    ax_map = fig.add_subplot(gs[0, :])
    ax_n = fig.add_subplot(gs[1, 0])
    ax_q = fig.add_subplot(gs[1, 1])

    panel_map(ax_map, fam)
    panel_sample(ax_n, n_by_country, fam)
    panel_items(ax_q, cohort)

    fig.text(0.012, 0.972, "A", fontsize=9.5, fontweight=700, color=st.INK)
    fig.text(0.020, 0.360, "B", fontsize=9.5, fontweight=700, color=st.INK)
    fig.text(0.545, 0.360, "C", fontsize=9.5, fontweight=700, color=st.INK)

    fig.text(0.038, 0.978,
             "66 societies of World Values Survey wave 7 (2017-2023), "
             "colored by the language family assigned in this study",
             fontsize=7.2, fontweight=600, color=st.INK, va="top")

    st.family_legend(ax_map, ncol=2, bbox=(0.005, -0.02), loc="lower left",
                     markers=False, fontsize=6.3)
    st.save_caption_notes(OUT / "Figure1.png", [
        "Source: World Values Survey wave 7, v6.0.",
        "Panel B counts respondents remaining after the native-born and "
        "home-language filter.",
    ])

    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"Figure1.{ext}")
    print("wrote", OUT / "Figure1.png")


if __name__ == "__main__":
    main()
