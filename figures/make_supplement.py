"""Regenerate the supplementary figure set into ``WVS_v6_new/``.

Same analyses as the ``performAnalyses`` figures in ``WVS.py``, drawn through
the shared style module so the supplement matches the main text. Nothing is
written to ``WVS_v6/``; the old set stays where it is for comparison.

Beyond the restyle, four figure types were redrawn rather than recolored,
because their form was the problem:

* **Country x item clustermap.** 199 question labels cannot be set at page
  width. The column dendrogram and a theme color strip carry the item side;
  the wording lives in the question table.
* **Per-society distance charts.** 67 bars where all but the first dozen are
  visually identical, with a viridis fill re-encoding the y value while the
  tick labels encoded family. Now a ranked dot plot of the nearest twelve
  with a confidence interval.
* **t-SNE and UMAP.** Were two separate figures; they answer the same question
  and belong side by side.
* **NMF k=2.** A two-segment stacked bar whose second segment is one minus
  the first.

Usage:

    python make_supplement.py              # everything
    python make_supplement.py --quick      # skip the 66 per-society charts
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from adjustText import adjust_text
from matplotlib.gridspec import GridSpec
from scipy.cluster.hierarchy import dendrogram, leaves_list, linkage
from scipy.spatial.distance import squareform
from sklearn.manifold import TSNE

import wvs_panels as wpan
import wvs_pipeline as wp
import wvs_style as st
from wvs_meta import COUNTRY_INFO, ENGLISH_PROFICIENCY, QUESTIONS

warnings.filterwarnings("ignore")

OUT = Path(__file__).resolve().parent.parent / "WVS_v6_new"
OUT.mkdir(exist_ok=True)

PROVENANCE = "World Values Survey wave 7, v6.0."
_INDEX: list[tuple[str, str]] = []          # (tex filename, section title)


# --------------------------------------------------------------------------- #
# output plumbing                                                              #
# --------------------------------------------------------------------------- #
def emit(fig, name, caption, notes=(), label=None, section=None):
    """Save one supplementary figure as PDF + PNG + .tex + caption sidecar."""
    pdf = OUT / f"{name}.pdf"
    png = OUT / f"{name}.png"
    fig.savefig(pdf)
    fig.savefig(png, dpi=200)
    plt.close(fig)

    st.save_caption_notes(png, [PROVENANCE, *notes])

    label = label or f"fig:supp:{name.replace('_', '-')}"
    (OUT / f"{name}.tex").write_text(
        "\\begin{figure}[htbp]\n"
        "  \\centering\n"
        f"  \\includegraphics[width=\\linewidth]{{WVS_v6_new/{name}.pdf}}\n"
        f"  \\caption{{{caption}}}\n"
        f"  \\label{{{label}}}\n"
        "\\end{figure}\n")
    _INDEX.append((f"{name}.tex", section))
    print(f"  {name}")


def short_question(code):
    """Trim the codebook wording to something that fits a label."""
    q = QUESTIONS.get(code, {}).get("question", code)
    q = (q.replace("Important child qualities: ", "Child: ")
          .replace("Important in life: ", "Life: "))
    return q if len(q) <= 44 else q[:42].rstrip(" ,;:") + "\u2026"


def _arrow_label(code):
    """Very short item label for a crowded biplot."""
    q = QUESTIONS.get(code, {}).get("question", code)
    for prefix in ("Important child qualities: ", "Important in life: ",
                   "Confidence: ", "Neighbors: ", "Justifiable: ",
                   "Political system: ", "Democracy: ", "Believe in: ",
                   "Active/Inactive membership: ", "Trust: ",
                   "Member: ", "How often do you "):
        q = q.replace(prefix, "")
    q = q if len(q) <= 26 else q[:24].rstrip(" ,;:") + "\u2026"
    return f"{q}"


def cohort_families(cohort):
    return pd.Series({c: st._canon(COUNTRY_INFO[c][3]) for c in cohort.countries})


def scatter_by_family(ax, xy, fam, labels=True, size=17, label_size=5.0):
    """The embedding scatter used by PCA, t-SNE and UMAP panels."""
    texts = []
    for f in st.FAMILY_ORDER:
        sel = xy[fam.reindex(xy.index) == f]
        if sel.empty:
            continue
        ax.scatter(sel.iloc[:, 0], sel.iloc[:, 1], s=size,
                   marker=st.family_marker(f), facecolor=st.family_color(f),
                   edgecolor="white", linewidth=0.45, zorder=3)
        if labels:
            for code, row in sel.iterrows():
                texts.append(ax.text(row.iloc[0], row.iloc[1], code,
                                     fontsize=label_size, color=st.INK,
                                     zorder=4, path_effects=st.halo(1.3)))
    if texts:
        adjust_text(texts, ax=ax, expand=(1.08, 1.14), force_text=(0.14, 0.22),
                    arrowprops=dict(arrowstyle="-", color=st.INK_FAINT, lw=0.3,
                                    shrinkA=0.5, shrinkB=1.5))
    st.tidy(ax, grid="both", spines=("left", "bottom"))
    return texts


# --------------------------------------------------------------------------- #
# per-cohort figures                                                           #
# --------------------------------------------------------------------------- #
def fig_pca(cohort, desc):
    scores, loadings, ev = cohort.pca(2)
    fam = cohort_families(cohort)

    st.use_style()
    fig = plt.figure(figsize=(st.PAGE_WIDTH, 5.6))
    gs = GridSpec(1, 1, figure=fig, left=0.085, right=0.985,
                  top=0.930, bottom=0.145)
    ax = fig.add_subplot(gs[0])

    scatter_by_family(ax, scores, fam)
    ax.axhline(0, color=st.RULE, lw=0.5, zorder=0)
    ax.axvline(0, color=st.RULE, lw=0.5, zorder=0)
    ax.set_xlabel(f"PC1  ({ev[0] * 100:.1f}% of variance)")
    ax.set_ylabel(f"PC2  ({ev[1] * 100:.1f}%)")
    st.panel_title(ax, "Principal components of country mean responses",
                   sub=desc, pad=11)
    st.family_legend(ax, ncol=5, loc="upper left", bbox=(0.0, -0.105),
                     fontsize=6.4)

    emit(fig, f"{cohort.tag}_pca",
         f"Principal component analysis of country mean responses. {desc}. "
         f"PC1 explains {ev[0] * 100:.1f}\\% and PC2 {ev[1] * 100:.1f}\\% of "
         f"the variance across {len(cohort.countries)} societies.",
         notes=["Items are standardized before decomposition.",
                "Marker shape and color both encode language family."],
         section=desc)


def fig_embeddings(cohort, desc):
    """t-SNE and UMAP side by side, on the standardized item matrix."""
    from sklearn.preprocessing import StandardScaler
    import umap

    X = StandardScaler().fit_transform(cohort.means_imp.values)
    fam = cohort_families(cohort)
    n = len(cohort.countries)

    perplexity = max(5, min(30, (n - 1) // 3))
    ts = TSNE(n_components=2, random_state=0, perplexity=perplexity,
              init="pca").fit_transform(X)
    um = umap.UMAP(n_components=2, random_state=0,
                   n_neighbors=min(15, n - 1)).fit_transform(X)

    st.use_style()
    fig = plt.figure(figsize=(st.PAGE_WIDTH, 3.7))
    gs = GridSpec(1, 2, figure=fig, left=0.030, right=0.985, top=0.885,
                  bottom=0.120, wspace=0.09)

    for ax, emb, title, sub in (
            (fig.add_subplot(gs[0]), ts, "t-SNE",
             f"perplexity {perplexity}, PCA initialization"),
            (fig.add_subplot(gs[1]), um, "UMAP",
             f"{min(15, n - 1)} neighbors")):
        xy = pd.DataFrame(emb, index=cohort.means_imp.index,
                          columns=["d1", "d2"])
        scatter_by_family(ax, xy, fam, label_size=4.6, size=14)
        ax.set_xticks([])
        ax.set_yticks([])
        st.tidy(ax, grid="none", spines=())
        st.panel_title(ax, title, sub=sub, pad=11)

    fig.text(0.006, 0.965, "a", fontsize=9.5, fontweight=700, color=st.INK)
    fig.text(0.512, 0.965, "b", fontsize=9.5, fontweight=700, color=st.INK)
    st.family_legend(fig, ncol=5, loc="lower center", bbox=(0.53, 0.012),
                     fontsize=6.4)

    emit(fig, f"{cohort.tag}_embeddings",
         f"Non-linear embeddings of the country mean responses. {desc}. "
         f"\\textbf{{a}}, t-SNE. \\textbf{{b}}, UMAP.",
         notes=["Axes of a non-linear embedding carry no units and distances "
                "between distant points are not meaningful; only local "
                "neighborhoods should be read.",
                "Both are computed on the standardized item matrix with a "
                "fixed random seed."],
         section=desc)


def fig_clustermap(cohort, desc):
    """Country x item heatmap with both dendrograms and two color strips.

    The 199 item labels are deliberately absent: at page width they are
    illegible, and the theme strip plus the question table carry the same
    information without pretending otherwise.
    """
    X = cohort.means_imp
    fam = cohort_families(cohort)

    row_link = linkage(X.values, method="average")
    col_link = linkage(X.values.T, method="average")
    ro = leaves_list(row_link)
    co = leaves_list(col_link)
    M = X.values[np.ix_(ro, co)]
    rows = [X.index[i] for i in ro]
    cols = [X.columns[i] for i in co]

    st.use_style()
    fig = plt.figure(figsize=(st.PAGE_WIDTH, 8.2))
    gs = GridSpec(3, 3, figure=fig,
                  width_ratios=[0.13, 0.018, 1.0],
                  height_ratios=[0.10, 0.016, 1.0],
                  left=0.075, right=0.905, top=0.950, bottom=0.098,
                  wspace=0.012, hspace=0.012)

    ax_col = fig.add_subplot(gs[0, 2])
    ax_colstrip = fig.add_subplot(gs[1, 2])
    ax_row = fig.add_subplot(gs[2, 0])
    ax_rowstrip = fig.add_subplot(gs[2, 1])
    ax_hm = fig.add_subplot(gs[2, 2])

    for ax, link, orient in ((ax_col, col_link, "top"),
                             (ax_row, row_link, "left")):
        dn = dendrogram(link, orientation=orient, no_plot=True,
                        distance_sort="descending")
        for xs, ys in zip(dn["icoord"], dn["dcoord"]):
            if orient == "left":
                ax.plot(ys, xs, color="#b9b9c2", lw=0.5)
            else:
                ax.plot(xs, ys, color="#b9b9c2", lw=0.5)
        ax.set_axis_off()
    ax_row.invert_xaxis()
    ax_row.set_ylim(0, len(rows) * 10)
    ax_col.set_xlim(0, len(cols) * 10)

    ax_rowstrip.imshow(
        np.array([[st.family_color(fam[c]) for c in rows]], dtype=object).T
        .tolist() if False else
        [[_rgb(st.family_color(fam[c]))] for c in rows],
        aspect="auto", interpolation="nearest")
    ax_colstrip.imshow(
        [[_rgb(st.THEME_COLOR.get(QUESTIONS.get(q, {}).get("category"), "#dddde3"))
          for q in cols]], aspect="auto", interpolation="nearest")
    for a in (ax_rowstrip, ax_colstrip):
        a.set_xticks([])
        a.set_yticks([])
        for sp in a.spines.values():
            sp.set_visible(False)

    # 1-5 is the nominal range, but almost every mean lies between 2 and 4;
    # stretching the ramp over the observed range is what makes the structure
    # visible rather than a wash of mid-blue.
    lo, hi = np.percentile(M, [1, 99])
    im = ax_hm.imshow(M, aspect="auto", cmap=st.SEQUENTIAL, vmin=lo, vmax=hi,
                      interpolation="nearest")
    ax_hm.set_xticks([])
    ax_hm.set_yticks(range(len(rows)))
    ax_hm.set_yticklabels(rows, fontsize=4.6)
    ax_hm.yaxis.tick_right()
    for tick, code in zip(ax_hm.get_yticklabels(), rows):
        tick.set_color(st.family_color(fam[code]))
        tick.set_fontweight(600)
    ax_hm.tick_params(axis="y", length=0, pad=1.5)
    for sp in ax_hm.spines.values():
        sp.set_visible(False)
    ax_hm.set_xlabel(f"{len(cols)} survey items, clustered", labelpad=4)

    cax = fig.add_axes([0.075, 0.056, 0.16, 0.010])
    cb = fig.colorbar(im, cax=cax, orientation="horizontal")
    cb.set_label("Mean response (1-5 scale)", fontsize=6.0, labelpad=2)
    cb.ax.tick_params(labelsize=5.6, length=2)
    cb.outline.set_visible(False)

    st.family_legend(fig, ncol=5, loc="lower center", bbox=(0.60, 0.022),
                     markers=False, fontsize=6.0)
    fig.text(0.075, 0.972, f"Country by item clustermap  \u2014  {desc}",
             fontsize=7.5, fontweight=600, color=st.INK)

    emit(fig, f"{cohort.tag}_clustermap",
         f"Clustered heatmap of country mean responses. {desc}. "
         f"Rows are societies, colored by language family; columns are the "
         f"{len(cols)} value items, colored by thematic category. Both "
         f"dimensions are ordered by average-linkage clustering on Euclidean "
         f"distance.",
         notes=["Item labels are omitted: at this width they are illegible. "
                "Column order is given by the dendrogram and item wording by "
                "the question table.",
                "The color ramp spans the 1st to 99th percentile of the "
                "observed means rather than the nominal 1-5 range.",
                "Missing country means are KNN-imputed from the two nearest "
                "societies before clustering."],
         section=desc)


def _rgb(hexcolor):
    h = hexcolor.lstrip("#")
    return [int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)]


def fig_distance_matrix(cohort, desc):
    d = cohort.distances()
    link = linkage(squareform(d.values, checks=False), method="average")
    order = leaves_list(link)
    M = d.values[np.ix_(order, order)]
    labels = [d.index[i] for i in order]
    fam = cohort_families(cohort)

    st.use_style()
    fig = plt.figure(figsize=(st.PAGE_WIDTH, 7.0))
    gs = GridSpec(2, 2, figure=fig, width_ratios=[0.16, 1.0],
                  height_ratios=[0.16, 1.0],
                  left=0.070, right=0.895, top=0.945, bottom=0.092,
                  wspace=0.015, hspace=0.015)
    ax_top = fig.add_subplot(gs[0, 1])
    ax_left = fig.add_subplot(gs[1, 0])
    ax_hm = fig.add_subplot(gs[1, 1])

    for ax, orient in ((ax_top, "top"), (ax_left, "left")):
        dn = dendrogram(link, orientation=orient, no_plot=True,
                        distance_sort="descending")
        for xs, ys in zip(dn["icoord"], dn["dcoord"]):
            if orient == "left":
                ax.plot(ys, xs, color="#b9b9c2", lw=0.5)
            else:
                ax.plot(xs, ys, color="#b9b9c2", lw=0.5)
        ax.set_axis_off()
    ax_left.invert_xaxis()
    ax_left.set_ylim(0, len(labels) * 10)
    ax_top.set_xlim(0, len(labels) * 10)

    im = ax_hm.imshow(M, cmap=st.SEQUENTIAL, interpolation="nearest")
    ax_hm.set_xticks(range(len(labels)))
    ax_hm.set_xticklabels(labels, rotation=90, fontsize=4.4)
    ax_hm.set_yticks(range(len(labels)))
    ax_hm.set_yticklabels(labels, fontsize=4.4)
    ax_hm.yaxis.tick_right()
    for ticks in (ax_hm.get_xticklabels(), ax_hm.get_yticklabels()):
        for tick in ticks:
            tick.set_color(st.family_color(fam[tick.get_text()]))
            tick.set_fontweight(600)
    ax_hm.tick_params(length=0, pad=1.5)
    for sp in ax_hm.spines.values():
        sp.set_visible(False)

    cax = fig.add_axes([0.070, 0.052, 0.16, 0.010])
    cb = fig.colorbar(im, cax=cax, orientation="horizontal")
    cb.set_label("Euclidean distance", fontsize=6.0, labelpad=3)
    cb.ax.tick_params(labelsize=5.6, length=2)
    cb.outline.set_visible(False)

    fig.text(0.070, 0.972, f"Pairwise distance between societies  \u2014  {desc}",
             fontsize=7.5, fontweight=600, color=st.INK)

    emit(fig, f"{cohort.tag}_distance_matrix",
         f"Pairwise Euclidean distance between country mean response "
         f"profiles. {desc}. Rows and columns are ordered by average-linkage "
         f"clustering; label color denotes language family.",
         notes=["Distances are computed on the 1-5 item scale without "
                "standardization."],
         section=desc)


def fig_nmf(cohort, desc):
    """k = 2, 3 and 4 in one figure: a strip, a simplex and a stacked bar."""
    fam = cohort_families(cohort)
    w2 = cohort.nmf(2)[0]["C1"]
    w3 = cohort.nmf(3)[0]
    w4 = cohort.nmf(4)[0]

    st.use_style()
    fig = plt.figure(figsize=(st.PAGE_WIDTH, 8.0))
    gs = GridSpec(3, 1, figure=fig, height_ratios=[1.0, 1.35, 0.85],
                  left=0.135, right=0.985, top=0.955, bottom=0.085,
                  hspace=0.36)
    ax_a = fig.add_subplot(gs[0])
    ax_b = fig.add_subplot(gs[1])
    ax_c = fig.add_subplot(gs[2])

    # --- k = 2 ---------------------------------------------------------
    order = [f for f in st.FAMILY_ORDER if f in set(fam)]
    texts = []
    for i, f in enumerate(order):
        yi = len(order) - 1 - i
        vals = w2[fam[fam == f].index].sort_values()
        jitter = np.linspace(-0.14, 0.14, len(vals)) if len(vals) > 1 else [0.0]
        ax_a.plot([vals.min(), vals.max()], [yi, yi], color=st.RULE, lw=0.8,
                  zorder=1, solid_capstyle="round")
        ax_a.scatter(vals.values, yi + np.array(jitter), s=14,
                     marker=st.family_marker(f), facecolor=st.family_color(f),
                     edgecolor="white", linewidth=0.4, zorder=3)
        for code, v, j in zip(vals.index, vals.values, jitter):
            texts.append(ax_a.text(v, yi + j + 0.18, code, fontsize=4.2,
                                   ha="center", va="bottom", color=st.INK_MUTED))
    adjust_text(texts, ax=ax_a, only_move={"text": "x", "static": "x"},
                expand=(1.06, 1.02), force_text=(0.08, 0.0))
    ax_a.set_yticks(range(len(order)))
    ax_a.set_yticklabels([st.family_label(f) for f in order[::-1]],
                         fontsize=6.2)
    for tick, f in zip(ax_a.get_yticklabels(), order[::-1]):
        tick.set_color(st.family_color(f))
        tick.set_fontweight(600)
    ax_a.tick_params(axis="y", length=0, pad=3)
    ax_a.set_ylim(-0.6, len(order) - 0.4)
    ax_a.set_xlim(-0.02, 1.02)
    ax_a.set_xlabel("Weight on component 1  (component 2 is the remainder)")
    st.tidy(ax_a, grid="x", spines=("bottom",))
    st.panel_title(ax_a, "Two components", sub=desc, pad=11)

    # --- k = 3 ---------------------------------------------------------
    a, b, c = w3.values[:, 0], w3.values[:, 1], w3.values[:, 2]
    s = a + b + c
    x, y = 0.5 * (2 * b + c) / s, (np.sqrt(3) / 2) * c / s
    tri = np.array([[0, 0], [1, 0], [0.5, np.sqrt(3) / 2], [0, 0]])
    ax_b.plot(tri[:, 0], tri[:, 1], color=st.RULE, lw=0.8, zorder=1)
    texts = []
    for f in st.FAMILY_ORDER:
        sel = [i for i, code in enumerate(w3.index) if fam[code] == f]
        if not sel:
            continue
        ax_b.scatter(x[sel], y[sel], s=14, marker=st.family_marker(f),
                     facecolor=st.family_color(f), edgecolor="white",
                     linewidth=0.4, zorder=3)
        for i in sel:
            texts.append(ax_b.text(x[i], y[i], w3.index[i], fontsize=4.2,
                                   color=st.INK, zorder=4,
                                   path_effects=st.halo(1.2)))
    adjust_text(texts, ax=ax_b, expand=(1.05, 1.12), force_text=(0.10, 0.18))
    for px, py, lab, va in ((0, -0.035, "component 1", "top"),
                            (1, -0.035, "component 2", "top"),
                            (0.5, np.sqrt(3) / 2 + 0.03, "component 3",
                             "bottom")):
        ax_b.text(px, py, lab, fontsize=5.8, color=st.INK_MUTED, ha="center",
                  va=va)
    ax_b.set_xlim(-0.10, 1.10)
    ax_b.set_ylim(-0.10, np.sqrt(3) / 2 + 0.10)
    ax_b.set_aspect("equal")
    ax_b.set_axis_off()
    st.panel_title(ax_b, "Three components, barycentric",
                   sub="Position gives the mix of the three latent profiles",
                   pad=11)

    # --- k = 4 ---------------------------------------------------------
    w4s = w4.loc[w4.sort_values("C1").index]
    xpos = np.arange(len(w4s))
    bottom = np.zeros(len(w4s))
    ramp = [st.SEQUENTIAL(v) for v in (0.20, 0.42, 0.64, 0.86)]
    for k, col in enumerate(w4s.columns):
        ax_c.bar(xpos, w4s[col].values, bottom=bottom, width=0.84,
                 color=ramp[k], edgecolor="white", linewidth=0.35,
                 label=f"Component {k + 1}")
        bottom += w4s[col].values
    ax_c.set_xticks(xpos)
    ax_c.set_xticklabels(w4s.index, rotation=90, fontsize=4.2)
    for tick, code in zip(ax_c.get_xticklabels(), w4s.index):
        tick.set_color(st.family_color(fam[code]))
        tick.set_fontweight(600)
    ax_c.tick_params(axis="x", length=0, pad=1.5)
    ax_c.set_xlim(-0.7, len(w4s) - 0.3)
    ax_c.set_ylim(0, 1)
    ax_c.set_ylabel("Weight")
    st.tidy(ax_c, grid="y", spines=("left",))
    leg = ax_c.legend(ncol=4, loc="lower left", bbox_to_anchor=(0.0, 1.02),
                      frameon=False, fontsize=6.0, handlelength=0.9)
    for t in leg.get_texts():
        t.set_color(st.INK)
    st.panel_title(ax_c, "Four components", pad=22)

    fig.text(0.006, 0.975, "a", fontsize=9.5, fontweight=700, color=st.INK)
    fig.text(0.006, 0.650, "b", fontsize=9.5, fontweight=700, color=st.INK)
    fig.text(0.006, 0.245, "c", fontsize=9.5, fontweight=700, color=st.INK)

    emit(fig, f"{cohort.tag}_nmf",
         f"Non-negative matrix factorization of the country mean responses. "
         f"{desc}. \\textbf{{a}}, Two components, one mark per society. "
         f"\\textbf{{b}}, Three components in barycentric coordinates. "
         f"\\textbf{{c}}, Four components, societies ordered by component 1.",
         notes=["Weights are row-normalized, so each society's components "
                "sum to one.",
                "The factorization is fitted with a fixed random seed and "
                "NNDSVDA initialization, so the component order is stable "
                "across runs but arbitrary in meaning."],
         section=desc)


def fig_society_distances(cohort, code, desc):
    st.use_style()
    fig = plt.figure(figsize=(st.COL_WIDTH * 1.5, 2.9))
    gs = GridSpec(1, 1, figure=fig, left=0.155, right=0.975, top=0.845,
                  bottom=0.145)
    ax = fig.add_subplot(gs[0])
    n = cohort.counts.median(axis=1)
    wpan.nearest_neighbors(
        ax, cohort, code, k=12,
        title=st.country_name(code, COUNTRY_INFO),
        sub=f"12 closest societies  (n = {int(n[code]):,})")
    emit(fig, f"{cohort.tag}_distances_{code}",
         f"The twelve societies closest to "
         f"{st.country_name(code, COUNTRY_INFO)} in mean response profile. "
         f"{desc}.",
         notes=["Bars are 95\\% delta-method intervals for the distance "
                "between two sample mean vectors, using each society's own "
                "sample size.",
                "Hollow marks flag societies with fewer than 200 "
                "respondents."],
         section=f"Distances from each society ({desc})")


# --------------------------------------------------------------------------- #
# family highlight, biplot, loading angles                                     #
# --------------------------------------------------------------------------- #
def fig_pca_by_family(cohort, desc):
    """One small multiple per family, each highlighting it against the rest.

    ``WVS.py`` wrote ten near-identical full-page PCAs per cohort, differing
    only in which societies were drawn large. As a facet grid the comparison
    the set exists to make -- how tightly each family sits together -- is one
    look instead of ten page turns.
    """
    scores, _, ev = cohort.pca(2)
    fam = cohort_families(cohort)
    families = [f for f in st.FAMILY_ORDER if f in set(fam)]

    st.use_style()
    ncol = 5
    nrow = int(np.ceil(len(families) / ncol))
    fig = plt.figure(figsize=(st.PAGE_WIDTH, 1.62 * nrow + 0.62))
    gs = GridSpec(nrow, ncol, figure=fig, left=0.030, right=0.988,
                  top=0.860, bottom=0.035, wspace=0.10, hspace=0.20)

    xlim = st.nice_limits(scores["PC1"], 0.09)
    ylim = st.nice_limits(scores["PC2"], 0.09)

    for k, f in enumerate(families):
        ax = fig.add_subplot(gs[k // ncol, k % ncol])
        member = fam.reindex(scores.index) == f
        ax.scatter(scores.loc[~member, "PC1"], scores.loc[~member, "PC2"],
                   s=5, marker="o", color="#dcdce3", zorder=2, linewidth=0)
        ax.scatter(scores.loc[member, "PC1"], scores.loc[member, "PC2"],
                   s=17, marker=st.family_marker(f),
                   facecolor=st.family_color(f), edgecolor="white",
                   linewidth=0.4, zorder=4)
        texts = [ax.text(r["PC1"], r["PC2"], c, fontsize=4.3, color=st.INK,
                         zorder=5, path_effects=st.halo(1.2))
                 for c, r in scores[member].iterrows()]
        adjust_text(texts, ax=ax, expand=(1.05, 1.10), force_text=(0.10, 0.16))

        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_xticks([])
        ax.set_yticks([])
        st.tidy(ax, grid="none", spines=())
        ax.set_facecolor("#fcfcfd")
        n_member = int(member.sum())
        ax.set_title(f"{st.family_label(f)}  ({n_member})", loc="left",
                     fontsize=6.6, fontweight=600,
                     color=st.family_color(f), pad=3)

    fig.text(0.030, 0.965,
             f"Where each language family sits in PC1/PC2  \u2014  {desc}",
             fontsize=7.5, fontweight=600, color=st.INK)
    fig.text(0.030, 0.930,
             f"Same decomposition in every panel; only the highlighted family "
             f"changes. PC1 {ev[0] * 100:.1f}%, PC2 {ev[1] * 100:.1f}% of "
             f"variance. Gray marks are the other societies.",
             fontsize=6.3, color=st.INK_MUTED)

    emit(fig, f"{cohort.tag}_pca_by_family",
         f"Principal components of country mean responses, with each language "
         f"family highlighted in turn. {desc}. All panels show the same "
         f"decomposition; axes are shared.",
         notes=["Panel counts give the number of societies in that family.",
                "Gray marks are societies outside the highlighted family."],
         section=desc)


def fig_biplot(cohort, desc, n_arrows=18):
    """Proper biplot: society scores behind the item loadings that drive them."""
    scores, loadings, ev = cohort.pca(2)
    fam = cohort_families(cohort)

    # Ranking by joint magnitude picks a degenerate set: the "Confidence in
    # institutions" battery has the longest vectors and they all point the
    # same way. Taking the extremes along each axis separately -- which is
    # what the notebook's is_top_5_component did -- spreads the arrows over
    # the four directions the reader is actually comparing.
    per_side = max(3, n_arrows // 4)
    top = list(dict.fromkeys(
        list(loadings["PC1"].nlargest(per_side).index)
        + list(loadings["PC1"].nsmallest(per_side).index)
        + list(loadings["PC2"].nlargest(per_side).index)
        + list(loadings["PC2"].nsmallest(per_side).index)))
    mag = np.hypot(loadings["PC1"], loadings["PC2"])

    st.use_style()
    fig = plt.figure(figsize=(st.PAGE_WIDTH, 5.2))
    gs = GridSpec(1, 1, figure=fig, left=0.075, right=0.985, top=0.895,
                  bottom=0.155)
    ax = fig.add_subplot(gs[0])

    # scores, rescaled onto the loading plane so both fit one set of axes
    k = 0.85 * mag.max() / np.abs(scores.values).max()
    ax.scatter(scores["PC1"] * k, scores["PC2"] * k, s=9,
               c=[st.family_color(fam[c]) for c in scores.index],
               alpha=0.55, linewidth=0, zorder=2)
    for c, r in scores.iterrows():
        ax.text(r["PC1"] * k, r["PC2"] * k, c, fontsize=3.9,
                color=st.INK_FAINT, ha="center", va="center", zorder=3)

    texts = []
    for q in top:
        x, y = loadings.loc[q, "PC1"], loadings.loc[q, "PC2"]
        color = st.THEME_COLOR.get(QUESTIONS.get(q, {}).get("category"),
                                   st.INK_FAINT)
        ax.annotate("", xy=(x, y), xytext=(0, 0),
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=1.0,
                                    shrinkA=0, shrinkB=0,
                                    mutation_scale=7), zorder=5)
        texts.append(ax.text(x, y, _arrow_label(q), fontsize=5.2,
                             color=color, fontweight=600, zorder=6,
                             ha="center", va="center",
                             path_effects=st.halo(2.0)))
    adjust_text(texts, ax=ax, expand=(1.25, 1.45), force_text=(0.55, 0.75),
                force_static=(0.25, 0.35), max_move=40,
                arrowprops=dict(arrowstyle="-", color=st.INK_FAINT, lw=0.3,
                                shrinkA=0.5, shrinkB=1.5))

    lx = np.abs(loadings.loc[top, "PC1"]).max() * 1.85
    ly = np.abs(loadings.loc[top, "PC2"]).max() * 1.28
    ax.set_xlim(-lx, lx)
    ax.set_ylim(-ly, ly)
    ax.axhline(0, color=st.RULE, lw=0.5, zorder=0)
    ax.axvline(0, color=st.RULE, lw=0.5, zorder=0)
    ax.set_xlabel(f"PC1  ({ev[0] * 100:.1f}% of variance)")
    ax.set_ylabel(f"PC2  ({ev[1] * 100:.1f}%)")
    st.tidy(ax, grid="both", spines=("left", "bottom"))
    st.panel_title(ax, "Biplot: societies and the items that place them",
                   sub=f"{desc}. Arrows: the {per_side} items at each end of "
                       f"PC1 and of PC2", pad=11)

    handles = [plt.Line2D([], [], marker="s", linestyle="none", markersize=3.6,
                          markerfacecolor=st.THEME_COLOR[t],
                          markeredgecolor="none", label=st.theme_label(t))
               for t in st.THEME_ORDER]
    leg = ax.legend(handles=handles, loc="upper left", ncol=6, frameon=False,
                    fontsize=6.2, bbox_to_anchor=(0.0, -0.075),
                    handletextpad=0.4, columnspacing=1.2)
    for t in leg.get_texts():
        t.set_color(st.INK)

    emit(fig, f"{cohort.tag}_biplot",
         f"Biplot of the first two principal components. {desc}. Society "
         f"scores are drawn faintly and rescaled onto the loading plane; "
         f"arrows give the {per_side} items at each end of PC1 and of PC2, "
         f"colored by thematic category.",
         notes=["Scores are multiplied by a constant so that both can share "
                "one pair of axes; only the direction and relative length of "
                "the arrows should be read.",
                "The sign of a principal component is arbitrary, so left and "
                "right carry no absolute meaning."],
         section=desc)


def fig_loading_angles(cohort, desc, bins=18):
    """Where each theme's items point in the PC1/PC2 plane.

    Replaces the six separate rose diagrams ``WVS.py`` wrote per cohort. A
    rose alone cannot be read quantitatively, so each panel carries the mean
    resultant vector -- the standard circular measure of how concentrated a
    set of directions is -- as a line and a number.
    """
    _, loadings, _ = cohort.pca(2)
    ang = np.arctan2(loadings["PC2"], loadings["PC1"]) % (2 * np.pi)
    theme = pd.Series({q: QUESTIONS.get(q, {}).get("category") for q in ang.index})

    groups = [("All items", ang, st.INK_MUTED)]
    for t in st.THEME_ORDER:
        sel = ang[theme == t]
        if len(sel) >= 3:
            groups.append((st.theme_label(t), sel, st.THEME_COLOR[t]))

    st.use_style()
    ncol = 4
    nrow = int(np.ceil(len(groups) / ncol))
    fig = plt.figure(figsize=(st.PAGE_WIDTH, 2.05 * nrow + 1.35))
    gs = GridSpec(nrow, ncol, figure=fig, left=0.035, right=0.980,
                  top=0.775, bottom=0.075, wspace=0.34, hspace=0.62)

    edges = np.linspace(0, 2 * np.pi, bins + 1)
    for k, (name, a, color) in enumerate(groups):
        ax = fig.add_subplot(gs[k // ncol, k % ncol], projection="polar")
        counts, _ = np.histogram(a.values, bins=edges)
        ax.bar(edges[:-1], counts, width=np.diff(edges), align="edge",
               color=color, alpha=0.80, edgecolor="white", linewidth=0.5,
               zorder=3)

        # mean resultant vector: direction and concentration in one mark
        R = np.abs(np.mean(np.exp(1j * a.values)))
        mu = np.angle(np.mean(np.exp(1j * a.values))) % (2 * np.pi)
        ax.plot([mu, mu], [0, counts.max() * R], color=st.INK, lw=1.3,
                zorder=5, solid_capstyle="round")

        ax.set_theta_zero_location("E")
        ax.set_xticks(np.linspace(0, 2 * np.pi, 4, endpoint=False))
        ax.set_xticklabels(["0", "90", "180", "270"], fontsize=4.8,
                           color=st.INK_FAINT)
        ax.tick_params(axis="x", pad=-2)
        ax.set_yticklabels([])
        ax.grid(color=st.RULE, lw=0.4)
        ax.spines["polar"].set_color(st.RULE)
        ax.spines["polar"].set_linewidth(0.6)
        ax.set_title(f"{name}\n{len(a)} items,  R = {R:.2f}", loc="center",
                     fontsize=6.3, fontweight=600, color=st.INK, pad=9)

    fig.text(0.035, 0.965,
             f"Direction of item loadings in the PC1/PC2 plane  \u2014  {desc}",
             fontsize=7.5, fontweight=600, color=st.INK, va="top")
    fig.text(0.035, 0.925,
             "Angle is the direction of an item's loading; radial bars count "
             "items per sector.\nThe black line is the mean resultant vector "
             "and R its length: R near 1 means the theme's items point one "
             "way,\nR near 0 that they are spread evenly around the circle.",
             fontsize=6.3, color=st.INK_MUTED, va="top", linespacing=1.55)

    emit(fig, f"{cohort.tag}_loading_angles",
         f"Angular distribution of item loadings in the PC1/PC2 plane, by "
         f"thematic category. {desc}. The black line is the mean resultant "
         f"vector; R is its length, a standard measure of circular "
         f"concentration.",
         notes=["Principal component signs are arbitrary, so the absolute "
                "direction of a sector carries no meaning; only the "
                "concentration within a theme, and the angle between themes, "
                "are interpretable.",
                "Bars count items, not respondents."],
         section=desc)


# --------------------------------------------------------------------------- #
# English proficiency                                                          #
# --------------------------------------------------------------------------- #
def _proficiency_frame(cohort, focus):
    """EF proficiency score against distance from ``focus``, for one cohort."""
    d = cohort.distances().loc[focus]
    rows = []
    for code, info in ENGLISH_PROFICIENCY.items():
        if code in d.index and code != focus:
            rows.append((code, float(info["score"]), float(d[code])))
    return pd.DataFrame(rows, columns=["code", "score", "distance"]).set_index("code")


def fig_proficiency_scatter(cohort, focus, desc):
    from scipy import stats as sstats

    df = _proficiency_frame(cohort, focus)
    if len(df) < 5:
        return
    r, p = sstats.pearsonr(df["score"], df["distance"])

    st.use_style()
    fig = plt.figure(figsize=(st.COL_WIDTH * 1.55, 2.9))
    gs = GridSpec(1, 1, figure=fig, left=0.115, right=0.980, top=0.825,
                  bottom=0.155)
    ax = fig.add_subplot(gs[0])

    b, a = np.polyfit(df["score"], df["distance"], 1)
    xs = np.linspace(df["score"].min(), df["score"].max(), 50)
    resid = df["distance"] - (a + b * df["score"])
    se = resid.std(ddof=2) * np.sqrt(
        1 / len(df) + (xs - df["score"].mean()) ** 2
        / ((df["score"] - df["score"].mean()) ** 2).sum())
    tcrit = sstats.t.ppf(0.975, len(df) - 2)
    ax.fill_between(xs, a + b * xs - tcrit * se, a + b * xs + tcrit * se,
                    color=st.RULE, alpha=0.55, linewidth=0, zorder=1)
    ax.plot(xs, a + b * xs, color=st.INK_FAINT, lw=1.0, zorder=2,
            linestyle=(0, (4, 2)))

    for code, row in df.iterrows():
        f = st._canon(COUNTRY_INFO[code][3])
        ax.scatter(row["score"], row["distance"], s=13,
                   marker=st.family_marker(f), facecolor=st.family_color(f),
                   edgecolor="white", linewidth=0.35, zorder=3)

    ptxt = "p < 0.001" if p < 1e-3 else f"p = {p:.3f}"
    ax.text(0.03, 0.95, f"r = {r:.2f}   {ptxt}   n = {len(df)}",
            transform=ax.transAxes, fontsize=6.2, color=st.INK, va="top",
            fontweight=600)
    ax.set_xlabel("EF English Proficiency Index score")
    ax.set_ylabel(f"Distance from {focus}")
    st.tidy(ax, grid="both", spines=("left", "bottom"))
    st.panel_title(ax, f"English proficiency and distance from "
                       f"{st.country_name(focus, COUNTRY_INFO)}",
                   sub=desc, pad=11)

    emit(fig, f"{cohort.tag}_engprof_{focus}",
         f"EF English Proficiency Index score against cultural distance from "
         f"{st.country_name(focus, COUNTRY_INFO)}. {desc}. The dashed line is "
         f"the least-squares fit with a 95\\% confidence band.",
         notes=["The proficiency index is a country-level measure from a "
                "self-selected test-taker sample and is not part of the "
                "World Values Survey.",
                "One such fit is computed for every society in the cohort; "
                "the p-value is not corrected for those comparisons."],
         section=f"English proficiency ({desc})")


def fig_proficiency_summary(cohort, desc):
    from scipy import stats as sstats

    rows = []
    for focus in cohort.countries:
        df = _proficiency_frame(cohort, focus)
        if len(df) >= 5:
            r, p = sstats.pearsonr(df["score"], df["distance"])
            rows.append((focus, r, p, len(df)))
    if not rows:
        return
    tab = (pd.DataFrame(rows, columns=["code", "r", "p", "n"])
           .set_index("code").sort_values("r"))
    fam = cohort_families(cohort)

    st.use_style()
    fig = plt.figure(figsize=(st.PAGE_WIDTH, 0.105 * len(tab) + 1.5))
    gs = GridSpec(1, 1, figure=fig, left=0.065, right=0.985, top=0.885,
                  bottom=0.105)
    ax = fig.add_subplot(gs[0])

    y = np.arange(len(tab))
    for yi, (code, row) in zip(y, tab.iterrows()):
        color = st.family_color(fam[code])
        ax.plot([0, row["r"]], [yi, yi], color=color, lw=0.9, alpha=0.55,
                zorder=2, solid_capstyle="round")
        ax.plot([row["r"]], [yi], marker=st.family_marker(fam[code]),
                markersize=3.8, markerfacecolor=color,
                markeredgecolor="white", markeredgewidth=0.4,
                linestyle="none", zorder=3)
        if row["p"] >= 0.05:
            ax.plot([row["r"]], [yi], marker="o", markersize=6.5,
                    markerfacecolor="none", markeredgecolor=st.INK_FAINT,
                    markeredgewidth=0.5, linestyle="none", zorder=4)
    ax.axvline(0, color=st.INK_FAINT, lw=0.7, zorder=1)

    ax.set_yticks(y)
    ax.set_yticklabels(tab.index, fontsize=5.0)
    for tick, code in zip(ax.get_yticklabels(), tab.index):
        tick.set_color(st.family_color(fam[code]))
        tick.set_fontweight(600)
    ax.tick_params(axis="y", length=0, pad=2)
    ax.set_ylim(-0.8, len(tab) - 0.2)
    ax.set_xlabel("Pearson correlation between EF proficiency score and "
                  "distance from that society")
    st.tidy(ax, grid="x", spines=("bottom",))
    st.panel_title(ax, "English proficiency against cultural distance, "
                       "by focal society",
                   sub=f"{desc}. Hollow rings mark p >= 0.05, uncorrected",
                   pad=11)

    emit(fig, f"{cohort.tag}_english_proficiency",
         f"Pearson correlation between the EF English Proficiency Index and "
         f"cultural distance from each society in turn. {desc}. Societies are "
         f"ordered by correlation; hollow rings mark correlations with "
         f"p $\\geq$ 0.05.",
         notes=["The proficiency index is a country-level measure from a "
                "self-selected test-taker sample and is not part of the "
                "World Values Survey.",
                f"{len(tab)} correlations are computed here and the p-values "
                f"are not corrected for multiple comparisons."],
         section=desc)


# --------------------------------------------------------------------------- #
# cohort contrasts                                                             #
# --------------------------------------------------------------------------- #
def fig_contrast(cohort_a, cohort_b, label_a, label_b, name, top_k=6):
    """Within-society distance between two subpopulations, sorted.

    ``WVS.py`` drew this as a plain bar chart and wrote the most-divergent
    items to a separate table. Naming the items on the extremes puts the two
    together.
    """
    shared = [c for c in cohort_a.countries if c in set(cohort_b.countries)]
    A = cohort_a.means_imp.loc[shared]
    B = cohort_b.means_imp.loc[shared]
    dist = pd.Series(np.linalg.norm(A.values - B.values, axis=1),
                     index=shared).sort_values()
    fam = pd.Series({c: st._canon(COUNTRY_INFO[c][3]) for c in shared})
    gap = (A - B).abs()

    st.use_style()
    fig = plt.figure(figsize=(st.PAGE_WIDTH, 4.6))
    gs = GridSpec(2, 1, figure=fig, height_ratios=[1.0, 0.62],
                  left=0.065, right=0.985, top=0.900, bottom=0.090,
                  hspace=0.68)
    ax = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    ax2.set_position([0.245, 0.090, 0.740, 0.215])   # room for item labels

    x = np.arange(len(dist))
    ax.bar(x, dist.values, width=0.78,
           color=[st.family_color(fam[c]) for c in dist.index],
           edgecolor="white", linewidth=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels(dist.index, rotation=90, fontsize=4.6)
    for tick, code in zip(ax.get_xticklabels(), dist.index):
        tick.set_color(st.family_color(fam[code]))
        tick.set_fontweight(600)
    ax.tick_params(axis="x", length=0, pad=1.5)
    ax.set_xlim(-0.7, len(dist) - 0.3)
    ax.set_ylabel(f"Distance, {label_a} vs {label_b}")
    st.tidy(ax, grid="y", spines=("left",))
    st.panel_title(ax, f"How far apart {label_a} and {label_b} sit, "
                       f"within each society",
                   sub="Euclidean distance between the two mean response "
                       "profiles of the same society", pad=11)

    # which items drive the biggest gaps
    overall = gap.mean().sort_values(ascending=False)[:top_k][::-1]
    yy = np.arange(len(overall))
    ax2.barh(yy, overall.values, height=0.6,
             color=[st.THEME_COLOR.get(QUESTIONS.get(q, {}).get("category"),
                                       st.INK_FAINT) for q in overall.index],
             edgecolor="white", linewidth=0.5)
    ax2.set_yticks(yy)
    ax2.set_yticklabels([short_question(q) for q in overall.index],
                        fontsize=5.6, color=st.INK)
    ax2.tick_params(axis="y", length=0, pad=2)
    ax2.set_xlabel(f"Mean absolute gap across societies (1-5 scale)")
    st.tidy(ax2, grid="x", spines=("bottom",))
    st.panel_title(ax2, "Items with the largest gap",
                   sub=f"Averaged over all {len(shared)} societies", pad=11)

    fig.text(0.006, 0.960, "a", fontsize=9.5, fontweight=700, color=st.INK)
    fig.text(0.006, 0.360, "b", fontsize=9.5, fontweight=700, color=st.INK)

    emit(fig, name,
         f"Within-society distance between the {label_a} and {label_b} mean "
         f"response profiles. \\textbf{{a}}, Euclidean distance per society, "
         f"sorted, colored by language family. \\textbf{{b}}, The {top_k} "
         f"items whose mean absolute gap between the two groups is largest, "
         f"averaged across societies.",
         notes=["Both profiles are computed on the same item set, so the "
                "distance is comparable across societies.",
                "Splitting the sample halves the respondents behind every "
                "country mean; societies with small samples move more, which "
                "is why Andorra (n = 35 before splitting) sits at the "
                "extreme.",
                "Panel b is reported in raw item units. Q171 and Q172 are "
                "scored 1-7 and 1-8 rather than 1-5, so their gaps are not "
                "directly comparable with the rest; see the methods note on "
                "item rescaling."],
         section="Contrasts between subpopulations")


# --------------------------------------------------------------------------- #
# driver                                                                       #
# --------------------------------------------------------------------------- #
#: Cohort prefixes used by ``WVS.py`` in ``WVS_v6/``, for cross-referencing.
LEGACY_PREFIX = {
    "unfiltered": "unfiltered", "filtered": "filtered",
    "cankaz": "CANKAZ_filter", "uspolitics": "USAPolitics_filter",
    "secondlanguage": "SecondLanguageFilter", "male": "male",
    "female": "female", "age_16_29": "16-29_Filter",
    "age_30_49": "30-49_Filter", "age_50plus": "50andOver_Filter",
    "cde_16_29": "16-29_CADUSA", "cde_30_49": "30-49_CADUSA",
    "cde_50plus": "50andOver_CADUSA",
}


def write_name_map():
    """Old WVS_v6/ prefix -> new WVS_v6_new/ prefix, so the two line up."""
    lines = ["# WVS_v6/ prefix -> WVS_v6_new/ prefix",
             "# e.g. WVS_v6/CANKAZ_filter_distances_sorted_THA.png",
             "#   ->  WVS_v6_new/cankaz_distances_THA.pdf", ""]
    for tag, legacy in LEGACY_PREFIX.items():
        lines.append(f"{legacy:24s} {tag}")
    lines += ["", "# figure-type mapping (several old types are consolidated)",
              "clustermap_imp / clustermaplang_imp / heatmap / rawheatmap"
              "  ->  <tag>_clustermap",
              "nmf_c2_imp / nmf_c3_imp / nmf_c3_triangle / nmf_c4_imp"
              "  ->  <tag>_nmf",
              "tsne_imp / umap_1_2_imp                    ->  <tag>_embeddings",
              "nmf_pairwise_dendrogram_imp                ->  <tag>_distance_matrix",
              "pca_pc1_pc2_imp                            ->  <tag>_pca",
              "distances_sorted_<ISO>                     ->  <tag>_distances_<ISO>"]
    (OUT / "cohort_name_map.txt").write_text("\n".join(lines) + "\n")


def write_index():
    lines = ["% Auto-generated by figures/make_supplement.py.",
             "% \\input this from the supplement preamble.", ""]
    seen = set()
    for texfile, section in _INDEX:
        if section and section not in seen:
            seen.add(section)
            lines += ["", f"\\subsection{{{section}}}"]
        lines.append(f"\\input{{WVS_v6_new/{texfile}}}")
    (OUT / "supplement_figures.tex").write_text("\n".join(lines) + "\n")
    print(f"\nwrote {OUT / 'supplement_figures.tex'} "
          f"({len(_INDEX)} figures)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true",
                    help="skip the per-society distance charts")
    ap.add_argument("--only", nargs="*", help="restrict to these cohort tags")
    args = ap.parse_args()

    tags = args.only or list(wp.COHORT_SPECS)
    cs = wp.cohorts(tags=tags)

    for tag in tags:
        cohort = cs[tag]
        desc = wp.COHORT_SPECS[tag][2]
        print(f"\n{tag}  ({len(cohort.countries)} societies)")
        fig_pca(cohort, desc)
        fig_pca_by_family(cohort, desc)
        fig_biplot(cohort, desc)
        fig_loading_angles(cohort, desc)
        fig_clustermap(cohort, desc)
        fig_distance_matrix(cohort, desc)
        fig_nmf(cohort, desc)
        fig_embeddings(cohort, desc)
        fig_proficiency_summary(cohort, desc)

    if not args.quick:
        for tag in tags:
            cohort = cs[tag]
            desc = wp.COHORT_SPECS[tag][2]
            print(f"\n{tag}: per-society distance charts "
                  f"({len(cohort.countries)})")
            for code in cohort.countries:
                fig_society_distances(cohort, code, desc)
            print(f"{tag}: English-proficiency scatters")
            for code in cohort.countries:
                fig_proficiency_scatter(cohort, code, desc)

    if not args.only:
        print("\ncohort contrasts")
        fig_contrast(cs["male"], cs["female"], "men", "women",
                     "contrast_male_female")
        fig_contrast(cs["age_16_29"], cs["age_50plus"], "16-29", "50 and over",
                     "contrast_age_young_old")

    write_index()
    write_name_map()


if __name__ == "__main__":
    main()
