"""Shared visual system for the WVS paper figures.

One place defines type, color, spacing and the panel-label convention, so every
panel in every figure is drawn to the same rules.  Figures are composed natively
in matplotlib (one canvas, one gridspec) rather than stitched from PNGs, which
is what makes the panel sizes, fonts and baselines agree.

Color
------
``FAMILY_COLOR`` is a 10-slot categorical palette derived by search rather than
by eye.  It clears, on *all* 66 pairs (not merely adjacent ones):

    worst CVD dE 8.5 (protan/deutan, Machado 2009 severity 1.0; gate 8)
    worst normal-vision dE 16.0 (gate 15)
    OKLCH L within 0.43-0.77, C >= 0.10 for every slot

Three slots sit below 3:1 contrast against the page, so every mark that uses
them also carries a visible text label -- the "relief" requirement.  Hue
assignment deliberately tracks the original notebook palette (Anglosphere red,
Latin gold, Semitic green, Turkic purple, ...) so the figures stay legible to
anyone who has read the earlier drafts.
"""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

# --------------------------------------------------------------------------- #
# color                                                                       #
# --------------------------------------------------------------------------- #
FAMILY_COLOR = {
    "Anglosphere":      "#df5038",   # vermilion
    "Indo-Iranian":     "#90442c",   # umber
    "Latin":            "#d39c43",   # ochre
    "EastAsia":         "#4bc08e",   # mint
    "Semitic":          "#1c8b6e",   # deep teal-green
    "Germanic":         "#3a85d4",   # blue
    "SubSaharanAfrica": "#2147db",   # deep blue
    "Other":            "#b198e0",   # lavender
    "Turkic":           "#7c37a1",   # purple
    "Slavic":           "#de45aa",   # magenta
}

#: Marker shapes give every family a second, color-independent channel.
FAMILY_MARKER = {
    "Anglosphere": "o", "Indo-Iranian": "s", "Latin": "^", "EastAsia": "D",
    "Semitic": "v", "Germanic": "P", "SubSaharanAfrica": "X",
    "Other": "<", "Turkic": ">", "Slavic": "p",
}

#: Display order: reading order for legends, fixed everywhere.
FAMILY_ORDER = ["Anglosphere", "Germanic", "Latin", "Slavic", "Turkic",
                "Semitic", "Indo-Iranian", "EastAsia", "SubSaharanAfrica",
                "Other"]

#: ``WVS.py`` spells this "Semetic"; corrected for publication.
FAMILY_LABEL = {
    "Semitic": "Semitic",
    "EastAsia": "East Asian",
    "SubSaharanAfrica": "Sub-Saharan Africa",
    "Indo-Iranian": "Indo-Iranian",
}

THEME_COLOR = {
    "Morality":   "#df5038",
    "Politics":   "#3a85d4",
    "Religion":   "#1c8b6e",
    "Economical": "#d39c43",
    "Security":   "#7c37a1",
    "Gender":     "#de45aa",
}
THEME_LABEL = {"Economical": "Economic"}
THEME_ORDER = ["Morality", "Politics", "Economical", "Religion",
               "Gender", "Security"]

# ink
INK = "#16161a"          # primary text
INK_MUTED = "#5c5c66"    # secondary text, axis labels
INK_FAINT = "#9a9aa4"    # tick labels, annotations
RULE = "#d8d8dc"         # grid / spines
SURFACE = "#ffffff"

#: single-hue sequential ramp for magnitude (blue, light -> dark)
SEQUENTIAL = mpl.colors.LinearSegmentedColormap.from_list(
    "wvs_seq", ["#e3edfa", "#b9d3f2", "#7fadea", "#3a85d4", "#1f5ba3", "#123a6d"])

#: diverging ramp, warm/cool poles about a neutral gray midpoint
DIVERGING = mpl.colors.LinearSegmentedColormap.from_list(
    "wvs_div", ["#123a6d", "#3a85d4", "#a9c6e8", "#eeeeef",
                "#eab6a4", "#df5038", "#8e2a17"])


def family_color(fam: str) -> str:
    return FAMILY_COLOR.get(_canon(fam), INK_FAINT)


def family_marker(fam: str) -> str:
    return FAMILY_MARKER.get(_canon(fam), "o")


def family_label(fam: str) -> str:
    fam = _canon(fam)
    return FAMILY_LABEL.get(fam, fam)


def _canon(fam: str) -> str:
    """Fold the notebook's spellings onto the published family names."""
    return {"Semetic": "Semitic", "Isolate": "Other"}.get(fam, fam)


def theme_label(theme: str) -> str:
    return THEME_LABEL.get(theme, theme)


# --------------------------------------------------------------------------- #
# type + rc                                                                    #
# --------------------------------------------------------------------------- #
BASE_FONT = ["Lato", "TeX Gyre Heros", "Carlito", "DejaVu Sans"]
MONO_FONT = ["Ubuntu Mono", "DejaVu Sans Mono"]

#: journal single/double column widths in inches
COL_WIDTH = 3.46
PAGE_WIDTH = 7.20


def use_style():
    """Install the paper's rcParams.  Call once per figure script."""
    mpl.rcParams.update({
        "figure.dpi": 130,
        "savefig.dpi": 400,
        "savefig.bbox": None,          # layouts here are explicit, not cropped
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,

        "font.family": "sans-serif",
        "font.sans-serif": BASE_FONT,
        "font.size": 7.0,
        "axes.titlesize": 7.5,
        "axes.labelsize": 7.0,
        "xtick.labelsize": 6.2,
        "ytick.labelsize": 6.2,
        "legend.fontsize": 6.5,

        "axes.edgecolor": RULE,
        "axes.labelcolor": INK_MUTED,
        "axes.titlecolor": INK,
        "axes.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titlelocation": "left",
        "axes.titlepad": 4.0,
        "axes.labelpad": 2.5,
        "axes.axisbelow": True,

        "grid.color": RULE,
        "grid.linewidth": 0.5,
        "grid.alpha": 0.9,

        "xtick.color": INK_FAINT,
        "ytick.color": INK_FAINT,
        "xtick.labelcolor": INK_MUTED,
        "ytick.labelcolor": INK_MUTED,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "xtick.direction": "out",
        "ytick.direction": "out",

        "legend.frameon": False,
        "legend.handlelength": 1.0,
        "legend.handletextpad": 0.5,
        "legend.labelspacing": 0.35,
        "legend.columnspacing": 1.1,
        "legend.borderpad": 0.0,

        "lines.linewidth": 1.1,
        "lines.markersize": 3.4,
        "patch.linewidth": 0.5,

        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


# --------------------------------------------------------------------------- #
# panel furniture                                                              #
# --------------------------------------------------------------------------- #
def panel_label(ax, letter, dx=-0.055, dy=1.045, size=9.5):
    """Bold panel letter in figure-consistent position and weight.

    Placed in axes coordinates so every panel's letter sits the same distance
    from its own plotting rectangle -- the thing the stitched-PNG figures could
    not do.
    """
    ax.text(dx, dy, letter, transform=ax.transAxes, fontsize=size,
            fontweight=700, color=INK, ha="left", va="baseline")


def panel_title(ax, text, sub=None, pad=None):
    """Left-aligned title with an optional lighter subtitle underneath."""
    ax.set_title(text, loc="left", fontsize=7.5, fontweight=600,
                 color=INK, pad=(pad if pad is not None else (12 if sub else 4)))
    if sub:
        ax.text(0, 1.012, sub, transform=ax.transAxes, fontsize=6.4,
                color=INK_MUTED, ha="left", va="bottom")


def tidy(ax, grid="y", spines=("left", "bottom")):
    """Recessive grid, only the spines that carry meaning."""
    for name, sp in ax.spines.items():
        sp.set_visible(name in spines)
    if grid in ("y", "both"):
        ax.yaxis.grid(True, lw=0.5, color=RULE)
    if grid in ("x", "both"):
        ax.xaxis.grid(True, lw=0.5, color=RULE)
    if grid in ("y", "none"):
        ax.xaxis.grid(False)
    if grid in ("x", "none"):
        ax.yaxis.grid(False)
    return ax


def family_legend(fig_or_ax, families=None, ncol=5, loc="lower center",
                  bbox=(0.5, 0.0), markers=True, title=None, fontsize=6.6):
    """The one legend shared by every figure that colors by language family."""
    from matplotlib.lines import Line2D
    families = families or FAMILY_ORDER
    handles = [
        Line2D([], [], linestyle="none",
               marker=(family_marker(f) if markers else "s"),
               markersize=4.0, markerfacecolor=family_color(f),
               markeredgecolor="white", markeredgewidth=0.4,
               label=family_label(f))
        for f in families
    ]
    leg = fig_or_ax.legend(handles=handles, loc=loc, bbox_to_anchor=bbox,
                           ncol=ncol, frameon=False, fontsize=fontsize,
                           handletextpad=0.4, columnspacing=1.2,
                           labelspacing=0.4, title=title,
                           borderaxespad=0.0)
    if title:
        leg.get_title().set_fontsize(fontsize)
        leg.get_title().set_color(INK_MUTED)
    for txt in leg.get_texts():
        txt.set_color(INK)
    return leg


def country_name(code, info):
    """Human-readable country name (the notebook stores them underscored)."""
    if code not in info:
        return code
    return info[code][1].replace("_", " ")


def save_caption_notes(png_path, notes):
    """Write the figure's provenance and caveat lines to a sidecar file.

    These used to be set in small type at the bottom-left of the canvas.  They
    belong in the caption instead: the typesetter sets them at the caption's own
    size, and they reflow with the column rather than being frozen into the
    artwork at whatever size the figure happens to be scaled to.

    ``FigureN.png`` gets ``FigureN.png_add_to_caption``, holding one sentence
    per line, to be appended to the caption at build time.
    """
    from pathlib import Path

    path = Path(str(png_path) + "_add_to_caption")
    text = "\n".join(n.strip() for n in notes if n and n.strip())
    path.write_text(text + "\n" if text else "")
    return path


def halo(width=1.6, color="white"):
    """Path effect that keeps text legible over dense marks."""
    import matplotlib.patheffects as pe
    return [pe.withStroke(linewidth=width, foreground=color)]


def nice_limits(values, pad=0.06):
    lo, hi = float(np.nanmin(values)), float(np.nanmax(values))
    span = hi - lo
    return lo - pad * span, hi + pad * span
