# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A re-analysis of World Values Survey wave 7 (v6.0, 97,220 respondents, 66
societies) asking whether a society's values track its **language family**
rather than its geography or religion. The motivating case, in `paper`, is
Morocco: does it pattern with other Africans, other Muslims, or other
Arabic-speakers?

There is no application and no test suite. The deliverable is a set of paper
figures plus their LaTeX includes.

## Environment

Everything runs under `~/pyenv`, **not** the system Python:

```bash
~/pyenv/bin/python figures/make_all.py
```

`pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `scipy`,
`geopandas`, `adjustText`, `umap`, `ternary` are installed there. There is no
parquet engine — the cache is pickle.

## Commands

Run from `figures/`:

```bash
~/pyenv/bin/python make_all.py              # the 5 main-text figures + .tex includes (~1 min)
~/pyenv/bin/python make_supplement.py       # 1,869 supplementary figures -> ../WVS_v6_new/ (~40 min)
~/pyenv/bin/python make_supplement.py --quick              # skip the 1,750 per-society charts
~/pyenv/bin/python make_supplement.py --only filtered      # one cohort, for iterating
~/pyenv/bin/python fig3_nmf.py              # rebuild a single main figure
FORCE=1 ~/pyenv/bin/python wvs_pipeline.py  # re-derive matrices from the raw CSV (~3 min)
```

`make_supplement.py` writes ~7,500 files and takes tens of minutes; run it in
the background and poll, never in a foreground call that will time out.

## Architecture

### Two generations of code

`WVS.py` (4,183 lines) is the original notebook export and is **not runnable**
— 30 `get_ipython()` / `plt.show()` calls and four hardcoded
`/home/gabriel/projects/WVS/` paths that do not match this checkout. It is
kept as the reference for what the analysis did. Do not try to run or repair
it; the pipeline was ported instead.

`figures/` is the working code. Read it in this order:

| file | role |
|---|---|
| `wvs_pipeline.py` | the analysis. Loads, filters, rescales to 1-5, aggregates to country means, KNN-imputes, then PCA / NMF / distances. Caches to `cache/*.pkl` |
| `wvs_meta.py` | country and question tables, generated once by `ast.literal_eval` on `WVS.py`'s dicts so they cannot drift by transcription. **Do not hand-edit** |
| `wvs_style.py` | the single visual system: palette, type, spacing, panel furniture |
| `wvs_panels.py` | panel builders shared by more than one figure |
| `figN_*.py` | one script per main-text figure |
| `make_supplement.py` | the supplementary set |

### The `Cohort` abstraction

Everything downstream hangs off `wvs_pipeline.Cohort` — one analysed dataset
(a filter applied to the raw survey), holding `means`, `variances`, `counts`,
`means_imp`, `var_imp`, with `.pca()`, `.nmf(k)`, `.distances()` and
`.distance_ci()` on top. `COHORT_SPECS` declares all 13 as
`tag -> (parent, selector, description)`; `cohorts()` builds or loads them.
Adding a cohort means adding one entry there, nothing else.

### Figures are composed natively, never stitched

The superseded approach (`make_figures.ipynb` with PIL, `make_figure*.R` with
magick) pasted finished PNGs onto a canvas — that is what produced the
mismatched panel letters and the 350 MB supplement. Each figure is now a
single matplotlib canvas with one `GridSpec`, saved as vector PDF. Do not
reintroduce raster compositing.

### Output conventions

Every figure emits four files: `X.pdf` (for LaTeX), `X.png` (for review),
`X.tex` (the `figure` environment and main caption), and
`X.png_add_to_caption` (provenance and caveat sentences, one per line).

The sidecar exists because those sentences used to be set in 5.6 pt type at
the bottom of the canvas; they belong in the caption, where the typesetter
sets them at caption size. **Nothing appends them automatically** — the LaTeX
build must concatenate them.

Main-text figures use **uppercase** panel letters (A, B, C); supplementary
figures use lowercase. Panel references in captions and sidecars must match.

## Reproducibility contract

The port reproduces the published numbers exactly: PC1/PC2 explain
23.88% / 13.48% on the `filtered` cohort and 24.56% / 13.17% on `uspolitics`,
matching the titles of the pre-existing `Figure2.png` and `Figure5.png` at the
repo root. **If a change moves those numbers, it is a bug** unless the change
is deliberate and documented in `wvs_pipeline.DEVIATIONS`.

Two things are computed differently from `WVS.py`, both opt-in and both
documented there:

- `distance_se(..., mode="delta")` uses the delta-method SE with each
  society's own sample size. `mode="notebook"` reproduces the old
  `sqrt(sum(var)) / sqrt(len(df))`, which used the whole sample's n.
- Societies below n = 200 are drawn hollow and labelled rather than plotted as
  ordinary points.

## Known problems in the analysis

`figures/METHODOLOGY_NOTES.md` is the full write-up. The two that bite hardest
when touching the pipeline:

- **Q171 and Q172 are never rescaled.** `build_item_matrix` branches on the
  observed maximum (`== 10`, `== 4`, `== 3`, `== 2`, `== 1`); items whose
  maximum is 6-9 fall through and keep their native range. Exactly two do, and
  both are religiosity items, so Q172 alone contributes 4.7% of the squared
  distance between any two societies. This is faithfully preserved from
  `WVS.py`. If you fix it, the published numbers move — see the contract
  above.
- **Andorra is n = 35** after the native-born filter, and n = 7 in the 16-29
  cohort. Any new figure that ranks or plots societies needs the small-n
  treatment in `wvs_panels.SMALL_N`.

Also: `dict_countrycode2info` and `language_families` in `WVS.py` disagree on
Uzbekistan (Turkic vs Slavic). `wvs_meta.COUNTRY_INFO` is the single source of
truth; use `wvs_style._canon()` to fold `WVS.py`'s spellings (`Semetic`,
`Isolate`) onto the published names.

## Palette

Ten language families need ten categorical colors, past the point where color
alone can carry identity. `wvs_style.FAMILY_COLOR` was chosen by search and
clears every gate on **all 45 pairs**: worst CVD dE 8.5, worst normal-vision
dE 16.0, OKLCH L within 0.43-0.77, chroma >= 0.10. Three slots fall below 3:1
contrast on white, so every mark using them also carries a visible label, and
`FAMILY_MARKER` gives a second, color-free channel.

Do not add or substitute a family color by eye. Re-run the validator in the
`dataviz` skill (`scripts/validate_palette.js ... --mode light --pairs all`).

## Directories

- `WVS_v6/` — 2,150 PNGs from the original `WVS.py`, 4.1 GB. Historical; leave
  it alone.
- `WVS_v6_new/` — the current supplement, 1,869 figures, 177 MB.
  `cohort_name_map.txt` there maps old prefixes onto new ones.
- `figures/cache/` — 454 MB of pickles, regenerable. Not committed.
- Repo root — the old stitched `Figure1-5.png` and the LaTeX supplement
  drafts, kept for comparison.

Only `README.md` and `WVS.py` are tracked in git; everything else is
untracked. Check before assuming a file is under version control.

## Writing

US spelling throughout — prose, code comments, docstrings, axis labels,
captions.
