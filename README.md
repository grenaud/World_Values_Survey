# World Values Survey — language family and values

Code for a re-analysis of **wave 7 of the World Values Survey** (v6.0; 97,220
respondents across 66 societies, fielded 2017–2023), testing whether a
society's value profile tracks the family its language belongs to rather than
its geography or its religion. The motivating case was Morocco: does it pattern
with other African societies, with other Muslim-majority societies, or with
other Arabic-speaking ones?

This repository holds the analysis pipeline and the figure code. It does not
hold the survey data, the generated figures, or the manuscript — see
[Getting the data](#getting-the-data) and
[What is not in this repository](#what-is-not-in-this-repository).

## Repository layout

| path | what it is |
|---|---|
| `figures/` | **the working code**: the analysis pipeline and the figure builders. [`figures/README.md`](figures/README.md) documents the figure system in detail |
| `figures/METHODOLOGY_NOTES.md` | the long-form write-up of the known problems summarized below |
| `WVS.py` | the original notebook export, 4,183 lines. **Not runnable** — it contains `get_ipython()` calls and hardcoded paths from another checkout. Kept as the reference for what the analysis originally did, and as the source that `wvs_meta.py` is generated from |
| `WVS_v5.py`, `WVS2/` | earlier generations of the same analysis |
| `make_figure*.R`, `make_figures.ipynb` | the superseded figure approach: composite finished PNGs onto a canvas with ImageMagick or PIL. Retained for reference only |
| `generate_supp.ipynb` | the superseded supplement generator |
| `paper`, `Figures`, `supplementary`, `todo`, `cmds`, `header.txt` | working notes: the argument the paper is meant to make, figure plans, scratch commands |

### Inside `figures/`

| module | role |
|---|---|
| `wvs_pipeline.py` | the analysis. Load → filter → rescale → aggregate → impute, then PCA, NMF, distances, and their confidence intervals. Caches to `cache/*.pkl` |
| `wvs_meta.py` | country and question tables. **Generated, do not hand-edit** — produced once by `ast.literal_eval` on the dicts in `WVS.py`, so they cannot drift from the original by transcription |
| `wvs_style.py` | the single visual system: palette, markers, type stack, spacing, panel furniture, and `save_caption_notes` |
| `wvs_panels.py` | panel builders used by more than one figure (`nearest_neighbors`, `rank_shift`, and the `SMALL_N` threshold) |
| `figN_*.py` | one script per main-text figure; each is standalone and writes its own four output files |
| `make_all.py` | runs all five figure scripts and writes the LaTeX include files |
| `make_supplement.py` | the supplementary set: every cohort × every figure type, written to `../WVS_v6_new/` |

The dependency order is `wvs_meta` → `wvs_pipeline` → `wvs_style` →
`wvs_panels` → the figure scripts. Nothing imports in the other direction, so a
figure script can be run on its own and the pipeline can be used from a REPL
without touching matplotlib.

## Getting the data

The raw survey data is **not** in this repository. WVS asks that users obtain
it from the official site rather than from third parties.

1. Download the **wave 7 cross-national file, v6.0, CSV** from
   <https://www.worldvaluessurvey.org>.
2. Place it in the repository root as
   `WVS_Cross-National_Wave_7_csv_v6_0.csv.gz`.

`wvs_pipeline.RAW_CSV` points there. Nothing will run until it exists.

## Environment

Everything runs under a virtualenv at `~/pyenv`, not the system Python:

```bash
~/pyenv/bin/python figures/make_all.py
```

It needs `pandas`, `numpy`, `scipy`, `scikit-learn`, `matplotlib`, `seaborn`,
`geopandas`, `adjustText`, `umap-learn`, and `python-ternary`. Results were
produced with Python 3.13.7, numpy 2.3.5, pandas 2.3.3, scipy 1.16.3,
scikit-learn 1.7.2, matplotlib 3.10.7, seaborn 0.13.2, umap-learn 0.5.9.

There is no parquet engine installed, so the derived-matrix cache is pickle.

## Running it

All commands are run from `figures/`.

```bash
~/pyenv/bin/python make_all.py           # 5 main-text figures + .tex includes   (~1 min)
~/pyenv/bin/python fig3_nmf.py           # rebuild a single main figure
~/pyenv/bin/python wvs_pipeline.py       # print every cohort's n, societies, items

~/pyenv/bin/python make_supplement.py                  # 1,869 figures -> ../WVS_v6_new/  (~40 min)
~/pyenv/bin/python make_supplement.py --quick          # skip the 1,750 per-society charts
~/pyenv/bin/python make_supplement.py --only filtered  # one cohort, for iterating

FORCE=1 ~/pyenv/bin/python wvs_pipeline.py             # re-derive matrices from the raw CSV (~3 min)
```

`make_supplement.py` writes about 7,500 files over tens of minutes. Run it in
the background rather than in a foreground call that will time out.

### Caching

The first run reads the 21 MB CSV, builds the item matrix, aggregates, and
imputes for each of the thirteen cohorts, then pickles the five derived
matrices per cohort into `figures/cache/`. Every later run loads from there,
which is why figure tweaks cost seconds rather than minutes. The cache is keyed
by cohort tag only — **it does not detect that the code that produced it has
changed.** After editing anything in the preprocessing chain, re-derive with
`FORCE=1`, or the figures will be drawn from stale matrices.

## What the pipeline does

### Preprocessing

1. **Load** the raw CSV; drop the six World Governance Indicator columns and
   the v6.0 post-stratification weight; rename `B_COUNTRY_ALPHA` to
   `COUNTRIES` and upper-case the column names. No sampling weights are applied
   anywhere, so every country mean is unweighted.
2. **Select items** in four deterministic steps: drop any column where the
   "not asked in this country" code (`-4`) covers more than 80% of respondents;
   keep only columns beginning `Q`, `X`, or `V`; drop a curated list of 142
   demographic and country-level columns; take the first 199 that remain. The
   result is 199 value items — 77 morality, 59 politics, 26 economic,
   23 religion, 14 gender.
3. **Map negative codes to missing**, then **rescale** each item onto 1–5 by a
   lookup on its observed maximum: 10 → ÷2, 4 → ×5/4, 3 → ×5/3,
   2 → `{1→5, 2→1}`, 1 → `{1→5, 0→1}`.
4. **Aggregate** to per-country means, variances, and response counts.
5. **Impute** missing country–item entries with `KNNImputer(n_neighbors=2)`,
   uniform weights, `nan_euclidean` metric. Means and variances are imputed
   separately with the same settings.

### Analyses

| method | how it is configured |
|---|---|
| PCA | items standardized across the 66 societies, then SVD; 2 components, fixed seed |
| NMF | on the imputed means directly (non-negative by construction); coordinate descent, `nndsvda` init, `max_iter=1000`, weights row-normalized to sum to 1; k = 2, 3, 4 |
| distances | Euclidean between imputed mean vectors, in raw 1–5 units |
| intervals | delta-method SE for a distance between two mean vectors, using each society's own n; the original whole-sample expression is retained as `mode="notebook"` |
| clustering | average linkage (UPGMA) on the condensed distance matrix; no cut is applied |
| t-SNE | on the standardized matrix; PCA init, perplexity `max(5, min(30, (n-1)/3))` |
| UMAP | on the standardized matrix; `n_neighbors=min(15, n-1)` |
| Welch tests | individual-level, unequal variance, with Cohen's d; **no multiple-comparison correction** |
| Random Forest | one-vs-rest per family, 100 trees, 80/20 split, Gini importance |

Every stochastic step takes a fixed random seed, so repeated runs on the same
input are identical.

## The `Cohort` abstraction

Everything downstream hangs off `wvs_pipeline.Cohort` — one analyzed dataset,
that is, one filter applied to the raw survey:

```python
from wvs_pipeline import cohorts

c = cohorts()["filtered"]           # builds or loads from cache
c.means, c.variances, c.counts      # raw per-country aggregates
c.means_imp, c.var_imp              # after KNN imputation
c.n_total, c.countries, c.families()

scores, loadings, explained = c.pca(k=2)
weights, loadings           = c.nmf(3)
d                           = c.distances()
d, lo, hi                   = c.distance_ci(alpha=0.95, se_mode="delta")
```

Thirteen cohorts are declared in `COHORT_SPECS` as
`tag -> (parent, selector, description)`. **Adding one means adding a single
entry there and nothing else** — `cohorts()` will build it, and
`make_supplement.py` will pick it up and emit its full figure set.

| tag | derived from | respondents | populations |
|---|---|---|---|
| `unfiltered` | raw survey | 97,220 | 66 |
| `filtered` | raw survey | 83,770 | 66 |
| `cankaz` | `filtered` | 83,770 | 68 |
| `uspolitics` | `filtered` | 83,317 | 68 |
| `secondlanguage` | raw survey | 97,220 | 73 |
| `male` / `female` | `filtered` | 39,787 / 43,935 | 66 |
| `age_16_29` / `age_30_49` / `age_50plus` | `filtered` | 21,984 / 33,450 / 28,103 | 66 |
| `cde_16_29` / `cde_30_49` / `cde_50plus` | `uspolitics` | 21,867 / 33,250 / 27,967 | 68 |

`filtered` keeps respondents born in the country of interview to two
locally-born parents (`Q263`, `Q264`, `Q265`), and in the six high-immigration
countries additionally restricts to those speaking the dominant language at
home (`Q272`). The bilingual and party splits relabel `COUNTRIES` in place, so
CDE, CDF, KZK, KZR, USD, and USR flow through the rest of the pipeline as
ordinary societies.

## Output conventions

Every figure emits **four** files:

| file | for |
|---|---|
| `X.pdf` | LaTeX |
| `X.png` | quick review |
| `X.tex` | the `figure` environment with the main caption and label |
| `X.png_add_to_caption` | provenance and caveat sentences, one per line |

The sidecar exists because those sentences used to be set in 5.6 pt type at the
bottom of the canvas. They belong in the caption, where the typesetter sets them
at caption size and they reflow with the column. **Nothing appends them
automatically** — the LaTeX build has to concatenate them.

Main-text figures use uppercase panel letters (A, B, C); supplementary figures
use lowercase. Panel references in captions and sidecars must match.

Figures are composed natively: each is a single matplotlib canvas with one
`GridSpec`, saved as vector PDF. The superseded approach pasted finished PNGs
onto a canvas with PIL or ImageMagick, which is what produced the mismatched
panel letters and a 350 MB supplement. Please do not reintroduce raster
compositing.

## Reproducibility contract

The port reproduces the originally published numbers exactly, and these serve
as the regression check on any change to the preprocessing chain:

```
filtered      PC1 23.88%   PC2 13.48%
uspolitics    PC1 24.56%   PC2 13.17%
```

If a change moves those figures it is a bug, unless the change is deliberate
and recorded in `wvs_pipeline.DEVIATIONS`. Two quantities are computed
differently from `WVS.py`. Both are opt-in and both default to the original
behavior:

* **`se_mode`** — the original divided by the size of the whole sample, which
  omits the gradient of the distance with respect to the means and assigns
  Andorra (n = 35) the same precision as Brazil (n = 1,653). The understatement
  is roughly √(N/nᵢ): about 8× for a median-sized society, about 49× for
  Andorra. `se_mode="delta"` uses the first-order delta-method expression with
  each society's own n.
* **`distance_scaling`** — `"z"` standardizes items before computing distances,
  so that items with a wider native range do not dominate the metric. The
  original used raw 1–5 units.

## Known problems

Real, and preserved rather than fixed, because fixing them would move the
numbers above. `figures/METHODOLOGY_NOTES.md` is the full write-up.

* **Two items are never rescaled.** The rescaling lookup has no branch for an
  observed maximum between 6 and 9, so such items pass through unchanged.
  Exactly two do — Q171 (attendance at religious services, 1.75–6.54) and Q172
  (frequency of prayer, 1.16–7.77) — and both are religiosity items. Since
  distances are computed in raw units, a wider item contributes more squared
  distance: Q172 alone accounts for 4.7% of the squared spread between
  societies, about nine times the 0.5% it would contribute on a common scale.
  Anything computed on standardized items (the PCA, t-SNE, UMAP) is unaffected;
  the raw-unit distances are not.
* **Dichotomies are mapped to the endpoints `{1, 5}`**, so a binary item spans
  the full scale while a graded item is spread across the same range. In raw
  units a dichotomy therefore outweighs a graded item measuring the same thing.
* **Andorra is n = 35** after the native-born filter, and n = 7 in the 16–29
  cohort. Any figure that ranks or plots societies needs the small-n treatment
  in `wvs_panels.SMALL_N` (hollow marker plus explicit sample size below 200).
* **`WVS.py` contradicts itself on Uzbekistan** — `dict_countrycode2info` calls
  it Turkic, `language_families` calls it Slavic. `wvs_meta.COUNTRY_INFO` is the
  single source of truth; use `wvs_style._canon()` to fold the old spellings
  (`Semetic`, `Isolate`) onto the published names.
* **Two of the ten groups are geographic, not genealogical.** "East Asian" and
  "Sub-Saharan Africa" correspond to no single language family; they exist to
  keep the analysis tractable. Assigning whole countries to one family also
  erases linguistic minorities — Turkey is labelled Turkic despite a sizeable
  Kurdish-speaking population. Neither simplification is a claim about those
  speakers.

## Palette

Ten language families need ten categorical colors, past the point where color
alone can carry identity. `wvs_style.FAMILY_COLOR` was chosen by search, not by
eye, and clears every gate on all 45 pairs: worst CVD ΔE 8.5, worst
normal-vision ΔE 16.0, OKLCH lightness within 0.43–0.77, chroma ≥ 0.10.

Three slots fall below 3:1 contrast on white, so every mark using them also
carries a visible label, and `FAMILY_MARKER` provides a second, color-free
channel. **Do not add or substitute a family color by eye** — re-run the
validator described in [`figures/README.md`](figures/README.md).

## The manuscript

The paper and its supplement live in a **separate repository** with an Overleaf
remote. Two things about that wiring are easy to trip over:

* The supplement numbers its floats S1…S1869 and the main text reaches them
  through the `xr` package, so **the supplement must be compiled first** or
  every supplementary cross-reference prints `??`.
* The generated per-figure `.tex` files declare `[htbp]`. At 1,869 floats that
  overruns LaTeX's float queue, so the supplement preamble forces `[H]` and
  caps figure height.

## What is not in this repository

Roughly 9 GB of generated output and raw data, all either reproducible from
this code or subject to redistribution terms:

| excluded | size | why |
|---|---|---|
| `WVS/`, `WVS_v6/` | 8.3 GB | figure output from the earlier generations |
| `WVS_v6_new/` | 177 MB | the current supplement; rebuilt by `make_supplement.py` |
| `figures/cache/` | 454 MB | derived matrices; rebuilt with `FORCE=1` |
| `figures/out/` | 3.6 MB | the built main figures; rebuilt by `make_all.py` |
| the raw CSV, `WVS.ipynb` | 216 MB | see *Getting the data*; `WVS.py` is the notebook's export |

## Conventions

US spelling throughout — prose, code comments, docstrings, axis labels, and
captions.

## Attribution

Survey data: Haerpfer, C., Inglehart, R., Moreno, A., et al. (eds.), *World
Values Survey: Round Seven — Country-Pooled Datafile Version 6.0*. JD Systems
Institute & WVSA Secretariat. Use of the data is governed by the WVS conditions
of use; please cite it directly rather than citing this repository for the
underlying data.
