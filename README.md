# World Values Survey — does language predict values?

A re-analysis of **wave 7 of the World Values Survey** (v6.0; 97,220
respondents across 66 societies, fielded 2017–2023) asking a single question:
when two societies hold similar values, what best predicts it — their
geography, their religion, or the **family their language belongs to**?

The short answer is language family, imperfectly but more consistently than
either alternative. The longer answer, and the caveats that go with it, are
below.

## The motivating case

The study started from Morocco. Does it pattern with:

* **a)** other African societies,
* **b)** other Muslim-majority societies, or
* **c)** other Arabic-speaking societies?

The answer is (c), and the same logic generalizes: Semitic, Slavic, Turkic,
Indo-Iranian, Latin, and English-speaking societies each cluster with their
linguistic relatives more tightly than with their neighbors or co-religionists.
Ethiopia is the instructive counterexample — Semitic-speaking, but its value
profile sits squarely with its Sub-Saharan African neighbors.

## What the analysis does

Individual responses are filtered to value items (dropping questions about
circumstances rather than values, such as "I feel unsafe at night"), rescaled
onto a common 1–5 range, aggregated to country means, and KNN-imputed where a
question was not asked in a country. That 66 × 199 matrix is then put through
PCA, NMF, hierarchical clustering, pairwise distances with confidence
intervals, t-SNE, UMAP, Welch's t-tests, and Random Forest feature importance.

Two components carry most of the interpretable structure:

| | variance | what it separates |
|---|---|---|
| **PC1** | 23.9% | religious traditionalism versus secular permissiveness — importance of God, belief in heaven and hell, and religious authority at one end; the justifiability of sex before marriage, homosexuality, euthanasia, and abortion at the other |
| **PC2** | 13.5% | confidence in national and international institutions, together with active membership in voluntary associations |

The first NMF component is very nearly a re-expression of PC1 (r = 0.97), so
the moral axis is a property of the data rather than of one decomposition.

## Headline results

Distances below are Euclidean, between country mean response profiles over the
199 retained items.

* **Bilingual Kazakhstan behaves differently from bilingual Canada.** Split by
  language spoken at home, Kazakh (KZK) and Russian (KZR) speakers are each
  other's nearest neighbor by a wide margin (3.51; the next closest society to
  either is Russia). Beyond that first rank they part along linguistic lines —
  Russian speakers continue into the Slavic group, Kazakh speakers into the
  Turkic one — but national identity dominates.
* **Canada does not reciprocate.** French-speaking Canadians (CDF) are closest
  to English-speaking Canadians (CDE) at 4.56. English-speaking Canadians are
  closer to Great Britain (3.87), Australia (3.94), New Zealand (4.09), and
  Northern Ireland (4.36) than to their Francophone compatriots, who fall to
  fifth.
* **The USA is the least typical member of the Anglosphere**, and party
  identification explains it. Split into Democrat- and Republican-identifying
  respondents, the two halves sit **8.25** apart — wider than any pair of
  distinct societies within the Anglosphere. US Democrats are closer to English
  Canada (4.60), Northern Ireland, Australia, Great Britain, and New Zealand
  than to US Republicans. US Republicans have no close counterpart anywhere in
  the dataset.
* **A generational realignment in North America.** Among 16–29 year olds, US
  Democrats are the population closest to English-speaking Canada; among those
  50 and over, Great Britain is, and US Democrats fall to sixth.

## Repository layout

| path | what it is |
|---|---|
| `figures/` | **the working code.** The analysis pipeline and the figure builders. See [`figures/README.md`](figures/README.md) for the internals |
| `WVS.py` | the original notebook export, 4,183 lines. **Not runnable** — it contains `get_ipython()` calls and hardcoded paths from another checkout. Kept as the reference for what the analysis originally did, and as the source `wvs_meta.py` is generated from |
| `WVS_v5.py`, `WVS2/` | earlier generations of the same analysis |
| `make_figure*.R`, `make_figures.ipynb` | the superseded figure approach: composite finished PNGs onto a canvas with ImageMagick or PIL. Retained for reference; see below |
| `generate_supp.ipynb` | the superseded supplement generator |
| `paper`, `Figures`, `supplementary`, `todo`, `cmds`, `header.txt` | working notes: the argument the paper is meant to make, figure plans, and scratch commands |

## Getting the data

The raw survey data is **not** in this repository. WVS asks that users obtain
it from the official site rather than from third parties.

1. Download the **wave 7 cross-national file, v6.0, CSV** from
   <https://www.worldvaluessurvey.org>.
2. Place it in the repository root as
   `WVS_Cross-National_Wave_7_csv_v6_0.csv.gz`.

`figures/wvs_pipeline.py` reads it from there. Nothing else in the repository
will run until it exists.

## Running it

Everything runs under a virtualenv at `~/pyenv`, not the system Python. It
needs `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `scipy`,
`geopandas`, `adjustText`, `umap-learn`, and `python-ternary`. There is no
parquet engine, so the derived-matrix cache is pickle.

```bash
cd figures

~/pyenv/bin/python make_all.py           # 5 main-text figures + .tex includes  (~1 min)
~/pyenv/bin/python fig3_nmf.py           # rebuild one main figure
~/pyenv/bin/python make_supplement.py    # 1,869 supplementary figures -> ../WVS_v6_new/  (~40 min)
~/pyenv/bin/python make_supplement.py --quick          # skip the 1,750 per-society charts
~/pyenv/bin/python make_supplement.py --only filtered  # one cohort, for iterating

FORCE=1 ~/pyenv/bin/python wvs_pipeline.py             # re-derive matrices from the raw CSV (~3 min)
```

`make_supplement.py` writes about 7,500 files over tens of minutes. Run it in
the background rather than in a foreground call that will time out.

## Cohorts

Everything downstream hangs off `wvs_pipeline.Cohort` — one analyzed dataset,
holding `means`, `variances`, `counts`, `means_imp`, `var_imp`, with `.pca()`,
`.nmf(k)`, `.distances()`, and `.distance_ci()` on top. Thirteen are declared
in `COHORT_SPECS`; adding one means adding a single entry there.

| tag | n | populations | what it is |
|---|---|---|---|
| `unfiltered` | 97,220 | 66 | every respondent |
| `filtered` | 83,770 | 66 | native-born, speaking the dominant language at home. **The default for the main text** |
| `cankaz` | 83,770 | 68 | filtered, with Canada and Kazakhstan split by home language |
| `uspolitics` | 83,317 | 68 | filtered, with the USA split by party and Canada by language |
| `secondlanguage` | 97,220 | 73 | second-most-spoken language of each high-immigration country treated as its own population |
| `male` / `female` | 39,787 / 43,935 | 66 | filtered, by sex |
| `age_16_29` / `age_30_49` / `age_50plus` | 21,984 / 33,450 / 28,103 | 66 | filtered, by age band |
| `cde_16_29` / `cde_30_49` / `cde_50plus` | 21,867 / 33,250 / 27,967 | 68 | the USA/Canada split, by age band |

## The paper

The manuscript and its supplement live in a **separate repository** with an
Overleaf remote, not here. Two things about that wiring are easy to trip over:

* The supplement numbers its floats **S1…S1869** and the main text reaches
  them through the `xr` package. **The supplement must be compiled first**, or
  every supplementary cross-reference in the main text prints `??`.
* The generated per-figure `.tex` files declare `[htbp]`. At 1,869 floats that
  overruns LaTeX's float queue, so the supplement preamble forces `[H]` and
  caps figure height.

Each figure emits four files: `X.pdf` for LaTeX, `X.png` for review, `X.tex`
with the `figure*` environment and caption, and `X.png_add_to_caption` with
provenance and caveat sentences. **Nothing appends that last one
automatically** — the LaTeX build has to concatenate it.

Main-text figures use uppercase panel letters (A, B, C); supplementary figures
use lowercase.

## Reproducibility contract

The `figures/` port reproduces the originally published numbers exactly: PC1
and PC2 explain **23.88% / 13.48%** on the `filtered` cohort and **24.56% /
13.17%** on `uspolitics`. If a change moves those numbers it is a bug, unless
the change is deliberate and recorded in `wvs_pipeline.DEVIATIONS`.

Two things are computed differently from `WVS.py`. Both are opt-in and both
default to the notebook's behavior:

* **Confidence intervals.** The notebook divided by the size of the whole
  sample, which gives Andorra (n = 35) the same precision as Brazil
  (n = 1,762) and understates the standard error by roughly 9×.
  `se_mode="per_country"` uses each society's own n.
* **Distance scaling.** `distance_scaling="z"` standardizes items first, so
  that dichotomous items recoded to {1, 5} do not dominate the metric.

## Known problems

These are real and they are preserved rather than fixed, because fixing them
would move the published numbers. `figures/METHODOLOGY_NOTES.md` is the full
write-up.

* **Two items are never rescaled.** `build_item_matrix` branches on each item's
  observed maximum (`== 10`, `== 4`, `== 3`, `== 2`, `== 1`). Items whose
  maximum falls between 6 and 9 fall through and keep their native range.
  Exactly two do — Q171 (attendance at religious services, observed 1.75–6.54)
  and Q172 (frequency of prayer, 1.16–7.77) — and both are religiosity items.
  **Q172 alone accounts for 4.7%** of the squared spread between societies,
  purely because its scale was never compressed.
* **Andorra is n = 35** after the native-born filter, and n = 7 in the 16–29
  cohort. Any new figure that ranks or plots societies needs the small-n
  treatment in `wvs_panels.SMALL_N`.
* **`WVS.py` contradicts itself on Uzbekistan** — `dict_countrycode2info` calls
  it Turkic, `language_families` calls it Slavic. `wvs_meta.COUNTRY_INFO` is
  the single source of truth; use `wvs_style._canon()` to fold the old
  spellings (`Semetic`, `Isolate`) onto the published names.
* **Two of the ten groups are geographic, not genealogical.** "East Asian" and
  "Sub-Saharan Africa" do not correspond to any single language family; they
  exist to keep the analysis tractable. Assigning whole countries to one family
  also erases linguistic minorities — Turkey is called Turkic despite a
  sizeable Kurdish-speaking population. Neither simplification is meant as a
  claim about those speakers.

## What is deliberately not in this repository

Roughly 9 GB of generated output and raw data is excluded, all of it either
reproducible from this code or subject to redistribution terms:

| excluded | size | why |
|---|---|---|
| `WVS/`, `WVS_v6/` | 8.3 GB | figure output from the earlier generations |
| `WVS_v6_new/` | 177 MB | the current supplement, rebuilt by `make_supplement.py` |
| `figures/cache/` | 454 MB | derived matrices; rebuilt with `FORCE=1` |
| `figures/out/` | 3.6 MB | the built main figures; rebuilt by `make_all.py` |
| the raw CSV, `WVS.ipynb` | 216 MB | see *Getting the data*; `WVS.py` is the notebook's export |

## Why the figure code was rewritten

The earlier approach composed each figure by pasting finished PNGs onto a
canvas with PIL or ImageMagick. That is what produced the uneven panel letters,
the mismatched type sizes between panels, and a 350 MB supplement. Each figure
is now a single matplotlib canvas with one `GridSpec`, so panels share a
coordinate system, a font stack, and a baseline grid, and the output is vector.
Please do not reintroduce raster compositing.

## Attribution

Survey data: Haerpfer, C., Inglehart, R., Moreno, A., et al. (eds.), *World
Values Survey: Round Seven — Country-Pooled Datafile Version 6.0*. JD Systems
Institute & WVSA Secretariat. Use of the data is governed by the WVS
conditions of use; please cite it directly rather than citing this repository
for the underlying data.
