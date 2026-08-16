# Paper figures

Vector figures for the main text, built natively in matplotlib.

```
python make_all.py               # the five main-text figures + LaTeX includes
python make_supplement.py        # the supplementary set -> ../WVS_v6_new/
FORCE=1 python wvs_pipeline.py   # re-derive matrices from the raw CSV (~3 min)
```

Output lands in `out/` as both PDF (for LaTeX) and PNG (for quick viewing),
plus per figure:

* `FigureN.tex` — the `figure*` environment with the main caption
* `FigureN.png_add_to_caption` — provenance and caveat sentences, one per
  line, to be appended to the caption at build time

Those sentences used to be set in 5.6 pt type at the bottom-left of the
canvas. They belong in the caption: the typesetter sets them at the caption's
own size, and they reflow with the column instead of being frozen into the
artwork at whatever scale the figure lands at. Nothing appends them
automatically — wire the sidecar into your LaTeX build, or paste them into the
caption.

## Layout

| file | what it is |
|---|---|
| `wvs_pipeline.py` | the analysis: load, filter, rescale, aggregate, impute, PCA/NMF/distances. Caches derived matrices to `cache/*.pkl` |
| `wvs_meta.py` | country and question tables, extracted from `WVS.py` with `ast.literal_eval` so they cannot drift by transcription |
| `wvs_style.py` | the shared visual system: palette, type, spacing, panel furniture |
| `wvs_panels.py` | panel builders used by more than one figure |
| `figN_*.py` | one script per main-text figure |
| `make_supplement.py` | the supplementary set, written to `../WVS_v6_new/` |

## Why this replaces the PIL/ImageMagick approach

`make_figures.ipynb` and `make_figure*.R` composed each figure by pasting
finished PNGs onto a canvas. That is what produced the uneven panel letters,
the mismatched type sizes between panels, the acres of white space, and the
40 MB rasters. Here each figure is a single matplotlib canvas with one
`GridSpec`, so panels share a coordinate system, a font stack and a baseline
grid, and the output is vector.

## Reproducibility check

The pipeline is a faithful port of `performAnalyses` in `WVS.py`. It
reproduces the published numbers exactly: PC1/PC2 explain 23.88% / 13.48% of
variance on the filtered cohort and 24.56% / 13.17% on the US-politics
cohort, matching the titles of the earlier `Figure2.png` and `Figure5.png`.

Two things are computed differently, both deliberately and both documented in
`wvs_pipeline.DEVIATIONS`:

* **Confidence intervals.** `WVS.py` used
  `sqrt(sum_k(var_ik + var_jk)) / sqrt(N)` with `N = len(df)`, the size of the
  *whole* sample. That is not the standard error of a distance between two
  country means: it omits the gradient of the distance with respect to the
  means, and it gives Andorra (n = 35) the same precision as Brazil
  (n = 1,762). `distance_se(..., mode="delta")` uses the first-order
  delta-method expression with each society's own sample size. Pass
  `mode="notebook"` to get the old numbers back.
* **Small samples are flagged.** After the native-born / home-language filter
  Andorra retains 35 respondents; in the age-split cohorts several societies
  fall below 50. Panels mark these rather than plotting them as if they were
  comparable estimates.

## Palette

Ten language families need ten categorical colors, which is past the point
where color alone can carry identity. The palette was therefore chosen by
search rather than by eye, against the checks in the `dataviz` skill, and it
clears every gate on **all 45 pairs** (not merely adjacent ones):

```
worst CVD ΔE      8.5   (protan/deutan, Machado 2009 severity 1.0; gate 8)
worst normal ΔE  16.0   (gate 15)
OKLCH lightness  0.43–0.77 for every slot;  chroma ≥ 0.10
```

Verify with:

```
node <dataviz>/scripts/validate_palette.js \
  "#df5038,#90442c,#d39c43,#4bc08e,#1c8b6e,#3a85d4,#2147db,#b198e0,#7c37a1,#de45aa" \
  --mode light --pairs all
```

Three slots fall below 3:1 contrast against white, so every mark that uses
them also carries a visible text label. Marker shape is a second, color-free
channel, and every point in every scatter is directly labeled with its ISO
code — identity is never carried by color alone.


## The supplement

`make_supplement.py` writes to `../WVS_v6_new/`, leaving the original
`WVS_v6/` untouched for comparison. Thirteen cohorts (unfiltered, filtered,
the two splits, second-language, both sexes, three age bands, three age bands
of the USA/Canada split) each get nine figures:

| figure | replaces |
|---|---|
| `<tag>_pca` | `pca_pc1_pc2_imp` |
| `<tag>_pca_by_family` | the ten `pca_<FAMILY>_pc1_pc2_imp` plates |
| `<tag>_biplot` | `BiplotPCA` |
| `<tag>_loading_angles` | `biplotPCA_circ` and the five `biplotPCA_circ_<THEME>` |
| `<tag>_clustermap` | `clustermap_imp`, `clustermaplang_imp`, `heatmap`, `rawheatmap` |
| `<tag>_distance_matrix` | `nmf_pairwise_dendrogram_imp` |
| `<tag>_nmf` | `nmf_c2_imp`, `nmf_c3_imp`, `nmf_c3_triangle`, `nmf_c4_imp` |
| `<tag>_embeddings` | `tsne_imp`, `umap_1_2_imp` |
| `<tag>_english_proficiency` | `eng_pro_corr` |

plus, per society, `<tag>_distances_<ISO>` and `<tag>_engprof_<ISO>`; plus two
cross-cohort contrasts (`contrast_male_female`, `contrast_age_young_old`);
plus `supplement_figures.tex`, which `\input`s everything in order under
per-cohort subsections, and `cohort_name_map.txt`, which maps the old
`WVS_v6/` prefixes onto the new ones.

Four figure types were redrawn rather than restyled, because the form was the
problem:

| type | what changed |
|---|---|
| per-family PCA | was ten near-identical full-page plates per cohort, differing only in which societies were drawn large. Now one facet grid, so the thing the set exists to show — how tightly each family sits together — is a single look |
| PCA biplot | ranked arrows by joint loading magnitude, which picks a degenerate set: the "confidence in institutions" battery has the longest vectors and they all point the same way. Now the extremes along each axis separately, which is what the notebook's own `is_top_5_component` intended |
| loading-angle roses | six separate rose diagrams per cohort, which cannot be read quantitatively. Now one figure of seven polar panels, each carrying the mean resultant vector and its length R, the standard circular measure of concentration |
| country x item clustermap | 199 item labels are illegible at page width; the column dendrogram and a theme color strip carry the item side instead. The ramp spans the 1st-99th percentile of the observed means, not the nominal 1-5, or the plate is a wash of mid-blue |
| per-society distance chart | 67 bars, all but the first dozen visually identical, with a viridis fill re-encoding the y value while the tick labels encoded family. Now a ranked dot plot of the nearest twelve with a confidence interval |
| t-SNE and UMAP | were two separate figures answering the same question; now two panels of one |
| NMF | k = 2, 3 and 4 were four figures; now one, and the k = 2 stacked bar (whose second segment is one minus the first) is a strip plot |

Everything else is the same analysis as `performAnalyses`, drawn through
`wvs_style` so the supplement matches the main text.
