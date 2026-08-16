# Methodology notes on `WVS.py`

Written while porting `performAnalyses` into `wvs_pipeline.py`. The port
reproduces the published numbers exactly (PC1/PC2 = 23.88% / 13.48% on the
filtered cohort; 24.56% / 13.17% on the US-politics cohort), so everything
below is about the analysis as run, not about a failure to reproduce it.

Ordered by how much they could change a claim in the paper.

---

## 1. The grouping variable is not a language classification

`dict_langfam2color` has ten levels, and only about half are language
families. `Anglosphere`, `EastAsia`, `SubSaharanAfrica`, `Latin` and `Other`
are geographic or cultural-bloc labels. `EastAsia` in particular pools
Austronesian (Indonesian, Filipino, Malay), Austroasiatic (Vietnamese),
Kra-Dai (Thai), Japonic, Koreanic and Sino-Tibetan — six unrelated families
whose only shared property is location. `Semetic` (sic) pools the Arabic
states with Ethiopia, where the surveyed languages include Cushitic Oromo.

This matters because the paper's headline is that **language**, not geography
or religion, explains the clustering. A grouping variable that is itself
partly geographic cannot separate those hypotheses. Two ways out, either of
which is defensible:

- retitle the variable to what it is (a language-and-region bloc), and soften
  the claim accordingly; or
- test the claim properly. Build three competing partitions of the 66
  societies — genealogical language family (Glottolog/WALS), geographic
  region, and majority religion — and compare how much of the between-society
  distance matrix each explains (PERMANOVA / `adonis`, or a Mantel test
  against a geographic-distance matrix, or simply silhouette width per
  partition). That is a two-afternoon analysis on the matrices already in
  `figures/cache/` and it converts the central claim from an assertion about
  a scatter plot into a testable number.

The same point applies with more force to the Morocco question in `paper`
(African vs Muslim vs Arabic-speaking). Neither "African" nor "Muslim" exists
as a variable anywhere in the pipeline, so option (C) is currently the only
one on the ballot. Add a majority-religion column and a continent column and
the question answers itself.

## 2. Two items were never rescaled, and they dominate the distance metric

`build_item_matrix` rescales each item onto 1–5 by branching on its observed
maximum (`== 10`, `== 4`, `== 3`, `== 2`, `== 1`). Items whose maximum is 6,
7, 8 or 9 fall through every branch and keep their native range. Exactly two
items do this, and they are both religiosity items:

| item | wording | range after "rescaling" |
|---|---|---|
| Q171 | How often do you attend religious services | 1–7 |
| Q172 | How often do you pray | 1–8 |

Distances, NMF and the heatmaps all run on these raw units (only PCA
standardizes). The consequence is measurable: **Q172 alone contributes 4.7% of
the squared Euclidean distance between any two societies**, more than twice any
other item, and Q171+Q172 together contribute 6.8% where two average items
would contribute about 1%. Since PC1 is described in the draft as a
religion-and-morality axis, an accidental 7× overweighting of the two prayer
and attendance items is not a cosmetic problem.

Fix: replace the `max`-branching with an explicit min–max rescale,
`(x - min) / (max - min) * 4 + 1`, using the codebook range rather than the
observed one.

Related, and easy to miss: the branch keys are read off the **observed**
maximum of whatever subset is being analyzed. A dichotomous item where nobody
in the male-only subset picked option 2 would be rescaled by a different rule
than in the female-only subset, making the two subsets non-comparable. Using
codebook ranges fixes this too.

## 3. The confidence intervals are not intervals for the statistic

```python
n = len(df)                                   # every respondent in the dataset
var_matrix = variances_i + variances_j        # summed over items
se_matrix  = sqrt(var_matrix) / sqrt(n)
ci = dist ± t.ppf(0.975, n - 1) * se
```

Two independent problems:

- **`n` is the whole sample.** A country mean is estimated from that country's
  respondents, not from all 83,770. As written, Andorra (n = 35 after the
  filter) is given the same precision as Brazil (n = 1,762).
- **The gradient is missing.** For `d = ||m_i − m_j||` the delta method gives
  `Var(d) ≈ Σ_k ((m_ik − m_jk)/d)² (s²_ik/n_i + s²_jk/n_j)`. The notebook sums
  the raw variances instead, i.e. it projects onto no direction at all.

The two errors point in opposite directions, so the published bars land within
a factor of ~3 of a correct interval (ratio to delta-method SE ranges 0.31–2.72
across pairs on the CAN/KAZ cohort). They are not, however, computed from
the sampling distribution of the quantity plotted, and they do not respond to
sample size at all — which is the one thing an error bar on this plot has to
do. `wvs_pipeline.distance_se(..., mode="delta")` implements the expression
above; `mode="notebook"` reproduces the old numbers.

## 4. Small samples after filtering are not flagged anywhere

The native-born + home-language filter removes 13,450 respondents overall,
but unevenly:

| society | before | after |
|---|---|---|
| Andorra | 1,004 | **35** |
| Macau | ~1,023 | 197 |
| Northern Ireland | 440 | 372 |

Andorra is a country where most residents were born elsewhere, so the filter
removes almost everyone. AND is then plotted as an ordinary point in every
PCA, every dendrogram and every distance ranking.

It gets worse in the age-split cohorts that Figure 5's argument rests on:
`16–29` leaves AND n = 7, NZL n = 39, NIR n = 41, AUS n = 82; `50+` leaves
AND n = 18, MAC n = 22. The claim that young English Canadians sit closer to
US Democrats while older ones sit closer to the British is a claim about
rank order among societies whose means are, in several cases, estimated from
tens of people.

The redrawn figures mark every society below n = 200 with a hollow marker and
print its n. That is a presentational patch, not a fix — consider a minimum-n
threshold for inclusion, or bootstrap the rankings over respondents.

## 5. The random-forest accuracies do not support anything

For each family, a one-vs-rest `RandomForestClassifier` is trained on
individual respondents and the accuracy is written to
`top_5_features_per_language_family.tex`. Two problems:

- **Accuracy on a very imbalanced binary task.** Always predicting "not this
  family" already scores 0.968 for Germanic, 0.967 for Other, 0.957 for
  Sub-Saharan Africa, 0.930 for Slavic. Any reported accuracy has to be read
  against that floor, and none of the numbers in the table are. Report
  balanced accuracy, AUC or F1 instead.
- **Respondents from the same country appear in train and test.**
  `train_test_split(X, y, test_size=0.2, random_state=42)` splits at the
  respondent level, but the label is a deterministic function of country. The
  model can memorize country-specific response styles and score well without
  learning anything about language. A grouped split (`GroupKFold` on country,
  leave-one-country-out) is the honest version, and it is the version that
  would actually test the paper's thesis.

The feature-importance rankings that the table reports are probably still
informative; the accuracies next to them are not.

## 6. Uzbekistan has two different language families

```python
dict_countrycode2info["UZB"][3] == "Turkic"     # used for scatter/bar colors
language_families["UZB"]        == "Slavic"     # used for the map and clustermap row colors
```

So UZB is purple in the PCA and pink on the map, in the same paper. Turkic is
presumably intended. This is the only disagreement between the two tables —
`wvs_meta.py` was generated by `ast.literal_eval` on both dicts specifically
to check. The redrawn figures use `COUNTRY_INFO` throughout.

## 7. Smaller things

- **`USN` / `USS` are inverted.** `O2_LATITUDE >= 39` is assigned `'USS'`
  (South) and `< 39` is assigned `'USN'` (North). The latitude split is
  commented out of the analysis, so nothing published is affected, but the
  label dictionary and `DICT_DATASET_TAG2DESC` still describe it.
- **`Semetic` → `Semitic`** throughout, and `Great_Britain` → `Great Britain`
  in figure titles. Both appear in the current figures.
- **The 80% item filter counts respondents, not countries.** `prop_minus_four`
  is computed over rows, so a question asked only in the four largest samples
  survives while one asked in twenty small ones may not. Counting distinct
  countries is the more natural criterion. In practice this turns out to be
  benign here — after filtering, only 0.9% of the 66 × 199 country-item mean
  cells need imputing, and no item is missing in more than half the societies.
  Worth a sentence in Methods rather than a rerun. (Egypt is the outlier at
  12% of its items imputed; that is worth a look.)
- **KNN imputation feeds the distance matrix.** Missing country means are
  filled from the 2 nearest countries, and those filled values then enter the
  distance matrix used to decide which countries are near each other. Given
  the 0.9% figure above the circularity is not doing real damage, but it
  should be stated, and Egypt's 12% should be checked.
- **Distances are computed on unstandardized item means while PCA
  standardizes.** Both are defensible; using different conventions in adjacent
  figures of the same paper is what needs a justification in Methods.
