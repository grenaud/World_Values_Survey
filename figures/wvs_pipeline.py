"""Reproducible WVS-7 analysis pipeline.

Extracted from ``WVS.py`` (``performAnalyses``) so that figure rendering is
decoupled from the analysis.  The preprocessing here is a faithful port of the
original notebook code: same column filters, same 1-5 rescaling, same
country-level aggregation, same KNN imputation.  Anything that *changes* a
number relative to the notebook is opt-in and flagged in ``DEVIATIONS`` below.

Derived matrices are cached as Parquet under ``figures/cache/`` so that figure
tweaks do not pay the ~2 min cost of re-reading the raw survey.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import NMF, PCA
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler

from wvs_meta import COLS_TO_REMOVE, COUNTRY_INFO, LANGUAGE_FAMILY, QUESTIONS

ROOT = Path(__file__).resolve().parent.parent
CACHE = Path(__file__).resolve().parent / "cache"
CACHE.mkdir(exist_ok=True)

RAW_CSV = ROOT / "WVS_Cross-National_Wave_7_csv_v6_0.csv.gz"

DEVIATIONS = """
Deviations from WVS.py, all opt-in:
  * ``se_mode="per_country"`` divides each country's item variance by that
    country's own n instead of by len(df) (the whole sample).  The notebook
    used the latter, which understates the standard error by ~sqrt(N/n) ~ 9x.
  * ``distance_scaling="z"`` computes distances on standardized items so that
    dichotomous items (recoded to {1,5}) do not dominate the Euclidean metric.
    The notebook used raw 1-5 units.
Both default to the notebook's behavior so figures reproduce as-published.
"""

V6_DROP = ["td_ctrlcorr", "td_goveff", "td_polstab", "td_regqual",
           "td_rulelaw", "td_voiacc", "PWGHT"]

# Respondents who speak the country's dominant language at home, for the
# high-immigration countries.  1240 English, 1530 German, 1190 Dutch,
# 1400 French, 2230 Kazakh, 3630 Russian.
HOME_LANGUAGE = {
    "USA": [1240], "AUS": [1240], "NZL": [1240], "GBR": [1240],
    "DEU": [1530], "NLD": [1190], "CAN": [1240, 1400], "KAZ": [2230, 3630],
}


# --------------------------------------------------------------------------- #
# raw load                                                                     #
# --------------------------------------------------------------------------- #
def load_raw(force: bool = False) -> pd.DataFrame:
    """Read the WVS wave-7 v6 CSV, harmonised and with COUNTRIES/upper columns."""
    pkl = CACHE / "raw.pkl"
    if pkl.exists() and not force:
        return pd.read_pickle(pkl)

    df = pd.read_csv(RAW_CSV, compression="gzip", encoding="utf8", low_memory=False)
    df = df.drop(columns=V6_DROP, errors="ignore")
    if "N_REG_NUTS1" in df.columns and "reg_nuts1" not in df.columns:
        df = df.rename(columns={"N_REG_NUTS1": "reg_nuts1"})
    df = df.rename(columns={"B_COUNTRY_ALPHA": "COUNTRIES"})
    df.columns = df.columns.str.upper()
    df.to_pickle(pkl)
    return df


# --------------------------------------------------------------------------- #
# cohort construction                                                          #
# --------------------------------------------------------------------------- #
def native_filter(df: pd.DataFrame) -> pd.DataFrame:
    """Notebook's ``df_filter_all``: native-born (self + both parents) and, in
    high-immigration countries, speaking the dominant language at home.

    ``-4`` (question not asked in this country) is retained, so the filter only
    actually bites in countries where Q263-Q265 were fielded.
    """
    out = df.loc[df["Q263"].isin([1, -4])
                 & df["Q264"].isin([1, -4])
                 & df["Q265"].isin([1, -4])]

    cond = np.zeros(len(out), dtype=bool)
    for country, langs in HOME_LANGUAGE.items():
        cond |= ((out["COUNTRIES"] == country) & out["Q272"].isin(langs)).to_numpy()
    cond = pd.Series(cond, index=out.index)
    return out.loc[cond | (~out["COUNTRIES"].isin(HOME_LANGUAGE))]


def split_can_kaz(df: pd.DataFrame) -> pd.DataFrame:
    """Canada -> CDE/CDF, Kazakhstan -> KZK/KZR by language spoken at home."""
    out = df.copy()
    q272 = pd.to_numeric(out["Q272"], errors="coerce")
    out.loc[(out["COUNTRIES"] == "CAN") & (q272 == 1240), "COUNTRIES"] = "CDE"
    out.loc[(out["COUNTRIES"] == "CAN") & (q272 == 1400), "COUNTRIES"] = "CDF"
    out.loc[(out["COUNTRIES"] == "KAZ") & (q272 == 2230), "COUNTRIES"] = "KZK"
    out.loc[(out["COUNTRIES"] == "KAZ") & (q272 == 3630), "COUNTRIES"] = "KZR"
    return out


def split_us_politics(df: pd.DataFrame) -> pd.DataFrame:
    """USA -> USR/USD by party identification; Canada -> CDE/CDF.

    Respondents who fall in neither bucket are dropped, as in the notebook.
    """
    out = df.copy()
    q223 = pd.to_numeric(out["Q223"], errors="coerce")
    q272 = pd.to_numeric(out["Q272"], errors="coerce")
    out.loc[(out["COUNTRIES"] == "USA") & (q223 == 840001), "COUNTRIES"] = "USR"
    out.loc[(out["COUNTRIES"] == "USA") & (q223 == 840002), "COUNTRIES"] = "USD"
    out = out.loc[out["COUNTRIES"] != "USA"]
    out.loc[(out["COUNTRIES"] == "CAN") & (q272 == 1240), "COUNTRIES"] = "CDE"
    out.loc[(out["COUNTRIES"] == "CAN") & (q272 == 1400), "COUNTRIES"] = "CDF"
    return out.loc[out["COUNTRIES"] != "CAN"]


# --------------------------------------------------------------------------- #
# item matrix                                                                  #
# --------------------------------------------------------------------------- #
def build_item_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """COUNTRIES + the 199 value items, rescaled to a common 1-5 range.

    Faithful port of the notebook: drop items unasked for >80% of *respondents*,
    keep Q/X/V columns, drop the demographic block, map negative codes to NaN,
    then rescale each item by its observed maximum.
    """
    minus_four = df.apply(lambda x: (x == -4).sum(), axis=0) / len(df)
    df = df.drop(columns=minus_four[minus_four > 0.8].index)

    items = df.loc[:, df.columns.str.startswith(("Q", "X", "V"))]
    items = items.drop(columns=["Q_MODE", "VERSION"], errors="ignore")
    items = items.drop(columns=[c for c in COLS_TO_REMOVE if c in items.columns])

    # notebook layout: [COUNTRIES, K_DURATION, *items]; subset takes items 0..198
    subset = pd.concat([df[["COUNTRIES"]], items.iloc[:, :199]], axis=1)

    q = subset.columns[1:]
    subset[q] = subset[q].astype(int)
    subset[q] = subset[q].mask(subset[q] < 0)

    # rescale each item onto 1-5 using its observed maximum
    max_values = subset[q].max()
    for col in q:
        m = max_values[col]
        if m == 10:
            subset[col] = subset[col] / 2
        elif m == 4:
            subset[col] = subset[col] * 5 / 4
        elif m == 3:
            subset[col] = subset[col] * 5 / 3
        elif m == 2:
            subset[col] = subset[col].replace({1: 5, 2: 1})
        elif m == 1:
            subset[col] = subset[col].replace({1: 5, 0: 1})
    return subset


def aggregate(subset: pd.DataFrame, n_neighbors: int = 2):
    """Country means / variances / counts, plus KNN-imputed means."""
    grouped = subset.groupby("COUNTRIES")
    means = grouped.mean()
    variances = grouped.var()
    counts = grouped.count()

    imputer = KNNImputer(n_neighbors=n_neighbors)
    means_imp = pd.DataFrame(imputer.fit_transform(means),
                             columns=means.columns, index=means.index)
    var_imp = pd.DataFrame(imputer.fit_transform(variances),
                           columns=variances.columns, index=variances.index)
    return means, variances, counts, means_imp, var_imp


# --------------------------------------------------------------------------- #
# derived analyses                                                             #
# --------------------------------------------------------------------------- #
def run_pca(means_imp: pd.DataFrame, n_components: int = 2):
    X = StandardScaler().fit_transform(means_imp.values)
    pca = PCA(n_components=n_components, random_state=0)
    scores = pd.DataFrame(pca.fit_transform(X),
                          index=means_imp.index,
                          columns=[f"PC{i + 1}" for i in range(n_components)])
    loadings = pd.DataFrame(pca.components_.T,
                            index=means_imp.columns,
                            columns=scores.columns)
    return scores, loadings, pca.explained_variance_ratio_


def run_nmf(means_imp: pd.DataFrame, n_components: int, random_state: int = 0):
    """Row-normalized NMF weights (each country's components sum to 1)."""
    model = NMF(n_components=n_components, init="nndsvda",
                random_state=random_state, max_iter=1000)
    W = model.fit_transform(means_imp.values)
    W = W / W.sum(axis=1, keepdims=True)
    weights = pd.DataFrame(W, index=means_imp.index,
                           columns=[f"C{i + 1}" for i in range(n_components)])
    loadings = pd.DataFrame(model.components_.T, index=means_imp.columns,
                            columns=weights.columns)
    return weights, loadings


def distance_matrix(means_imp: pd.DataFrame, scaling: str = "raw") -> pd.DataFrame:
    """Pairwise Euclidean distance between country mean-response vectors."""
    X = means_imp.values
    if scaling == "z":
        X = StandardScaler().fit_transform(X)
    elif scaling != "raw":
        raise ValueError(f"unknown scaling {scaling!r}")
    d = np.sqrt(((X[:, None, :] - X[None, :, :]) ** 2).sum(-1))
    return pd.DataFrame(d, index=means_imp.index, columns=means_imp.index)


def distance_se(means_imp, var_imp, counts, n_total, mode="notebook",
                scaling="raw"):
    """Standard error of each pairwise distance.

    ``mode="notebook"`` reproduces WVS.py: sqrt(sum_k(var_ik + var_jk)) / sqrt(N)
    with N the number of respondents in the *whole* dataset, and with no
    gradient weighting.

    ``mode="delta"`` is the first-order delta-method SE actually implied by the
    statistic, using each country's own sample size:
        Var(d) ~ sum_k ((m_ik - m_jk) / d)^2 * (s2_ik / n_i + s2_jk / n_j)
    """
    idx = means_imp.index
    V = var_imp.values
    if mode == "notebook":
        se = np.sqrt(V[:, None, :].sum(-1) + V[None, :, :].sum(-1)) / np.sqrt(n_total)
        return pd.DataFrame(se, index=idx, columns=idx)

    if mode != "delta":
        raise ValueError(f"unknown mode {mode!r}")

    M = means_imp.values
    n = counts.median(axis=1).values.astype(float)  # per-country n, item-median
    if scaling == "z":
        s = M.std(axis=0, ddof=0)
        s[s == 0] = 1.0
        M = (M - M.mean(axis=0)) / s
        V = V / s ** 2
    diff = M[:, None, :] - M[None, :, :]
    d = np.sqrt((diff ** 2).sum(-1))
    with np.errstate(divide="ignore", invalid="ignore"):
        grad = diff / d[:, :, None]
        pooled = V[:, None, :] / n[:, None, None] + V[None, :, :] / n[None, :, None]
        var_d = (grad ** 2 * pooled).sum(-1)
    var_d[~np.isfinite(var_d)] = 0.0
    return pd.DataFrame(np.sqrt(var_d), index=idx, columns=idx)


# --------------------------------------------------------------------------- #
# cohort driver                                                                #
# --------------------------------------------------------------------------- #
class Cohort:
    """One analyzed dataset (a filter applied to the raw survey)."""

    def __init__(self, tag, means, variances, counts, means_imp, var_imp, n_total):
        self.tag = tag
        self.means = means
        self.variances = variances
        self.counts = counts
        self.means_imp = means_imp
        self.var_imp = var_imp
        self.n_total = n_total

    @property
    def countries(self):
        return list(self.means_imp.index)

    def families(self):
        return pd.Series({c: COUNTRY_INFO[c][3] for c in self.countries})

    def pca(self, k=2):
        return run_pca(self.means_imp, k)

    def nmf(self, k):
        return run_nmf(self.means_imp, k)

    def distances(self, scaling="raw"):
        return distance_matrix(self.means_imp, scaling)

    def distance_ci(self, alpha=0.95, se_mode="delta", scaling="raw"):
        from scipy import stats
        d = self.distances(scaling)
        se = distance_se(self.means_imp, self.var_imp, self.counts,
                         self.n_total, mode=se_mode, scaling=scaling)
        dof = (self.counts.median(axis=1).median() - 1) if se_mode == "delta" \
            else (self.n_total - 1)
        t = stats.t.ppf((1 + alpha) / 2, max(dof, 1))
        return d, d - t * se, d + t * se


def build_cohort(tag: str, df: pd.DataFrame, force: bool = False) -> Cohort:
    """Build (or load) the aggregated matrices for one filtered dataset."""
    paths = {k: CACHE / f"{tag}_{k}.pkl"
             for k in ("means", "var", "counts", "means_imp", "var_imp")}
    meta = CACHE / f"{tag}_n.txt"

    if all(p.exists() for p in paths.values()) and meta.exists() and not force:
        loaded = {k: pd.read_pickle(p) for k, p in paths.items()}
        return Cohort(tag, loaded["means"], loaded["var"], loaded["counts"],
                      loaded["means_imp"], loaded["var_imp"],
                      int(meta.read_text()))

    subset = build_item_matrix(df)
    means, variances, counts, means_imp, var_imp = aggregate(subset)
    for k, obj in zip(paths, (means, variances, counts, means_imp, var_imp)):
        obj.to_pickle(paths[k])
    meta.write_text(str(len(df)))
    return Cohort(tag, means, variances, counts, means_imp, var_imp, len(df))


#: Second-most-spoken language in the high-immigration countries, used by the
#: notebook's SecondLanguageFilter cohort.
SECOND_LANGUAGE = {
    "CAN": (1240, "CANCHN"), "USA": (1270, "USASPN"), "GBR": (3520, "GRBPLS"),
    "AUS": (2870, "AUSCHN"), "NZL": (2870, "NZLCHN"), "DEU": (4370, "DEUTRK"),
    "NLD": (1240, "NLDENG"),
}


def split_second_language(df: pd.DataFrame) -> pd.DataFrame:
    """Notebook's ``df_SecLang``: relabel second-language speakers separately."""
    out = df.copy()
    q272 = pd.to_numeric(out["Q272"], errors="coerce")
    for country, (code, tag) in SECOND_LANGUAGE.items():
        out.loc[(out["COUNTRIES"] == country) & (q272 == code), "COUNTRIES"] = tag
    return out


#: Every cohort the main text and the supplement need, as
#: ``tag -> (parent, selector, human-readable description)``.
COHORT_SPECS = {
    "unfiltered":        ("raw",    None,
                          "All respondents, including immigrants"),
    "filtered":          ("filt",   None,
                          "Native-born respondents speaking the dominant "
                          "language at home"),
    "cankaz":            ("cankaz", None,
                          "Filtered, with Canada and Kazakhstan split by "
                          "language spoken at home"),
    "uspolitics":        ("uspol",  None,
                          "Filtered, with the USA split by party "
                          "identification and Canada by language"),
    "secondlanguage":    ("seclang", None,
                          "Second-most-spoken language in high-immigration "
                          "countries treated as its own population"),
    "male":              ("filt",   ("Q260", 1), "Filtered, men only"),
    "female":            ("filt",   ("Q260", 2), "Filtered, women only"),
    "age_16_29":         ("filt",   ("X003R2", 1), "Filtered, aged 16-29"),
    "age_30_49":         ("filt",   ("X003R2", 2), "Filtered, aged 30-49"),
    "age_50plus":        ("filt",   ("X003R2", 3), "Filtered, aged 50 and over"),
    "cde_16_29":         ("uspol",  ("X003R2", 1),
                          "USA/Canada split, aged 16-29"),
    "cde_30_49":         ("uspol",  ("X003R2", 2),
                          "USA/Canada split, aged 30-49"),
    "cde_50plus":        ("uspol",  ("X003R2", 3),
                          "USA/Canada split, aged 50 and over"),
}


def cohorts(force: bool = False, tags=None) -> dict[str, Cohort]:
    """Build (or load from cache) the requested cohorts.

    With no ``tags`` this returns every cohort in ``COHORT_SPECS``.
    """
    tags = list(tags or COHORT_SPECS)
    if not force and all((CACHE / f"{t}_means_imp.pkl").exists() for t in tags):
        return {t: build_cohort(t, None) for t in tags}

    raw = load_raw()
    filt = native_filter(raw)
    parents = {
        "raw": raw,
        "filt": filt,
        "cankaz": split_can_kaz(filt),
        "uspol": split_us_politics(filt),
        "seclang": split_second_language(raw),
    }

    out = {}
    for tag in tags:
        parent, selector, _ = COHORT_SPECS[tag]
        df = parents[parent]
        if selector is not None:
            col, val = selector
            df = df[df[col] == val]
        out[tag] = build_cohort(tag, df, force)
    return out


if __name__ == "__main__":
    cs = cohorts(force=bool(os.environ.get("FORCE")))
    for tag, c in cs.items():
        print(f"{tag:12s} n={c.n_total:7d} countries={len(c.countries):3d} "
              f"items={c.means_imp.shape[1]}")
