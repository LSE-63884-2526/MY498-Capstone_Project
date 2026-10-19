"""
Robustness checks
=================
  0. Fit analyzers (maskwhite, fixed n_components per signifier)
  1. Micro-cluster silhouette, 768d vs UMAP
  2. Mask+whiten validation (hak / hakkını, demokrasi / demokrasinin)
  3. Stylometry speaker-confound check

Console output is also saved to robustness_report.txt.
Verbose cluster dumps from the fits go to robustness_fit.log.
"""

import contextlib
import os
import re
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import silhouette_score, silhouette_samples

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from signifier_analysis import SignifierAnalyzer

ROOT   = HERE.parent
CORPUS = ROOT / "Data" / "corpus_main_processed.csv"
STOP   = ROOT / "stopwords_tr.txt"
REPORT = HERE / "robustness_report.txt"
FITLOG = HERE / "robustness_fit.log"

# n_components per signifier, fixed for reproducibility
CONFIGS = {"demokrasi": 10, "millet": 10, "halk": 15, "hak": 12}
COMMON = dict(ranking="both", max_df_ratio=1, dim_ratio=0.85,
              hide_common=True, top_words=100, root_len=5, min_word_len=3,
              verbose=False)
# (signifier, surface form) pairs for the mask+whiten validation
MASKWHITE_CHECKS = [("hak", "hakkını"), ("demokrasi", "demokrasinin")]


def silhouette_two_spaces(emb_full, emb_umap, labels, metric="cosine"):
    """
    emb_full : (n, 768) raw BERT embeddings
    emb_umap : (n, d)   UMAP-reduced embeddings (this term's n_components)
    labels   : (n,)     HDBSCAN labels, noise = -1
    Returns: overall scores (dict) + per-cluster DataFrame
    """
    labels = np.asarray(labels)
    mask = labels != -1          # drop noise
    n_total = len(labels)
    n_noise = int((labels == -1).sum())
    n_clustered = int(mask.sum())
    n_clusters = len(set(labels[mask]))

    if n_clusters < 2:
        raise ValueError(f"Silhouette needs at least 2 clusters, found {n_clusters}.")

    Xf = np.asarray(emb_full)[mask]
    Xu = np.asarray(emb_umap)[mask]
    y = labels[mask]

    # overall scores
    sil_full = silhouette_score(Xf, y, metric=metric)
    sil_umap = silhouette_score(Xu, y, metric=metric)

    # per-cluster mean silhouette, to find weak clusters
    samp_full = silhouette_samples(Xf, y, metric=metric)
    samp_umap = silhouette_samples(Xu, y, metric=metric)

    rows = []
    for c in sorted(set(y)):
        cm = y == c
        rows.append({
            "cluster": int(c),
            "n": int(cm.sum()),
            "sil_768d": round(float(samp_full[cm].mean()), 3),
            "sil_umap": round(float(samp_umap[cm].mean()), 3),
        })
    per_cluster = pd.DataFrame(rows).sort_values("sil_768d").reset_index(drop=True)

    summary = {
        "n_total": n_total,
        "n_clustered": n_clustered,
        "n_noise": n_noise,
        "noise_pct": round(n_noise / n_total * 100, 1),
        "n_clusters": n_clusters,
        "silhouette_768d": round(float(sil_full), 3),
        "silhouette_umap": round(float(sil_umap), 3),
    }
    return summary, per_cluster


def report_all(signifiers, metric="cosine"):
    """
    signifiers: {'hak': (emb_full, emb_umap, labels), 'halk': (...), ...}
    Reports all four terms side by side in one table.
    """
    summ_rows = []
    per_cluster_all = {}
    for name, (ef, eu, lb) in signifiers.items():
        s, pc = silhouette_two_spaces(ef, eu, lb, metric=metric)
        s = {"signifier": name, **s}
        summ_rows.append(s)
        per_cluster_all[name] = pc
    summary_df = pd.DataFrame(summ_rows)[
        ["signifier", "n_clusters", "n_clustered", "noise_pct",
         "silhouette_768d", "silhouette_umap"]
    ]
    return summary_df, per_cluster_all



VENUES = ["rally", "press", "tv", "parliament", "event"]


# ================================================================
# CATEGORY B — METHOD VALIDATION
#   Shows that masking + whitening disperses the morphological surface.
#   Run on one or two representative terms (usually hak), not on every term.
# ================================================================

def hhi_from_forms(forms_df, form):
    """Herfindahl-Hirschman Index over forms_across_clusters() output.
    HHI = sum(p_i^2), p_i = the form's share in cluster i. N_eff = 1/HHI.
    Low HHI / high N_eff = the form is spread across clusters, i.e. masking+whitening worked.
    form: the surface form searched (column n_<form>).
    """
    col = f"n_{form}"
    counts = forms_df[col].values.astype(float)
    total = counts.sum()
    if total == 0:
        return None
    p = counts / total
    hhi = float((p ** 2).sum())
    n_eff = 1.0 / hhi if hhi > 0 else float("inf")
    # 90% coverage: how many clusters carry a cumulative 90%
    sorted_p = np.sort(p)[::-1]
    cum = np.cumsum(sorted_p)
    n_90 = int(np.searchsorted(cum, 0.90) + 1)
    return {
        "form": form,
        "total_tokens": int(total),
        "clusters_with_form": int((counts > 0).sum()),
        "HHI": round(hhi, 4),
        "N_eff": round(n_eff, 1),
        "clusters_90pct": n_90,
    }


def validate_mask_whiten(analyzer_normal, analyzer_maskwhite, form):
    """Masking+whitening check: same form, normal vs maskwhite pipeline.
    Both analyzers must be fitted (version='normal' and 'maskwhite').
    Returns a side-by-side DataFrame. N_eff should rise clearly under maskwhite.
    """
    fn = analyzer_normal.forms_across_clusters(form=form, exact=True)
    fm = analyzer_maskwhite.forms_across_clusters(form=form, exact=True)
    hn = hhi_from_forms(fn, form)
    hm = hhi_from_forms(fm, form)
    rows = []
    for metric in ["total_tokens", "clusters_with_form", "HHI", "N_eff", "clusters_90pct"]:
        rows.append({
            "metric": metric,
            "normal": hn[metric] if hn else None,
            "maskwhite": hm[metric] if hm else None,
        })
    return pd.DataFrame(rows)

def venue_check(m, cluster_col="cluster", venue_col="venue",
                min_cluster_n=10, flag=0.60):
    """Dominant venue share per cluster, against the corpus-wide distribution.

    flag: mark a cluster if more than this share comes from a single venue.
    """
    df = m[m[cluster_col] != -1]                 # drop noise

    corpus = df[venue_col].value_counts(normalize=True)
    print("corpus venue distribution:", dict(corpus.round(2)))
    print()

    for cid in sorted(df[cluster_col].unique()):
        members = df[df[cluster_col] == cid]
        if len(members) < min_cluster_n:
            continue
        vc = members[venue_col].value_counts(normalize=True)
        dom, share = vc.index[0], vc.iloc[0]
        expected = corpus.get(dom, 0)
        mark = "  <-- venue-skewed" if (share > flag and share - expected > 0.2) else ""
        print(f"c{cid}: n={len(members)} {dom} {share*100:.0f}% (expected {expected*100:.0f}%){mark}")

"""
Stylometry speaker-confound check.

Purpose: the Özel phase (P3) changes SPEAKER and PERIOD at once (Özel replaces
Kılıçdaroğlu, 2023-11). No in-corpus test can separate the two (zero speaker
overlap). This script instead BOUNDS the idiolect gap: how stylistically
different are the two speakers, and is that gap larger than the within-speaker
(P1 vs P2) period drift?

Content-free stylometry only (function words, sentence/word length, lexical
diversity), so it measures STYLE, not the signifiers' content.

Key comparison:
    within-KK (P1 vs P2, same speaker)  vs  KK->Özel (P2 vs P3, speaker change)
If the KK->Özel style gap >> within-KK gap, an idiolect component is present in
the P3 embedding drift and must be acknowledged (limitations).

Reads Data/corpus_main_processed.csv (text_raw, date, speaker) and stopwords_tr.txt.
"""


def tr_lower(t):
    return (str(t).replace("İ", "i").replace("I", "ı").replace("Ğ", "ğ")
            .replace("Ş", "ş").replace("Ç", "ç").replace("Ö", "ö")
            .replace("Ü", "ü").lower())


def phase_of(ts):
    if pd.isna(ts):
        return 0
    if ts < pd.Timestamp("2019-04-01"):
        return 1
    if ts < pd.Timestamp("2023-11-05"):
        return 2
    return 3


def run(corpus=CORPUS, stop=STOP, mattr_window=250, seed=0):
    df = pd.read_csv(corpus)
    df["dt"] = pd.to_datetime(df["date"], dayfirst=True, errors="coerce")
    df["phase"] = df["dt"].apply(phase_of)
    df = df[df["text_raw"].notna()].copy()
    df["low"] = df["text_raw"].astype(str).map(tr_lower)

    # dedup + lower stopwords (they are the function-word feature set)
    sw = sorted(set(tr_lower(w.strip()) for w in open(stop, encoding="utf-8") if w.strip()))

    # ---- 1) aggregate style per phase ----
    print("=== Aggregate stylometric features per phase ===")
    print("phase  n_docs  words/sent  meanWordLen  MATTR")
    for p in (1, 2, 3):
        sub = df[df["phase"] == p]["low"]
        toks, wps, wl = [], [], []
        for t in sub:
            sents = [s for s in re.split(r"[.!?]+", t) if s.strip()]
            w = re.findall(r"\b\w+\b", t)
            if sents:
                wps.append(len(w) / len(sents))
            wl += [len(x) for x in w]
            toks += w
        W = mattr_window
        ttrs = [len(set(toks[i:i + W])) / W for i in range(0, max(1, len(toks) - W), W)]
        print(f"  {p}    {len(sub):5d}    {np.mean(wps):6.2f}     "
              f"{np.mean(wl):6.2f}      {np.mean(ttrs):.3f}")

    # ---- 2) function-word style vectors (content-free) ----
    vec = TfidfVectorizer(vocabulary=sw, use_idf=False, norm="l1")  # relative freq
    X = vec.fit_transform(df["low"]).toarray()
    Xs = (X - X.mean(0)) / (X.std(0) + 1e-9)
    cent = {p: Xs[(df["phase"] == p).values].mean(0) for p in (1, 2, 3)}
    d = lambda a, b: float(np.linalg.norm(cent[a] - cent[b]))
    print(f"\n=== Function-word style centroid distances (standardized, {len(sw)} words) ===")
    print(f"  d(P1,P2) within-KK, cross-period    = {d(1,2):.2f}")
    print(f"  d(P2,P3) KK->Özel (speaker change)  = {d(2,3):.2f}  ({d(2,3)/d(1,2):.1f}x within-KK)")
    print(f"  d(P1,P3) KK-Early -> Özel           = {d(1,3):.2f}")

    # ---- 3) style classifiers (function words only) ----
    kk = df["phase"].isin([1, 2]).values
    ysp = (df["phase"] == 3).astype(int).values
    auc_sp = cross_val_score(LogisticRegression(max_iter=2000), X, ysp,
                             cv=5, scoring="roc_auc").mean()
    yper = (df[kk]["phase"] == 2).astype(int).values
    auc_per = cross_val_score(LogisticRegression(max_iter=2000), X[kk], yper,
                              cv=5, scoring="roc_auc").mean()
    print("\n=== Style classifiers (function words ONLY, 5-fold CV ROC-AUC) ===")
    print(f"  KK vs Özel (SPEAKER+period)      : AUC = {auc_sp:.3f}")
    print(f"  P1 vs P2 within KK (PERIOD only) : AUC = {auc_per:.3f}   (placebo)")
    return dict(d12=d(1, 2), d23=d(2, 3), d13=d(1, 3),
                auc_speaker=auc_sp, auc_period=auc_per)


# ================================================================
# PIPELINE
# ================================================================

class Tee:
    """Write to the console and the report file at the same time."""
    def __init__(self, *streams):
        self.streams = streams

    def write(self, s):
        for st in self.streams:
            st.write(s)

    def flush(self):
        for st in self.streams:
            st.flush()


def section(title):
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def fit_analyzer(word, version, log):
    """Fit one analyzer. Its printed cluster dumps go to the log file."""
    with contextlib.redirect_stdout(log):
        a = SignifierAnalyzer(word, version=version)
        a.fit(n_components=CONFIGS[word], **COMMON)
    return a


def n_clusters_of(a):
    return int((a.m["cluster"].unique() != -1).sum())


def main():
    os.chdir(HERE)
    warnings.filterwarnings("ignore", category=UserWarning)
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.max_rows", 200)

    with open(REPORT, "w", encoding="utf-8") as rep, \
         open(FITLOG, "w", encoding="utf-8") as log:
        sys.stdout = Tee(sys.__stdout__, rep)
        try:
            # ---- 0) fit ----
            section("0. Fitting maskwhite analyzers")
            analyzers = {}
            for w in CONFIGS:
                analyzers[w] = fit_analyzer(w, "maskwhite", log)
                a = analyzers[w]
                print(f"  {w:<10} n={len(a.m):6d}  n_components={CONFIGS[w]:2d}  "
                      f"clusters={n_clusters_of(a)}")

            # ---- 1) silhouette ----
            section("1. Micro-cluster silhouette (cosine, noise excluded)")
            sig = {w: (a.v, a.clu, a.m["cluster"].values)
                   for w, a in analyzers.items()}
            sil_summary, _ = report_all(sig)
            print(sil_summary.to_string(index=False))

            # ---- 2) mask+whiten validation ----
            section("2. Mask+whiten validation (normal vs maskwhite)")
            for w, form in MASKWHITE_CHECKS:
                normal = fit_analyzer(w, "normal", log)
                print(f"\n--- {w} / '{form}' ---")
                print(f"  normal UMAP shape : {normal.clu.shape}")
                print(f"  normal clusters   : {n_clusters_of(normal)}")
                print(f"  maskwhite clusters: {n_clusters_of(analyzers[w])}")
                print(validate_mask_whiten(normal, analyzers[w], form=form)
                      .to_string(index=False))

            # ---- 3) stylometry ----
            section("3. Stylometry speaker-confound check")
            run()

        finally:
            sys.stdout = sys.__stdout__

    print(f"\nReport saved to {REPORT}")
    print(f"Fit dumps saved to {FITLOG}")


if __name__ == "__main__":
    main()