# ============================================================
# PPMI CO-OCCURRENCE SHIFT  (dislocation / event analysis)
# ============================================================
# Compares an ego signifier's PPMI neighbour profile before vs. after an event.
#   - ppmi_shift():                before/after neighbour lists + Jaccard + cosine
#   - dislocation_cosine():        1 - cosine, venue-balanced, permutation null
#   - dislocation_cosine_matrix(): ego x event table
#
# PPMI(ego, w) = max(0, log2[ P(w | ego context) / P(w) ]): how far w is
# over-represented within +/- window words of ego relative to its own base rate.
# Frequency-normalised, so no masking/whitening needed (raw corpus, not embeddings).
# ============================================================

import re
import math
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

# ---- CONFIG ----
ROOT        = Path(__file__).resolve().parent.parent
CORPUS_PATH = ROOT / "Data" / "corpus_main_processed.csv"
STOP_PATH   = ROOT / "stopwords_tr.txt"
TEXT_COL    = "text_clean"     # surface content words (text_lemma also works)
VENUE_COL   = "venue"
WINDOW      = 10               # +/- word context window
MIN_COUNT   = 5               # min corpus frequency for a context word
DEFAULT_MONTHS = 16

TOKEN_RX = re.compile(r"\w+", re.UNICODE)

_CORPUS = None
_STOP = None


def load_corpus(path=CORPUS_PATH):
    """Load the corpus once (parse dates, reset index). Cached."""
    global _CORPUS
    if _CORPUS is None:
        df = pd.read_csv(path)
        df["date"] = pd.to_datetime(df["date"], dayfirst=True, errors="coerce")
        _CORPUS = df.reset_index(drop=True)
    return _CORPUS


def load_stopwords(path=STOP_PATH):
    global _STOP
    if _STOP is None:
        _STOP = set(w.strip().lower() for w in open(path, encoding="utf-8") if w.strip())
    return _STOP


def _tokens(text):
    return TOKEN_RX.findall(str(text).lower())


def compute_ppmi(texts, ego, window=WINDOW, min_count=MIN_COUNT, stopwords=None):
    """Windowed PPMI within +/- window of ego (stem/regex, prefix-anchored).
    texts: documents for the period; marginal frequencies come from all of them.
    Returns (ppmi: {word: score}, ego occurrence count)."""
    stop = stopwords if stopwords is not None else set()
    ego_rx = re.compile(ego)
    f = Counter()      # marginal token frequency
    co = Counter()     # co-occurrence inside the ego window
    ego_occ = 0
    for text in texts:
        toks = _tokens(text)
        for t in toks:
            f[t] += 1
        pos = [i for i, t in enumerate(toks) if ego_rx.match(t)]
        for i in pos:
            ego_occ += 1
            lo, hi = max(0, i - window), min(len(toks), i + window + 1)
            for j in range(lo, hi):
                if j == i or ego_rx.match(toks[j]):
                    continue
                co[toks[j]] += 1

    def ok(w):
        return len(w) > 2 and w not in stop and not w.isdigit()

    sum_f = sum(v for w, v in f.items() if ok(w))
    sum_co = sum(v for w, v in co.items() if ok(w))
    if sum_co == 0 or sum_f == 0:
        return {}, ego_occ
    ppmi = {}
    for w, c in co.items():
        if not ok(w) or f[w] < min_count:
            continue
        pmi = math.log2((c / sum_co) / (f[w] / sum_f))
        if pmi > 0:
            ppmi[w] = pmi
    return ppmi, ego_occ


def ppmi_shift(ego, event_date, months=DEFAULT_MONTHS, top_k=50, window=WINDOW,
               min_count=MIN_COUNT, text_col=TEXT_COL, speaker=None,
               corpus=None, stopwords=None, verbose=True):
    """Ego's PPMI neighbours before vs. after an event date.
    Returns dict(jaccard, cosine, top_before, top_after, new_after, counts...)."""
    df = corpus if corpus is not None else load_corpus()
    stop = stopwords if stopwords is not None else load_stopwords()
    if speaker:
        df = df[df["speaker"] == speaker]
    ev = pd.Timestamp(event_date)
    b0, a1 = ev - pd.DateOffset(months=months), ev + pd.DateOffset(months=months)
    before = df[(df["date"] >= b0) & (df["date"] < ev)]
    after  = df[(df["date"] >= ev) & (df["date"] < a1)]

    pb, eb = compute_ppmi(before[text_col].tolist(), ego, window, min_count, stop)
    pa, ea = compute_ppmi(after[text_col].tolist(),  ego, window, min_count, stop)
    top_b = [w for w, _ in sorted(pb.items(), key=lambda x: -x[1])[:top_k]]
    top_a = [w for w, _ in sorted(pa.items(), key=lambda x: -x[1])[:top_k]]
    sb, sa = set(top_b), set(top_a)
    jac = len(sb & sa) / len(sb | sa) if (sb | sa) else float("nan")
    va, vb, _ = _ppmi_vector(pb, pa)
    cos = _cosine(va, vb) if va is not None else float("nan")
    new = [w for w in top_a if w not in sb]     # after-only neighbours

    # documents containing ego
    ndocs_b = int(before[text_col].astype(str).str.contains(ego, case=False, na=False).sum())
    ndocs_a = int(after[text_col].astype(str).str.contains(ego, case=False, na=False).sum())

    res = {
        "ego": ego, "event": event_date, "months": months, "window": window,
        "n_before_docs": len(before), "n_after_docs": len(after),
        "n_before_ego": ndocs_b, "n_after_ego": ndocs_a,
        "ego_occ_before": eb, "ego_occ_after": ea,
        "jaccard": round(jac, 3), "cosine": round(cos, 3),
        "top_before": [(w, round(pb[w], 2)) for w in top_b],
        "top_after": [(w, round(pa[w], 2)) for w in top_a],
        "new_after": [(w, round(pa[w], 2)) for w in new],
    }
    if verbose:
        print(f"[{ego}] {event_date} ±{months}mo win±{window} | window docs "
              f"{len(before)}->{len(after)} | ego-docs {ndocs_b}->{ndocs_a} "
              f"| ego occ {eb}->{ea}")
        print(f"  Jaccard(top{top_k}) = {res['jaccard']}   cosine = {res['cosine']}")
        print(f"  NEW neighbours after (top {min(top_k, 40)}, PPMI):")
        print("   ", ", ".join(f"{w}({p})" for w, p in res["new_after"][:40]))
    return res


# ---------- dislocation cosine (event scan) ----------

def _ppmi_vector(ppmi_a, ppmi_b):
    """Align two PPMI dicts into vectors on a shared vocabulary axis."""
    vocab = sorted(set(ppmi_a) | set(ppmi_b))
    if not vocab:
        return None, None, vocab
    va = np.array([ppmi_a.get(w, 0.0) for w in vocab])
    vb = np.array([ppmi_b.get(w, 0.0) for w in vocab])
    return va, vb, vocab


def _cosine(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return np.nan
    return float(a @ b / (na * nb))


def _venue_balanced_sample(df_before, df_after, rng):
    """Keep an equal number of speeches per venue in before and after (confounder control)."""
    if VENUE_COL not in df_before.columns:
        return df_before, df_after, None
    vb = df_before[VENUE_COL].value_counts()
    va = df_after[VENUE_COL].value_counts()
    common = set(vb.index) & set(va.index)
    keep_before, keep_after, per_venue = [], [], {}
    for v in common:
        n = min(vb[v], va[v])
        per_venue[v] = n
        idx_b = df_before[df_before[VENUE_COL] == v].index.to_numpy()
        idx_a = df_after[df_after[VENUE_COL] == v].index.to_numpy()
        keep_before += rng.choice(idx_b, n, replace=False).tolist()
        keep_after += rng.choice(idx_a, n, replace=False).tolist()
    return df_before.loc[keep_before], df_after.loc[keep_after], per_venue


def dislocation_cosine(ego, event_date, months=DEFAULT_MONTHS, balance_venue=True,
                       n_perm=200, seed=42, window=WINDOW, min_count=MIN_COUNT,
                       text_col=TEXT_COL, speaker=None, corpus=None, stopwords=None,
                       verbose=True):
    """Neighbourhood shift around an event = 1 - cosine. Venue-balanced, with a
    permutation null: the shift is significant if the observed cosine falls below it."""
    df = corpus if corpus is not None else load_corpus()
    stop = stopwords if stopwords is not None else load_stopwords()
    if speaker:
        df = df[df["speaker"] == speaker]
    rng = np.random.default_rng(seed)
    ev = pd.Timestamp(event_date)
    b0, a1 = ev - pd.DateOffset(months=months), ev + pd.DateOffset(months=months)
    before = df[(df["date"] >= b0) & (df["date"] < ev)].copy()
    after  = df[(df["date"] >= ev) & (df["date"] < a1)].copy()
    if balance_venue:
        before, after, per_venue = _venue_balanced_sample(before, after, rng)
    else:
        per_venue = None

    tb, ta = before[text_col].tolist(), after[text_col].tolist()
    ppmi_b, ego_b = compute_ppmi(tb, ego, window, min_count, stop)
    ppmi_a, ego_a = compute_ppmi(ta, ego, window, min_count, stop)
    va, vb, vocab = _ppmi_vector(ppmi_b, ppmi_a)
    if va is None:
        return {"ego": ego, "event": event_date, "cosine": np.nan, "note": "insufficient neighbours"}
    cos = _cosine(va, vb)
    result = {
        "ego": ego, "event": event_date, "months": months,
        "cosine": round(cos, 4), "dislocation": round(1 - cos, 4),
        "n_before": len(tb), "n_after": len(ta),
        "ego_before": ego_b, "ego_after": ego_a, "n_neighbors": len(vocab),
        "venue_balanced": balance_venue, "per_venue": per_venue,
    }
    if n_perm > 0:
        pooled = tb + ta
        nb = len(tb)
        null_cos = []
        for _ in range(n_perm):
            perm = rng.permutation(len(pooled))
            g1 = [pooled[i] for i in perm[:nb]]
            g2 = [pooled[i] for i in perm[nb:]]
            p1, _ = compute_ppmi(g1, ego, window, min_count, stop)
            p2, _ = compute_ppmi(g2, ego, window, min_count, stop)
            x, y, _ = _ppmi_vector(p1, p2)
            if x is not None:
                null_cos.append(_cosine(x, y))
        null_cos = np.array([c for c in null_cos if not np.isnan(c)])
        if len(null_cos):
            result["p_value"] = round(float((null_cos <= cos).mean()), 4)
            result["null_cosine_mean"] = round(float(null_cos.mean()), 4)
            result["null_cosine_std"] = round(float(null_cos.std()), 4)
    if verbose:
        print(f"[{ego}] {event_date} ±{months}mo | cosine={result['cosine']} "
              f"dislocation={result['dislocation']} | n_b={result['n_before']} "
              f"n_a={result['n_after']} ego={ego_b}/{ego_a}"
              + (f" | p={result.get('p_value', '-')}" if n_perm else ""))
    return result


def dislocation_cosine_matrix(egos, events, months=DEFAULT_MONTHS, balance_venue=True,
                              n_perm=100, seed=42, **kw):
    """Full ego x event matrix. Cell = dislocation (1 - cosine) + p."""
    rows = []
    for ev in events:
        row = {"event": ev}
        for ego in egos:
            r = dislocation_cosine(ego, ev, months=months, balance_venue=balance_venue,
                                   n_perm=n_perm, seed=seed, verbose=False, **kw)
            row[ego] = r.get("dislocation", np.nan)
            row[f"{ego}_p"] = r.get("p_value", np.nan)
        rows.append(row)
    return pd.DataFrame(rows).set_index("event")
