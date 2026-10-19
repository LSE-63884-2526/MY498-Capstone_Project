"""
Phase-wise lexical + semantic triangulation
===========================================
Validates findings from the cluster/embedding analysis (the economic-demand
register shift, the phase behaviour of Six Arrows / halkçılık vocabulary)
against the raw lemma corpus.

Three layers:
  1) phase_frequency -> normalised frequency of a word set across the three phases
  2) phase_ppmi      -> phase-wise PPMI of a signifier's neighbours (association)
  3) triangulate     -> overlap between cluster neighbours and PPMI neighbours

PPMI logic and filter thresholds match ppmi_dislocation.
Corpus: text_lemma (whitespace-tokenised), date column.
"""

import pandas as pd
import numpy as np
from collections import Counter, defaultdict
from math import log2
from pathlib import Path
import re


# ============================================================
# SETTINGS (kept consistent with the PPMI module)
# ============================================================
ROOT = Path(__file__).resolve().parent.parent
CORPUS_PATH = ROOT / "Data" / "corpus_main_processed.csv"
STOP_PATH = ROOT / "stopwords_tr.txt"
TEXT_COL = "text_lemma"
DATE_COL = "date"

WINDOW   = 10   # ±N word neighbourhood window
MIN_COOC = 8    # min co-occurrences with ego inside the window
MIN_FREQ = 10   # min global frequency in that phase
MIN_DOCS = 3    # min distinct speeches sharing a window with ego

# ============================================================
# PHASE DEFINITION
#   KK-Early : 2015-01-01  ..  2019-04-01
#   KK-Late  : 2019-04-01  ..  2023-11-08
#   Özel     : 2023-11-08  ..  today
# ============================================================
PHASE_BOUNDS = [
    ("KK-Early", pd.Timestamp("2015-01-01"), pd.Timestamp("2019-04-01")),
    ("KK-Late",  pd.Timestamp("2019-04-01"), pd.Timestamp("2023-11-08")),
    ("Ozel",     pd.Timestamp("2023-11-08"), pd.Timestamp("2100-01-01")),
]
PHASES = [p[0] for p in PHASE_BOUNDS]


def assign_phase(dt):
    for name, start, end in PHASE_BOUNDS:
        if start <= dt < end:
            return name
    return None


# ============================================================
# LOAD CORPUS + ADD PHASE COLUMN
# ============================================================
def load_corpus(path=CORPUS_PATH, date_format=None):
    """Load the corpus, parse dates, add a phase column.
    date_format: the PPMI module uses '%d.%m.%Y'; None auto-detects."""
    df = pd.read_csv(path)
    if date_format:
        df[DATE_COL] = pd.to_datetime(df[DATE_COL], format=date_format, errors="coerce")
    else:
        df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
    df = df.dropna(subset=[DATE_COL, TEXT_COL]).reset_index(drop=True)
    df["phase"] = df[DATE_COL].apply(assign_phase)
    df = df.dropna(subset=["phase"]).reset_index(drop=True)
    return df

# ============================================================
# SIX ARROWS LEXICAL SCAN (raw 'text' column, regex; not lemma)
# ============================================================
def scan_sixarrow(corpus_df, patterns=None, text_col="text", phase_col="phase"):
    """Lexical triangulation: corpus-wide, phase-normalised frequency scan for the
    Six Arrows and their derivatives. Evidence independent of the embedding clustering.

    corpus_df : the full raw corpus, not signifier occurrences
    patterns  : {name: regex}; defaults to the Six Arrows + derivatives
    Returns   : DataFrame (term x phase counts and percentages)
    """
    import pandas as pd

    if patterns is None:
        patterns = {
            "Milliyetçilik":   r"milliyetçi(?:li[kğ])?",
            "Devletçilik":     r"devletçi(?:li[kğ])?",
            "Cumhuriyetçilik": r"cumhuriyetçi(?:li[kğ])?",
            "Halkçılık":       r"halkçı(?:lı[kğ])?",
            "Laiklik":         r"laik(?:li[kğ])?",
            "İnkılap":         r"[iı]nkıla[pb](?:çılı[kğ])?",
            "İhtilal":         r"ihtilal",
            "Devrim":          r"devrim(?:ci(?:li[kğ])?)?",
        }

    phase_totals = corpus_df.groupby(phase_col).size()
    rows = []
    for name, pat in patterns.items():
        mask = corpus_df[text_col].str.contains(pat, case=False, na=False, regex=True)
        row = {"term": name, "total": int(mask.sum())}
        for ph in sorted(phase_totals.index):
            n = int((mask & (corpus_df[phase_col] == ph)).sum())
            row[f"f{ph}_n"] = n
            row[f"f{ph}_%"] = round(n / phase_totals[ph] * 100, 2)
        rows.append(row)
    return pd.DataFrame(rows).set_index("term")


# ============================================================
# DISLOCATION KEYNESS (log-likelihood, signifier-independent) — Dunning 1993
# Words that enter (LL+) and leave (LL-) the discourse before vs. after a
# dislocation/election date. A demand or identity that peaks before an election
# and drops after it shows up on the LL- side. Not conditioned on a signifier.
# Self-contained: loads its own corpus and stopwords.
# ============================================================
def dislocation_keyness(event_date, window_months=4,
                        before_months=None, before_weeks=None,
                        after_months=None, after_weeks=None,
                        corpus=None, corpus_path=CORPUS_PATH,
                        text_col=TEXT_COL, date_col=DATE_COL, date_format=None,
                        speaker=None, speaker_col="speaker",
                        min_freq=5, min_docs=5, min_word_len=3, top_n=40,
                        stop_path=STOP_PATH, verbose=True):
    """Dunning (1993) log-likelihood keyness over the windows before
    (-window_months) and after (+window_months) a dislocation/election date.
    Signifier-independent: which words entered or left the discourse corpus-wide.

    Parameters
    ----------
    event_date    : 'YYYY-MM-DD' or Timestamp (dislocation/election date)
    window_months : window radius on each side of the event, in months
    corpus        : ready DataFrame (overrides corpus_path)
    corpus_path   : loaded from here when no DataFrame is given
    text_col      : lemma/clean text column ; date_col: date column
    date_format   : date parse format (None -> auto ; '%d.%m.%Y' for the raw CSV)
    speaker       : restrict to this speaker (via speaker_col)
    min_freq      : min total frequency (before+after)
    min_docs      : word must occur in this many distinct speeches in at least one
                    period (filters out single-rally artefacts)
    min_word_len  : minimum word length ; top_n: words per direction
    stop_path     : stopword file

    Returns: (entered, left, full)
        entered = words entering the discourse (largest LL+)
        left    = words leaving it (smallest LL-; demand/identity decline shows here)
        full    = all scored words (word, before, after,
                  docs_before, docs_after, ll)
    """
    import re
    from collections import Counter
    import numpy as np
    import pandas as pd

    word_rx = re.compile(r"\b[a-zçğıöşü_]+\b")
    stop = set(w.strip() for w in open(stop_path, encoding="utf-8") if w.strip())

    # --- corpus + windows ---
    df = corpus.copy() if corpus is not None else pd.read_csv(corpus_path)
    df[date_col] = (pd.to_datetime(df[date_col], format=date_format, errors="coerce")
                    if date_format else pd.to_datetime(df[date_col], errors="coerce"))
    df = df.dropna(subset=[date_col, text_col])
    if speaker:
        df = df[df[speaker_col] == speaker]
    ev = pd.Timestamp(event_date)
    # asymmetric windows; falls back to symmetric window_months
    def _off(mo, wk):
        return (pd.DateOffset(months=window_months) if (mo is None and wk is None)
                else pd.DateOffset(months=mo or 0, weeks=wk or 0))
    b0 = ev - _off(before_months, before_weeks)
    a1 = ev + _off(after_months, after_weeks)
    before = df[(df[date_col] >= b0) & (df[date_col] < ev)]
    after  = df[(df[date_col] >= ev) & (df[date_col] < a1)]

    # --- frequency + document frequency ---
    def counts(texts):
        c, dfreq = Counter(), Counter()
        for t in texts:
            seen = set()
            for w in word_rx.findall(str(t).lower()):
                if len(w) >= min_word_len and w not in stop:
                    c[w] += 1
                    seen.add(w)
            for w in seen:
                dfreq[w] += 1
        return c, dfreq

    cb, dfb = counts(before[text_col])
    ca, dfa = counts(after[text_col])
    Nb, Na = sum(cb.values()), sum(ca.values())
    if Nb == 0 or Na == 0:
        raise ValueError(f"Empty window: before={len(before)} sp, after={len(after)} sp.")

    # --- signed Dunning log-likelihood (+ entering, - leaving) ---
    def ll(a, b):
        e_a, e_b = Na * (a + b) / (Na + Nb), Nb * (a + b) / (Na + Nb)
        val = 0.0
        if a > 0:
            val += a * np.log(a / e_a)
        if b > 0:
            val += b * np.log(b / e_b)
        val *= 2
        return (1 if (a / Na) > (b / Nb) else -1) * val

    rows = []
    for w in set(cb) | set(ca):
        a, b = ca.get(w, 0), cb.get(w, 0)
        if a + b < min_freq:
            continue
        da, db = dfa.get(w, 0), dfb.get(w, 0)
        if max(da, db) < min_docs:
            continue
        lr = np.log2(((a + 0.5) / Na) / ((b + 0.5) / Nb))   # size-independent effect size (Hardie 2014)
        rows.append({"word": w, "before": b, "after": a,
                     "docs_before": db, "docs_after": da,
                     "ll": round(ll(a, b), 1), "log_ratio": round(lr, 2)})
    full = pd.DataFrame(rows).sort_values("ll", ascending=False).reset_index(drop=True)
    entered = full[full["docs_after"] >= min_docs].nlargest(top_n, "ll").reset_index(drop=True)
    left    = full[full["docs_before"] >= min_docs].nsmallest(top_n, "ll").reset_index(drop=True)

    if verbose:
        print(f"[{ev.date()}] before {b0.date()}→{ev.date()} ({len(before)} sp/{Nb} tok)"
              f" | after {ev.date()}→{a1.date()} ({len(after)} sp/{Na} tok)")
        for title, sub in [
            ("WORDS ENTERING THE DISCOURSE AFTER THE DISLOCATION (keyness+)", entered),
            ("WORDS LEAVING THE DISCOURSE WITH THE DISLOCATION (keyness-)", left),
        ]:
            print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")
            for _, r in sub.iterrows():
                print(f"  {r['word']:20} LL={r['ll']:7.1f}  LR={r['log_ratio']:+5.1f}  "
                      f"(freq {int(r['before'])}→{int(r['after'])}, "
                      f"docs {int(r['docs_before'])}→{int(r['docs_after'])})")
    return entered, left, full


# ============================================================
# EGO COLLOCATE KEYNESS (log-likelihood) — signifier-conditioned
# Scores an ego's +/- window collocates with Dunning LL: words over-represented
# in the ego window relative to the corpus baseline are the ego's distinctive
# context (its chain). The LL counterpart of PPMI, with less rare-word noise.
# With event_date, computes before/after separately plus Jaccard.
# ============================================================
def ego_keyness(ego, event_date=None, window_months=4, window=12,
                corpus=None, corpus_path=CORPUS_PATH, text_col=TEXT_COL,
                date_col=DATE_COL, date_format=None, speaker=None, speaker_col="speaker",
                min_cooc=5, min_docs=5, top_n=40,
                stop_path=STOP_PATH, verbose=True):
    """Ego collocate keyness (Dunning LL). ego: stem/prefix, matched with re.match.
    event_date None -> single-period collocate keyness over the whole corpus, DataFrame.
    event_date given -> before/after keyness + Jaccard + new_after. Returns a dict."""
    import re
    from collections import Counter
    import numpy as np
    import pandas as pd

    word_rx = re.compile(r"\b[a-zçğıöşü_]+\b")
    ego_rx = re.compile(ego)
    stop = set(w.strip() for w in open(stop_path, encoding="utf-8") if w.strip())

    df = corpus.copy() if corpus is not None else pd.read_csv(corpus_path)
    df[date_col] = (pd.to_datetime(df[date_col], format=date_format, errors="coerce")
                    if date_format else pd.to_datetime(df[date_col], errors="coerce"))
    df = df.dropna(subset=[date_col, text_col])
    if speaker:
        df = df[df[speaker_col] == speaker]

    def collocates(texts):
        """Returns: co (collocate frequency in the ego window), cdoc (collocate doc
        freq), f (period-wide word frequency), n_win (total window slots), N (total
        content tokens), ego_occ."""
        co, cdoc, f = Counter(), Counter(), Counter()
        n_win = N = ego_occ = 0
        for t in texts:
            toks = [w for w in word_rx.findall(str(t).lower())]
            for w in toks:
                if len(w) > 2 and w not in stop:
                    f[w] += 1; N += 1
            pos = [i for i, w in enumerate(toks) if ego_rx.match(w)]
            seen = set()
            for i in pos:
                ego_occ += 1
                lo, hi = max(0, i - window), min(len(toks), i + window + 1)
                for j in range(lo, hi):
                    w = toks[j]
                    if j == i or ego_rx.match(w) or len(w) <= 2 or w in stop:
                        continue
                    co[w] += 1; n_win += 1; seen.add((w, id(t)))
            for w, _ in seen:
                cdoc[w] += 1
        return co, cdoc, f, n_win, N, ego_occ

    def score(texts):
        co, cdoc, f, n_win, N, ego_occ = collocates(texts)
        rows = []
        for w, o_in in co.items():
            if o_in < min_cooc or cdoc.get(w, 0) < min_docs:
                continue
            ft, o_out, n_out = f[w], f[w] - o_in, N - n_win
            e_in = n_win * ft / N if N else 0
            ll = 0.0
            if o_in > 0 and e_in > 0:
                ll += o_in * np.log(o_in / e_in)
            if o_out > 0 and (n_out * ft / N) > 0:
                ll += o_out * np.log(o_out / (n_out * ft / N))
            ll *= 2
            if (o_in / n_win) < (ft / N):    # keep over-represented collocates only
                continue
            lr = np.log2((o_in / n_win) / (ft / N))   # PMI-like, size-independent effect
            rows.append({"word": w, "cooc": o_in, "docs": cdoc[w],
                         "ll": round(ll, 1), "log_ratio": round(lr, 2)})
        out = pd.DataFrame(rows).sort_values("ll", ascending=False).reset_index(drop=True)
        return out, ego_occ

    if event_date is None:
        tab, ego_occ = score(df[text_col])
        if verbose:
            print(f"[{ego}] single period | ego occ={ego_occ} | collocate keyness (LL):")
            for _, r in tab.head(top_n).iterrows():
                print(f"  {r['word']:20} LL={r['ll']:7.1f}  LR={r['log_ratio']:+5.1f}  (cooc {int(r['cooc'])}, docs {int(r['docs'])})")
        return tab.head(top_n)

    ev = pd.Timestamp(event_date)
    b0, a1 = ev - pd.DateOffset(months=window_months), ev + pd.DateOffset(months=window_months)
    before = df[(df[date_col] >= b0) & (df[date_col] < ev)]
    after  = df[(df[date_col] >= ev) & (df[date_col] < a1)]
    tab_b, eb = score(before[text_col])
    tab_a, ea = score(after[text_col])
    top_b = set(tab_b.head(top_n)["word"]); top_a = set(tab_a.head(top_n)["word"])
    jac = len(top_b & top_a) / len(top_b | top_a) if (top_b | top_a) else float("nan")
    new_after = tab_a[~tab_a["word"].isin(top_b)].head(top_n).reset_index(drop=True)
    if verbose:
        print(f"[{ego}] {ev.date()} ±{window_months}mo | ego occ {eb}->{ea} | "
              f"Jaccard(top{top_n})={round(jac, 3)}")
        print(f"\n{'=' * 70}\nEGO'S NEW CONTEXT (after, LL collocate keyness)\n{'=' * 70}")
        for _, r in new_after.head(top_n).iterrows():
            print(f"  {r['word']:20} LL={r['ll']:7.1f}  LR={r['log_ratio']:+5.1f}  (cooc {int(r['cooc'])}, docs {int(r['docs'])})")
    return {"before": tab_b, "after": tab_a, "new_after": new_after,
            "jaccard": round(jac, 3), "ego_before": eb, "ego_after": ea}


# ============================================================
# KWIC — keyword in context (raw text, text_raw)
# Aligned concordance lines for a stem/prefix within a date range, +/- window
# words of context. Supports keyness findings with close reading.
# ============================================================
def kwic(word, start, end, window=8, corpus=None, corpus_path=CORPUS_PATH,
         text_col="text_raw", date_col=DATE_COL, date_format=None,
         speaker=None, speaker_col="speaker", max_hits=50, verbose=True):
    """Keyword-in-context concordance (raw text).
    word     : stem/prefix (re.match, applied to Turkish-lowercased tokens)
    start,end: date range [start, end)
    window   : context words on each side of the keyword
    text_col : defaults to 'text_raw' (NOT lemma)
    max_hits : row cap for frequent words (None = all)
    Returns: DataFrame(date, venue, left, keyword, right)."""
    import re
    import pandas as pd

    def tr_lower(s):
        return (s.replace("İ", "i").replace("I", "ı").replace("Ş", "ş")
                 .replace("Ç", "ç").replace("Ö", "ö").replace("Ü", "ü")
                 .replace("Ğ", "ğ").lower())

    df = corpus.copy() if corpus is not None else pd.read_csv(corpus_path)
    df[date_col] = (pd.to_datetime(df[date_col], format=date_format, errors="coerce")
                    if date_format else pd.to_datetime(df[date_col], errors="coerce"))
    df = df.dropna(subset=[date_col, text_col])
    if speaker:
        df = df[df[speaker_col] == speaker]
    df = df[(df[date_col] >= pd.Timestamp(start)) & (df[date_col] < pd.Timestamp(end))]
    df = df.sort_values(date_col)

    # phrase support: 'kul_hakkı' / 'asgari ücret' -> consecutive tokens, each a prefix
    subs = [s for s in re.split(r"[_\s]+", tr_lower(word).strip()) if s]
    sub_rx = [re.compile(s) for s in subs]
    k = len(sub_rx)
    tok_rx = re.compile(r"\w+", re.UNICODE)
    rows = []
    for _, r in df.iterrows():
        toks = tok_rx.findall(str(r[text_col]))
        low = [tr_lower(t) for t in toks]
        for i in range(len(low) - k + 1):
            if all(sub_rx[j].match(low[i + j]) for j in range(k)):
                rows.append({"date": r[date_col].date(),
                             "venue": r.get("venue", ""),
                             "left": " ".join(toks[max(0, i - window):i]),
                             "keyword": " ".join(toks[i:i + k]),
                             "right": " ".join(toks[i + k:i + k + window])})
                if max_hits and len(rows) >= max_hits:
                    break
        if max_hits and len(rows) >= max_hits:
            break
    out = pd.DataFrame(rows, columns=["date", "venue", "left", "keyword", "right"])

    if verbose and len(out):
        lw = window * 7
        print(f"KWIC '{word}'  {start} → {end}  ({len(out)} hit"
              + (f", first {max_hits}" if max_hits and len(out) == max_hits else "") + ")")
        for _, r in out.iterrows():
            print(f"  {str(r['date'])}  {r['left'][-lw:]:>{lw}}  "
                  f"[ {r['keyword']} ]  {r['right'][:lw]}")
    return out


# ============================================================
# SCRIPT — `python 04_lexical_analysis.py` prints every lexical analysis.
#   The corpus is loaded once and passed to each call.
# ============================================================
if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    DISLOCATIONS = {
        "Coup 2016":       "2016-07-15",
        "Referendum 2017": "2017-04-16",
        "General 2018":    "2018-06-24",
        "Local 2019":      "2019-03-31",
        "General 2023":    "2023-05-14",
        "Ozel 2023":       "2023-08-11",
        "Local 2024":      "2024-03-31",
        "İmamoğlu 2025":   "2025-03-18",
    }
    _df = pd.read_csv(CORPUS_PATH)   # load once, pass to every call

    print("#" * 72)
    print("# DISLOCATION KEYNESS (Dunning LL) — per date: entering (+) / leaving (-)")
    print("#" * 72)
    for _name, _date in DISLOCATIONS.items():
        print(f"\n\n{'#' * 30} {_name} ({_date}) {'#' * 30}")
        dislocation_keyness(_date, corpus=_df)

    print("\n\n" + "#" * 72)
    print("# EGO KEYNESS — context shift of 'helalleşme' (LL collocate keyness)")
    print("#" * 72)
    ego_keyness("helalleş", event_date="2021-11-01", window_months=16, corpus=_df)


# ============================================================
# REGISTER ANALYSIS
# Tracks thematic term groups over time and around dislocations:
# which register an event displaces, and how long it takes to return.
#
# Patterns in default_stems.py were written for DISPLAY, not counting.
# Used raw they can be wrong: \blaz[^\W_]* matches "lazım" (99% of hits).
# Always run register_contribution() first and read the dominance flag.
# ============================================================

REGISTER_WINDOW_MONTHS = 3       # +- window around an event
REGISTER_BASELINE_MONTHS = 12    # pre-event baseline
REGISTER_MIN_TOKENS = 3000       # skip thinner months
REGISTER_RECOVERY_FRAC = 0.90    # "recovered" = back to this share of baseline
REGISTER_DOMINANCE_MAX = 0.30    # pool unreportable if one term exceeds this

TERM_GROUPS = {
    "economic_demands": [
        r"\baltın[^\W_]*", r"\bbütçe[^\W_]*", r"\bdolar[^\W_]*", r"\bdöviz[^\W_]*",
        r"\bekmek[^\W_]*", r"\bfiyat[^\W_]*", r"\bgeçim[^\W_]*", r"\bişsiz[^\W_]*",
        r"\bkonut[^\W_]*", r"\bkuruş[^\W_]*", r"\blira[^\W_]*", r"\bmaaş[^\W_]*",
        r"\bmaden[^\W_]*", r"\bpara(?!f)[^\W_]*", r"\bprim[^\W_]*", r"\bvergi[^\W_]*",
        r"\bzam(?!an)[^\W_]*",
        r"\bücret(?!siz)[^\W_]*",   # covers "asgari ücret"; `asgari` omitted to avoid double counting
    ],
    "labour_register": [
        r"\bemekli[^\W_]*", r"\bemekçi[^\W_]*", r"\bişçi[^\W_]*", r"\bmemur[^\W_]*",
        r"\bgrev[^\W_]*", r"\bsendika[^\W_]*", r"\btaşeron[^\W_]*", r"\bkıdem[^\W_]*",
    ],
    "charity_register": [           # Islamic-moral vocabulary of deservingness
        r"\bfakir[^\W_]*", r"\bfukara[^\W_]*", r"\byetim[^\W_]*", r"\bmazlum[^\W_]*",
    ],
    "communal_identities": [
        r"\bkürt(?!aj)[^\W_]*", r"\balevi[^\W_]*", r"\bsünni[^\W_]*",
        r"\bçerkez[^\W_]*", r"\bçerkes[^\W_]*",
        r"\blaz(?!ı|kiye)[^\W_]*",   # excludes lazım*, Lazkiye
        # `türk` omitted: flat across phases and 42% of the pool, masks the rest
    ],
    "ideological_identities": [
        r"\bmilliyetçi[^\W_]*", r"\bülkücü[^\W_]*", r"\bsolcu[^\W_]*", r"\bliberal[^\W_]*",
    ],
    "antagonist_personal": [
        r"\berdoğan[^\W_]*",
        r"\bakp\b", r"\bak\s+parti[a-zçğıöşü]*",     # \bakp\b blocks the surname Akpolat
        r"\bsaray(?!bosna|köy)[^\W_]*",              # moves opposite to erdoğan, report separately
    ],
    "antagonist_corruption": [
        r"\byolsuzluk[^\W_]*", r"\bçete(?!le)[^\W_]*", r"\bmafya[^\W_]*",
        r"\bbaron[^\W_]*", r"\buyuşturucu[^\W_]*",
        r"\brant[^\W_]*", r"\brüşvet[^\W_]*", r"\bisraf[^\W_]*",
    ],
    "antagonist_institutional": [
        # check with kwic() first: `mahkeme` also covers rights-claiming uses (ECtHR)
        r"\bmahkeme[^\W_]*", r"\bhükümet[^\W_]*", r"\byargı(?!la)[^\W_]*",
        # `iktidar` omitted: often self-referential ("iktidar olacağız")
    ],
    "antagonist_religious_order": [r"\bcemaat[^\W_]*", r"\bfetö[^\W_]*"],
    "antagonist_tutelage": [r"\bvesayet[^\W_]*"],
    "antagonist_terror": [r"\bterör[^\W_]*"],
    "coup_frame": [
        # kept out of the antagonist groups: referent inverts (2016 putschists,
        # 2025 the incumbent). Pooling would average the inversion away.
        r"\bdarbe[^\W_]*", r"\bcunta[^\W_]*", r"\bkayyım[^\W_]*", r"\bkayyum[^\W_]*",
    ],
}

DISLOCATIONS = {
    "Coup attempt 2016":     "2016-07-15",
    "Referendum 2017":       "2017-04-16",
    "General election 2018": "2018-06-24",   # crowding-out control: economic event
    "Local election 2019":   "2019-03-31",
    "General election 2023": "2023-05-14",
    "Local election 2024":   "2024-03-31",
    "İmamoğlu arrest 2025":  "2025-03-19",
}

_REG_TOK = re.compile(r"[a-zçğıöşü]+")


def _tr_lower(s):
    """Turkish-aware lowercase."""
    return str(s).replace("İ", "i").replace("I", "ı").lower()


def register_contribution(group, corpus=None, text_col="text_bert",
                          phase_col="phase", verbose=True):
    """Per-term rate (per 10,000 tokens) by phase, plus a dominance check.

    Run this before reporting any pooled series. If one term is more than
    REGISTER_DOMINANCE_MAX of the pool, the pool is really just that term.
    Returns (DataFrame, dominance_ok).
    """
    pats = TERM_GROUPS[group] if isinstance(group, str) else group
    if corpus is None:
        corpus = load_corpus()
    text = {p: _tr_lower(" ".join(g[text_col].astype(str)))
            for p, g in corpus.groupby(phase_col)}
    total = {p: len(_REG_TOK.findall(t)) for p, t in text.items()}
    phases = sorted(text)

    out = pd.DataFrame(
        {pat: [len(re.findall(pat, text[p])) / total[p] * 10000 for p in phases]
         for pat in pats},
        index=phases).T
    out.columns = [f"P{p}" for p in phases]

    last = out.columns[-1]
    share = out[last] / out[last].sum() if out[last].sum() else out[last]
    out["share_last"] = share
    dominance_ok = bool(share.max() <= REGISTER_DOMINANCE_MAX)

    if verbose:
        print(out.round(2).sort_values(last, ascending=False).to_string())
        print(f"\n  largest term = {share.max():.0%} of the pool")
        print("  -> pool reportable" if dominance_ok else
              "  -> DO NOT pool, report term by term")
    return out, dominance_ok


def register_recovery(group, events=None, corpus=None, text_col="text_bert",
                      date_col=DATE_COL, verbose=True):
    """How a register moves around each dislocation, and when it returns.

    Baseline = REGISTER_BASELINE_MONTHS before the event. Change = mean over the
    REGISTER_WINDOW_MONTHS after it. Recovery = months until the series regains
    REGISTER_RECOVERY_FRAC of baseline (searched 24 months out).

    Keep at least one event whose content matches the register (e.g. the Aug-2018
    currency crisis for economic demands): if the register rises there and falls
    elsewhere, the drops are not just agenda crowding-out.
    """
    pats = TERM_GROUPS[group] if isinstance(group, str) else group
    if events is None:
        events = DISLOCATIONS
    if corpus is None:
        corpus = load_corpus()
    df = corpus.dropna(subset=[date_col, text_col]).sort_values(date_col).copy()
    df["_m"] = df[date_col].dt.to_period("M")

    monthly = {}
    for m, g in df.groupby("_m"):
        text = _tr_lower(" ".join(g[text_col].astype(str)))
        n = len(_REG_TOK.findall(text))
        if n < REGISTER_MIN_TOKENS:
            continue
        monthly[m.to_timestamp()] = sum(len(re.findall(p, text)) for p in pats) / n * 10000
    s = pd.Series(monthly).sort_index()
    s = s.rolling(REGISTER_WINDOW_MONTHS, center=True, min_periods=2).mean()

    rows = []
    for label, date in events.items():
        d = pd.Timestamp(date)
        base = s[(s.index >= d - pd.DateOffset(months=REGISTER_BASELINE_MONTHS)) & (s.index < d)]
        post = s[(s.index >= d) & (s.index < d + pd.DateOffset(months=REGISTER_WINDOW_MONTHS))]
        if len(base) < 6 or len(post) < 2:
            rows.append((label, np.nan, np.nan, np.nan, "insufficient data"))
            continue
        b, p = base.mean(), post.mean()
        change = (p - b) / b * 100
        recovery = "no drop"
        if change < -15:
            fwd = s[(s.index >= d) & (s.index <= d + pd.DateOffset(months=24))]
            back = fwd[fwd >= REGISTER_RECOVERY_FRAC * b]
            recovery = (f"{round((back.index[0] - d).days / 30.44):.0f} months"
                        if len(back) else ">24 months")
        rows.append((label, b, p, change, recovery))

    out = pd.DataFrame(rows, columns=["event", "baseline", "post", "change_pct", "recovery"])
    if verbose:
        print(out.round(1).to_string(index=False))
    return out
