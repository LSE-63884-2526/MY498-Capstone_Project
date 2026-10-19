"""
Contextual BERT embeddings for the main signifier terms.

Term matching is defined by MATCHERS at the top of this file — one regex per
target term, independent of default_stems. The inline comments record which
false positives each pattern excludes.

BLOCKERS lock a span but produce no output: they keep the "Halk" inside
"Cumhuriyet Halk Partisi" from counting as halk. Longest match wins, so the
long blocker claims the span first.

The corpus is never modified: the regex only labels spans, and the surface
column keeps the original form (milletin, hakkını, türkiyenin).

Each context window is written to its own pair of files:
    {term}_cw15vectors.npy / {term}_cw15metadata.csv
    {term}_cw20vectors.npy / {term}_cw20metadata.csv

NOTE: the embeddings in this repository were produced on Kaggle, on a CUDA GPU.
The script runs on CPU as well, but expect it to be considerably slower.
"""

import os, re
from pathlib import Path

import numpy as np
import pandas as pd
from collections import defaultdict

# =============================================================================
# BLOCKERS (lock a span, produce no output -> shield halk from CHP)
# =============================================================================
BLOCKERS = [
    r"cumhuriyet halk parti\w*",
    r"\bchp\b",
    r"adalet ve kalkınma parti\w*",   # full AKP name -> do not count as adalet
    r"\bak\s*parti\w*",               # AK Parti
]

# =============================================================================
# TARGET TERMS (single source of truth, defined here)
# Key = base label (file name + the term in metadata). Value = regex.
# Patterns run against tr_lower'd text (lowercase, Turkish i/I).
# =============================================================================
ET_RE = re.compile(r"^e[td]")
MATCHERS = {
    # millet (people/nation). Excludes milletvekili and milletlerarası.
    #"millet":     r"\bmillet(?!vekil|lerarası)\w*",

    # hak (right/justice). Excludes frequent false positives:
    #   hakkari (city), hakem (referee), hâkim (judge), hakkında/haklarında (about),
    #   hakikat (truth), hakaret (insult), haksız (unfair), hakan (Hakan Fidan).
    "hak":        r"\bhak(?!kari|ir|em|kâri|im|kında|kımda|kımızda|kınızda|larında|ikat|aret|sız|an|lı|kaniyet|[-\s]?iş)\w*",

    # halk (people). The CHP party name is shielded by BLOCKERS above.
    #"halk":       r"\bhalk\w*",

    # demokrasi. The demokra* stem covers demokrasi, demokrat and demokratik.
    #"demokrasi":  r"\bdemokra\w*",

    # türkiye (the country). türki* covers türkiyenin, türkiyeye... The 4-letter
    # "türk" (ethnic/adjectival) is not caught: the pattern requires 5 letters.
    #"türkiye":    r"\btürki\w*",

    # erdoğan. Multi-source: full name, tayyip, recep and tayyip bey all map here.
    #"erdoğan":    r"\b(?:recep\s+tayyip\s+erdoğan|tayyip\s+erdoğan|recep\s+tayyip|tayyip\s+bey|erdoğan|tayyip)\w*",
}

# ----------------- CONFIG -----------------
ROOT       = Path(__file__).resolve().parent.parent
BERT_CSV   = ROOT / "Data" / "corpus_main_processed.csv"
OUT_DIR    = ROOT / "Embeddings"
MODEL_NAME = "dbmdz/bert-base-turkish-uncased"

TEXT_COL   = "text_bert"
WINDOWS    = [15]     # one file per context window
BATCH_SIZE = 64
MAX_LEN    = 160
MIN_OCC    = 20
# -------------------------------------------------------------------------------
def whiten(X, eps=1e-6):
    mu = X.mean(axis=0, keepdims=True)
    Xc = X - mu
    cov = np.cov(Xc, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)
    W = eigvecs @ np.diag(1.0 / np.sqrt(eigvals + eps)) @ eigvecs.T
    return (Xc @ W).astype(np.float32)

def tr_lower(t):
    return (str(t).replace("İ","i").replace("I","ı").replace("Ğ","ğ")
            .replace("Ş","ş").replace("Ç","ç").replace("Ö","ö").replace("Ü","ü").lower())

def phase_of(ts):
    if pd.isna(ts): return 0
    if ts < pd.Timestamp("2019-04-01"): return 1   # KK early
    if ts < pd.Timestamp("2023-11-05"): return 2   # KK late
    return 3                                        # Özel

# ---- compile MATCHERS / BLOCKERS ----
def build_matcher():
    """base -> compiled regex, from MATCHERS."""
    return {base: re.compile(pat) for base, pat in MATCHERS.items()}

def build_blockers():
    """Compiled blocker patterns, from BLOCKERS."""
    return [re.compile(pat) for pat in BLOCKERS]

WORD_RE = re.compile(r"\S+")

def doc_words(text):
    """Scan the text once: words plus their (start, end) offsets."""
    words, starts, ends = [], [], []
    for m in WORD_RE.finditer(text):
        words.append(m.group()); starts.append(m.start()); ends.append(m.end())
    return words, starts, ends

def span_to_words(starts, ends, s, e):
    """Word indices overlapping the char range [s, e), via bisect."""
    import bisect
    lo = bisect.bisect_right(ends, s)
    out, i = [], lo
    while i < len(starts) and starts[i] < e:
        if ends[i] > s:
            out.append(i)
        i += 1
    return out

def find_occurrences(tl, n, matchers, blockers):
    """tl: tr_lower(text). Returns [(char_start, char_end, term), ...].
    Blockers lock a span but produce no output (base=None).
    Longest match wins and locks its span (occ), so matches never overlap:
    the long blocker ("cumhuriyet halk parti") locks before the short matcher
    ("halk") can claim it."""
    matches = []
    # blockers: base=None -> lock, emit nothing
    for p in blockers:
        for m in p.finditer(tl):
            matches.append((m.start(), m.end(), None))
    # matchers: base = term name
    for base, p in matchers.items():
        for m in p.finditer(tl):
            matches.append((m.start(), m.end(), base))
    matches.sort(key=lambda x: (-(x[1] - x[0]), x[0]))   # longest match first
    occ = bytearray(n)
    out = []
    for s, e, base in matches:
        if any(occ[s:e]):
            continue
        for k in range(s, e):
            occ[k] = 1
        if base is not None:                 # emit unless it is a blocker
            out.append((s, e, base))
    out.sort(key=lambda x: x[0])
    return out

# Occurrences of the same signifier within MERGE_WINDOW words merge into a single
# observation (anchored, so distant occurrences do not chain). Keeps one rhetorical
# passage ("millet, millet, millet") from counting as several rows.
MERGE_WINDOW = 15

def build_jobs(df, window, matchers, blockers, merge=True, W_merge=MERGE_WINDOW):
    jobs = []
    seen = {t: 0 for t in matchers}     # observations after merging
    raw  = {t: 0 for t in matchers}     # raw occurrences before merging (for logging)
    for _, row in df.iterrows():
        text = row.get(TEXT_COL)
        if not isinstance(text, str) or not text.strip():
            continue
        tl = tr_lower(text)
        words, starts, ends = doc_words(text)

        meta_base = {"doc_id": row.get("id"), "date": row.get("date"),
                     "year": row.get("year"), "month": row.get("month"),
                     "venue": row.get("venue"), "phase": row["_phase"]}

        # 1) collect this document's occurrences per term (word-index spans)
        occ_by_term = defaultdict(list)
        for s, e, term in find_occurrences(tl, len(tl), matchers, blockers):
            tgt = span_to_words(starts, ends, s, e)
            if tgt:
                occ_by_term[term].append(tgt)
                raw[term] += 1

        # 2) anchor-group per term, one job per group
        for term, occ_list in occ_by_term.items():
            occ_list.sort(key=lambda t: t[0])
            if merge:
                groups, cur, anchor = [], [], None
                for tgt in occ_list:
                    p = tgt[0]
                    if anchor is None or (p - anchor) < W_merge:
                        if anchor is None:
                            anchor = p
                        cur.append(tgt)
                    else:
                        groups.append(cur); cur = [tgt]; anchor = p
                if cur:
                    groups.append(cur)
            else:
                groups = [[tgt] for tgt in occ_list]     # merging off -> one job per occurrence

            for g in groups:
                all_idx = [i for tgt in g for i in tgt]
                lo = max(0, min(all_idx) - window)
                hi = min(len(words), max(all_idx) + window + 1)
                ctx = list(words[lo:hi])
                tgt_groups = []                          # local index list per mention
                for tgt in g:
                    loc = [i - lo for i in tgt]
                    for j in loc:
                        ctx[j] = ctx[j].strip('.!?,;:"“”‘’\'()[]')
                    tgt_groups.append(loc)
                surface = " | ".join(" ".join(ctx[j] for j in loc) for loc in tgt_groups)
                meta = dict(meta_base, surface=surface, context=" ".join(ctx),
                            n_merged=len(g))
                jobs.append((ctx, tgt_groups, term, meta))
                seen[term] += 1

    return jobs, seen, raw

# ----------------- EVERYTHING BELOW REQUIRES THE MODEL -----------------
def run(test_only=False, test_df=None, mask_target=False, mask_et=False):
    
    if mask_target:
        mode = "MASKED TARGET + WHITENED"
    elif mask_et:
        mode = "MASKED HAK ET (DESERVE) + WHITENED"
    else:
        mode = "NORMAL (unmasked, no whitening)"
    print(f"\n{'='*60}\n  MODE: {mode}\n{'='*60}")
    
    df = test_df if test_df is not None else pd.read_csv(BERT_CSV)
    df["_dt"] = pd.to_datetime(df["date"], dayfirst=True, errors="coerce")
    df["_phase"] = df["_dt"].apply(phase_of)

    matchers = build_matcher()
    blockers = build_blockers()

    if test_only:
        # no model: job list + counts for the first window only
        return build_jobs(df, WINDOWS[0], matchers, blockers)

    # --- load the model once, reuse it for every window ---
    import torch
    from transformers import AutoTokenizer, AutoModel
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME).to(device).eval()
    os.makedirs(OUT_DIR, exist_ok=True)

    from collections import defaultdict

    MASK_ID = tok.mask_token_id   # id of the [MASK] token

    
    @torch.no_grad()
    def flush(batch, vecs, metas, mask_target=False, mask_et=False):
        ctxs = [b[0] for b in batch]
        enc = tok(ctxs, is_split_into_words=True, padding=True,
                  truncation=True, max_length=MAX_LEN, return_tensors="pt").to(device)
    
        if mask_target or mask_et:
            for bi, (ctx, tgt_groups, term, meta) in enumerate(batch):
                wids = enc.word_ids(bi)
                target_word_ids = set(i for loc in tgt_groups for i in loc)
    
                # break up "hak et-": also mask the following word when it is the verb et-
                et_word_ids = set()
                if mask_et and term == "hak":
                    for wi in target_word_ids:
                        nxt = wi + 1
                        if nxt < len(ctx) and ET_RE.match(tr_lower(ctx[nxt])):
                            et_word_ids.add(nxt)
    
                for k, w in enumerate(wids):
                    if mask_target and w in target_word_ids:
                        enc["input_ids"][bi, k] = MASK_ID
                    if mask_et and w in et_word_ids:
                        enc["input_ids"][bi, k] = MASK_ID

        hid = model(**enc).last_hidden_state
        for bi, (ctx, tgt_groups, term, meta) in enumerate(batch):
            wids = enc.word_ids(bi)
            occ_vecs = []
            for loc in tgt_groups:
                pos = [k for k, w in enumerate(wids) if w in loc]
                if not pos:
                    continue
                occ_vecs.append(hid[bi, pos].mean(0))
            if not occ_vecs:
                continue
            v = torch.stack(occ_vecs).mean(0).cpu().numpy().astype(np.float32)
            vecs[term].append(v); metas[term].append(meta)

    # --- one run and one output file per window ---
    for window in WINDOWS:
        print(f"\n=== context window {window} ===")
        jobs, seen, raw = build_jobs(df, window, matchers, blockers)
        print(f"  Total {len(jobs)} merged observations "
              f"(from {sum(raw.values())} raw occurrences):")
        for term, n in seen.items():
            print(f"    {term:12} {raw[term]:6} -> {n:6}")

        vecs = defaultdict(list); metas = defaultdict(list)
        buf = []
        for job in jobs:
            buf.append(job)
            if len(buf) == BATCH_SIZE:
                flush(buf, vecs, metas, mask_target=mask_target, mask_et=mask_et); buf = []
        if buf:
            flush(buf, vecs, metas, mask_target=mask_target, mask_et=mask_et)

        if mask_target:
            tag = "_maskwhite"
        elif mask_et:
            tag = "_masket"
        else:
            tag = ""
        saved = 0

        for term in vecs:
            if len(vecs[term]) < MIN_OCC:
                continue
            do_whiten = mask_target or mask_et
            arr = np.vstack(vecs[term])
            if do_whiten:
                off_before = np.abs(np.cov(arr, rowvar=False) - np.diag(np.diag(np.cov(arr, rowvar=False)))).mean()
                arr = whiten(arr)
                off_after = np.abs(np.cov(arr, rowvar=False) - np.diag(np.diag(np.cov(arr, rowvar=False)))).mean()
                print(f"    {term}: whiten | off-diagonal {off_before:.4f} -> {off_after:.4f}")
            safe = term.replace("/", "_")
            np.save(os.path.join(OUT_DIR, f"{safe}_cw{window}vectors{tag}.npy"), arr)
            pd.DataFrame(metas[term]).to_csv(
                os.path.join(OUT_DIR, f"{safe}_cw{window}metadata{tag}.csv"), index=False)
            saved += 1

        zero = sorted(t for t in seen if seen[t] == 0)
        if zero:
            print(f"  WARNING cw{window} 0 occurrences:", zero)
        print(f"  cw{window}: {saved} terms")


    print(f"\nFinished. Documents -> {OUT_DIR}")

#run(mask_target=False)
#run(mask_target=True)
run(mask_et=True)