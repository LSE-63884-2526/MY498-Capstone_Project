"""
signifier_analysis.py
=====================
CHP discourse thesis: clustering, lift and contextual drift analysis of a single
signifier over its BERT context vectors.

Input (per term, under EMB_DIR):
    {term}_cw15vectors.npy    (N x 768, float32)
    {term}_cw15metadata.csv   (doc_id, date, year, month, venue, phase, surface, context)

The class loads a term ONCE; every method works on self.v and self.m.
Clustering happens in a single run, and both the UMAP plot and the lift table
come out of that same run.

Deliberately NOT in this file (separate concerns):
    - BERT embedding generation (torch/transformers, raw corpus) -> separate script
    - preprocessing                                              -> PreprocessingPipeline
    - KWIC over raw text                                         -> PreprocessingPipeline
"""

import os
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import umap
import hdbscan
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D
from scipy.stats import gaussian_kde
from numpy.linalg import norm
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- Output Adjustments ---
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["CMU Serif", "Computer Modern Roman", "DejaVu Serif"],
    "mathtext.fontset": "cm",
    "axes.unicode_minus": False,
    "pdf.fonttype": 42,   # embed fonts as vectors; prints cleanly in LaTeX
    "ps.fonttype": 42,
})

# ============================================================
# CONFIG (adjust paths for your environment)
# ============================================================
ROOT          = Path(__file__).resolve().parent.parent
EMB_DIR       = ROOT / "Embeddings"
PLOT_DIR      = ROOT / "plots"
STOP_PATH     = ROOT / "stopwords_tr.txt"
ZEMBEREK_JAR  = ROOT / "zemberek" / "zemberek-full.jar"
LATEX_FIG_DIR = None   # set to a path to mirror the heatmaps into a LaTeX figures dir

# ============================================================
# STEM MAP (merges inflections for display/summaries only)
# DEFAULT_STEMS: {regex: stem}. A word that fullmatches is reduced to that stem
# (hakkına/haklarını/hakkımız -> hak); otherwise it falls back to the first
# root_len characters. Does NOT affect the fit or the clustering — it only merges
# word summaries (_summarize, meta_ctfidf, print_group, group_members).
# ============================================================
try:
    from default_stems import DEFAULT_STEMS
    _STEM_COMPILED = [(re.compile(rx), st) for rx, st in DEFAULT_STEMS.items()]
except Exception:
    _STEM_COMPILED = []   # no default_stems: plain root_len behaviour

def to_stem(word, root_len=5, use_stems=True):
    """Reduce a word to its stem. With use_stems=True, a fullmatch in DEFAULT_STEMS
    wins; otherwise word[:root_len]. With use_stems=False, always word[:root_len]."""
    if use_stems and _STEM_COMPILED:
        for rx, st in _STEM_COMPILED:
            if rx.fullmatch(word):
                return st
    return word[:root_len]


# phase boundaries (derived from date when metadata has no phase column)
PHASE_BREAKS = [("2019-04-01", 1), ("2023-11-05", 2)]   # before 1, between 2, after 3

PHASE_COLOR = {"f1": "#228833", "f2": "#CCBB44", "f3": "#4477AA"}  # KK early / KK late / Özel
PHASE_LABEL = {"f1": "Phase 1 (KK Early)", "f2": "Phase 2 (KK Late)", "f3": "Phase 3 (Özel)"}
NEUTRAL     = "#cccccc"

CORPUS_PATH     = ROOT / "Data" / "corpus_main_processed.csv"
CORPUS_TEXT_COL = "text_bert"
CORPUS_ID_COL   = None


# ============================================================
# MODULE-LEVEL HELPERS
# ============================================================
def clean_context(text):
    """Drop apostrophe suffixes, Turkish-lowercase."""
    text = re.sub(r"['\u2018\u2019\u02bc]\w+", "", text)
    return text.replace("\u0130", "i").replace("I", "\u0131").lower()


def load_stopwords(path=STOP_PATH):
    with open(path, encoding="utf-8") as f:
        return [w.strip() for w in f if w.strip()]


def phase_of(ts):
    """Timestamp -> phase (int), when phase must be derived from date."""
    if pd.isna(ts):
        return 0
    for cut, ph in PHASE_BREAKS:
        if ts < pd.Timestamp(cut):
            return ph
    return PHASE_BREAKS[-1][1] + 1


# ---- zemberek POS: lazy singleton (the JVM starts once) ----
_MORPH = None


def _get_morphology(jar=ZEMBEREK_JAR):
    global _MORPH
    if _MORPH is None:
        import jpype
        if not jpype.isJVMStarted():
            jpype.startJVM(jpype.getDefaultJVMPath(), "-ea",
                           "-XX:+SuppressFatalErrorMessage", 
                           f"-Djava.class.path={jar}", convertStrings=False)
        TurkishMorphology = jpype.JClass("zemberek.morphology.TurkishMorphology")
        _MORPH = TurkishMorphology.createWithDefaults()
    return _MORPH


PROPER_KEEP = {"erdoğan", "imamoğlu", "kılıçdaroğlu", "özel", "bahçeli", "kurtuluş", "direniş", "kalkınma",
               "kuruluş", "yürüyüş"}
def is_content_word(word):
    if word.lower() in PROPER_KEEP:
        return True
    try:
        results = _get_morphology().analyze(word)
        analyses = results.getAnalysisResults()
        if analyses.isEmpty():
            return True
        pos = str(analyses.get(0).getDictionaryItem().primaryPos.shortForm)
        return pos in ("Noun", "Adj")
    except Exception:
        return True


def is_verb(word):
    """True if the word is a verb. Verbs are listed on their own row."""
    try:
        results = _get_morphology().analyze(word)
        analyses = results.getAnalysisResults()
        if analyses.isEmpty():
            return False
        pos = str(analyses.get(0).getDictionaryItem().primaryPos.shortForm)
        return pos == "Verb"
    except Exception:
        return False


# ============================================================
# MAIN CLASS
# ============================================================
class SignifierAnalyzer:
    """Analysis of a single signifier over its BERT context vectors.

    Usage:
        a = SignifierAnalyzer("millet", venue=None)
        a.fit(min_cluster_size=100, min_samples=15)     # clustering + lift
        a.plot_phase_heatmap()                           # phase-lift heatmap
        a.latex_group_table()                            # meta-cluster table
    """

    def __init__(self, term, venue=None, version="normal", emb_dir=EMB_DIR, stop_path=STOP_PATH,
                 use_stems=True):
        self.term = term
        self.emb_dir = emb_dir
        self.stop = load_stopwords(stop_path)
        self.use_stems = use_stems   # merge inflections via DEFAULT_STEMS (display/summary only)

        valid = {"normal", "maskwhite", "masket"}
        if version not in valid:
            raise ValueError(f"invalid version {version!r}; expected one of {valid}")
        self.version = version

        # --- load vectors and metadata according to the version ---
        if version == "maskwhite":
            tag = "_maskwhite"
        elif version == "masket":
            tag = "_masket"
        else:
            tag = ""
        self.v = np.load(f"{emb_dir}/{term}_cw15vectors{tag}.npy")
        self.m = pd.read_csv(f"{emb_dir}/{term}_cw15metadata{tag}.csv")

        if venue is not None:
            mask = self.m["venue"].str.contains(venue, case=False, na=False).values
            self.v, self.m = self.v[mask], self.m[mask].reset_index(drop=True)
        self.venue = venue

        # guarantee date + phase
        if "date" in self.m.columns:
            self.m["date"] = pd.to_datetime(self.m["date"], dayfirst=True, errors="coerce")
        if "phase" not in self.m.columns or self.m["phase"].isna().all():
            self.m["phase"] = self.m["date"].apply(phase_of)

        # clustering state
        self.clu = None          # 10D clustering space
        self._viz = None         # 2D visualisation space (lazy)
        self.out = None          # lift table
        self.cluster_groups = {}  # {"group name": [id, id, ...]}  meta-cluster

        self._load_corpus()       # load the raw corpus for KWIC (doc_id -> full text)

    def _load_corpus(self):
        """Load the raw corpus once and build a doc_id -> full text lookup for KWIC.

        If the corpus cannot be read (wrong path, missing file, broken id mapping)
        this is skipped silently; KWIC then falls back to the truncated context in
        the metadata, which still works.
        """
        if getattr(SignifierAnalyzer, "_corpus_cache", None) is None:
            try:
                raw = pd.read_csv(CORPUS_PATH)
                if CORPUS_ID_COL is None:
                    raw = raw.reset_index(drop=True)          # guarantee a 0-based index
                    lookup = raw[CORPUS_TEXT_COL].to_dict()    # key = 0,1,2,... = doc_id
                else:
                    lookup = dict(zip(raw[CORPUS_ID_COL], raw[CORPUS_TEXT_COL]))
                SignifierAnalyzer._corpus_cache = lookup   # class-level cache
            except Exception as e:
                print(f"WARNING: corpus not loaded ({e}); KWIC will fall back to truncated context.")
                SignifierAnalyzer._corpus_cache = {}
        self.corpus_lookup = SignifierAnalyzer._corpus_cache or None
        if not self.corpus_lookup:
            # empty dict -> drop the attribute so KWIC sees no corpus
            if hasattr(self, "corpus_lookup"):
                del self.corpus_lookup
        

    # ----------------------------------------------------------------
    # SENSITIVITY -> CLUSTERING -> LIFT -> WORDS   
    # ----------------------------------------------------------------
    def n_component_sensitivity(self, values=(5, 8, 10, 12, 15, 20),
                            min_cluster_size=None, min_samples=None,
                            venue=None, verbose=False, version="normal"):
        """Helps to choose n_component parameter for the fit function."""
        term = self.term
        if min_cluster_size is None:
            min_cluster_size = 50 if term == "türkiye" else 25
        if min_samples is None:
            min_samples = max(5, min_cluster_size // 3)
        rows = []
        for nc in values:
            a = SignifierAnalyzer(term, venue=venue, version=version)
            a.fit(n_components=nc, min_cluster_size=min_cluster_size,
                min_samples=min_samples, verbose=verbose)
            labels = a.m["cluster"]
            n_clust = labels.nunique() - (1 if -1 in labels.values else 0)
            n_noise = (labels == -1).sum()
            noise_pct = round(100 * n_noise / len(labels), 1)
            rows.append({"n_components": nc, "n_clusters": n_clust, "noise_%": noise_pct})
            print(f"  nc={nc:3}  clusters={n_clust:4}  noise={noise_pct}%")
        return pd.DataFrame(rows)

    def fit(self, min_cluster_size=None, min_samples=None, max_df_ratio=1,
            min_word_len=3, top_words=100, min_cluster_freq=5, pos_filter=True,
            ranking="both", root_len=5, verbose=True, dim_ratio=0.85, hide_common=False, n_components=10):
        """Cluster the term's context vectors and summarise each cluster.

        Reduces to n_components dimensions with UMAP, clusters with HDBSCAN, then
        hands off to _summarize for c-TF-IDF words, verbs and phase lift."""
        if min_cluster_size is None:
            min_cluster_size = 50 if self.term == "türkiye" else 25
        if min_samples is None:
            min_samples = max(5, min_cluster_size // 3) # set minimum value to 5
        if verbose:
            print(f"{self.term} | venue={self.venue or 'all'} | n={len(self.v)} | "
                  f"phase: {dict(self.m['phase'].value_counts().sort_index())}")
        if len(self.v) < min_cluster_size * 2:
            print("WARNING: little data, clustering may be unreliable")

        # --- 10D clustering space ---
        self.clu = umap.UMAP(n_neighbors=15, min_dist=0.0, n_components=n_components,
                             metric="cosine", random_state=42).fit_transform(self.v)
        self.m["cluster"] = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size,
                                            min_samples=min_samples).fit_predict(self.clu)
        n_clust = self.m.cluster.nunique() - (1 if -1 in self.m.cluster.values else 0)
        n_noise = (self.m.cluster == -1).sum()
        if verbose:
            print(f"number of clusters: {n_clust} | noise: {100 * n_noise // len(self.m)}%")

        return self._summarize(
            max_df_ratio=max_df_ratio, min_word_len=min_word_len, top_words=top_words,
            min_cluster_freq=min_cluster_freq, pos_filter=pos_filter, ranking=ranking,
            root_len=root_len, verbose=verbose, dim_ratio=dim_ratio, hide_common=hide_common)

    def _summarize(self, max_df_ratio=0.7, min_word_len=3, top_words=100,
                   min_cluster_freq=5, pos_filter=True, ranking="both",
                   root_len=5, verbose=True, dim_ratio=0.85, hide_common=False):
        """The part of fit that produces c-TF-IDF words, verbs, lift and self.out.

        Split out so split_cluster can call it after writing sub-clusters: the
        sub-clusters then appear in the same table, same universe and same
        frequency filters as ordinary clusters. Reads whatever self.m['cluster']
        currently holds."""
        # --- cluster words (c-TF-IDF) ---
        docs = (self.m[self.m.cluster != -1].groupby("cluster")["context"]
                .apply(lambda s: clean_context(" ".join(s))))
        cv = CountVectorizer(stop_words=list(self.stop) + [self.term],
                             token_pattern=rf"(?u)\b[a-z\u00e7\u011f\u0131\u00f6\u015f\u00fc]{{{min_word_len},}}\b",
                             lowercase=False)
        X = cv.fit_transform(docs)
        words = np.array(cv.get_feature_names_out())
        verb_words, verb_counts = None, None
        if pos_filter:
            content_mask = np.array([is_content_word(w) for w in words])
            verb_mask = np.array([is_verb(w) for w in words])
            verb_words = words[verb_mask]
            verb_counts = X[:, verb_mask].toarray()   # cluster x verb (docs = cluster texts)
            words, X = words[content_mask], X[:, content_mask]

        raw_counts = X.toarray()
        tf = raw_counts.astype(float)
        tf = tf / tf.sum(1, keepdims=True)
        idf = np.log((1 + tf.shape[0]) / (1 + (raw_counts > 0).sum(0)))
        ctfidf = tf * idf
        df_count = (raw_counts > 0).sum(0)
        keep = df_count <= max_df_ratio * tf.shape[0]
        words, ctfidf, raw_counts = words[keep], ctfidf[:, keep], raw_counts[:, keep]

        # for dimming: which stems are very common across clusters
        # stem-level spread: how many clusters each stem occurs in (inflections merged)
        root_cluster_presence = defaultdict(set)
        for ci in range(raw_counts.shape[0]):
            for i in np.where(raw_counts[ci] >= min_cluster_freq)[0]:
                root_cluster_presence[to_stem(words[i], root_len, self.use_stems)].add(ci)
        n_clusters = raw_counts.shape[0]
        root_dim_global = {r: len(cs) > dim_ratio * n_clusters
                   for r, cs in root_cluster_presence.items()}

        def top_by(score_matrix, ci, use_freq_filter=True, apply_hide=False):
            scores = score_matrix[ci].copy().astype(float)
            freqs = raw_counts[ci]
            if use_freq_filter:
                scores[freqs < min_cluster_freq] = 0
            root_freq, root_best, root_score = defaultdict(int), {}, defaultdict(float)
            for i in np.where(scores > 0)[0]:
                w, root = words[i], to_stem(words[i], root_len, self.use_stems)
                root_freq[root] += int(freqs[i])
                root_score[root] += scores[i]
                if root not in root_best or freqs[i] > root_best[root][1]:
                    root_best[root] = (w, freqs[i])
            roots = sorted(root_score, key=root_score.get, reverse=True)[:top_words]
            roots = sorted(roots, key=lambda r: root_freq[r], reverse=True)
            out_words = []
            for rank, r in enumerate(roots):
                if root_dim_global.get(r, False) and apply_hide:   # common stem, hide it
                    continue
                prefix = "·" if root_dim_global.get(r, False) else ""   # mark common stems with a dot
                if rank < 10:
                    label = f"{prefix}{root_best[r][0]}({root_freq[r]}; {root_best[r][1]})"
                else:
                    label = f"{prefix}{root_best[r][0]}({root_freq[r]})"
                out_words.append(label)
            return out_words
        # --- lift table ---
        phase_share = self.m["phase"].value_counts(normalize=True).to_dict()
        rows = []
        for ci, c in enumerate(docs.index):
            cl = self.m[self.m.cluster == c]
            lifts = {ph: (cl["phase"] == ph).mean() / phase_share.get(ph, 1e-9)
                     for ph in [1, 2, 3]}
            dom = max(lifts, key=lifts.get)
            row = {"cluster": c, "n": len(cl),
                   "lift1": lifts[1], "lift2": lifts[2], "lift3": lifts[3],
                   "dom_phase": dom, "dom_lift": lifts[dom]}
            if ranking in ("freq", "both"):
                row["freq_words"] = ", ".join(top_by(raw_counts, ci, use_freq_filter=False, apply_hide=False))
            if ranking in ("ctfidf", "both"):
                row["ctfidf_words"] = ", ".join(top_by(ctfidf, ci, apply_hide=hide_common))
            # most frequent verbs (own row): grouped by stem, commonest inflection + stem total
            if verb_counts is not None:
                vfreq, vbest = defaultdict(int), {}
                for i in np.where(verb_counts[ci] >= min_cluster_freq)[0]:
                    vr = to_stem(verb_words[i], root_len, self.use_stems)
                    vfreq[vr] += int(verb_counts[ci][i])
                    if vr not in vbest or verb_counts[ci][i] > vbest[vr][1]:
                        vbest[vr] = (verb_words[i], int(verb_counts[ci][i]))
                vtop = sorted(vfreq, key=vfreq.get, reverse=True)[:20]
                row["verb_words"] = ", ".join(f"{vbest[r][0]}({vfreq[r]})" for r in vtop)
            else:
                row["verb_words"] = ""
            rows.append(row)

        out = pd.DataFrame(rows).sort_values("dom_lift", ascending=False).reset_index(drop=True)

        if verbose:
            for _, r in out.iterrows():
                head = (f"c{r['cluster']:>2} n={r['n']:>4} "
                        f"f1={r['lift1']:.1f} f2={r['lift2']:.1f} f3={r['lift3']:.1f} "
                        f"[f{r['dom_phase']} {r['dom_lift']:.1f}]")
                if ranking == "both":
                    print(f"{head}\n   freq: {r['freq_words']}\n   ctfidf: {r['ctfidf_words']}"
                          f"\n   verbs: {r['verb_words']}")
                elif ranking == "freq":
                    print(f"{head} | {r['freq_words']}\n   verbs: {r['verb_words']}")
                else:
                    print(f"{head} | {r['ctfidf_words']}\n   verbs: {r['verb_words']}")

        out["dom_phase"] = out["dom_phase"].apply(lambda x: f"f{x}")
        self.out = out
        # for meta_words: word-level freq matrix + row->cluster mapping
        self._meta_raw = raw_counts          # cluster x word raw counts
        self._meta_words = words             # word array, aligned with raw_counts columns
        self._meta_rowmap = {int(c): ci for ci, c in enumerate(docs.index)}  # cluster id -> row
        return self

    def _ensure_fit(self):
        if self.out is None or "cluster" not in self.m.columns:
            raise RuntimeError("call .fit() first")


    # ----------------------------------------------------------------
    # UMAP PLOT
    # ----------------------------------------------------------------

    def set_cluster_names(self, mapping):
        """Map cluster_id -> a qualitative code name. Used together with group
        names (n=...). Repeated calls accumulate. Example:
            a.set_cluster_names({12: "Kemalist Populism", 4: "Popular Sovereignty"})
        """
        self.cluster_names = {**getattr(self, "cluster_names", {}), **mapping}
        return self

    def set_cluster_group(self, name, cluster_ids):
        """Group several clusters into one meta-cluster: one colour and one centred
        label in the plot, combined n/lift in the table. A single-element list is
        fine. Repeated calls accumulate.
            a.set_cluster_group("Populist Mobilisation", [19,29,30,31,32,33,34,35,36])
        """
        self.cluster_groups[name] = [int(c) for c in cluster_ids]
        return self

    def meta_lift(self, cluster_ids, verbose=True):
        """Combined lift of a cluster group, counting its members as one cluster.
        Does NOT average the member lifts; it pools the members and recomputes.
            a.meta_lift([19,29,30,31,32,33,34,35,36])
        Returns: {"n", "lift1", "lift2", "lift3"}
        """
        self._ensure_fit()
        ids = [int(c) for c in cluster_ids]
        mask = self.m["cluster"].isin(ids)
        total = int(mask.sum())
        phase_share = self.m["phase"].value_counts(normalize=True).to_dict()
        meta_ph = self.m.loc[mask, "phase"]
        lifts = {ph: (meta_ph == ph).mean() / phase_share.get(ph, 1e-9) for ph in [1, 2, 3]}
        if verbose:
            print(f"meta n={total} | f1={lifts[1]:.1f} f2={lifts[2]:.1f} f3={lifts[3]:.1f}")
        return {"n": total, "lift1": lifts[1], "lift2": lifts[2], "lift3": lifts[3]}
    
    def all_meta_lifts(self, verbose=False):
        """Combined n + phase lifts for every meta-cluster, as a DataFrame."""
        rows = []
        for name, ids in self.cluster_groups.items():
            r = self.meta_lift(ids, verbose=verbose)
            rows.append({"group": name, **r})
        return pd.DataFrame(rows).set_index("group")

    def plot_phase_heatmap(self, save_dir=None, show=True, fmt="pdf",
                           dpi=300, figsize=(9, 7), cmap="RdBu_r",
                           sort_by_centroid=True, label_fs=12, center=1.0,
                           text_thresh=0.35):
        """Phase-lift heatmap of the meta-clusters, from all_meta_lifts().
        sort_by_centroid=True: KK-Early-weighted groups on top, Özel at the bottom.
        cmap='RdBu_r' + center=1.0: lift<1 blue, lift>1 red, ~1 white — lift has a
        natural reference at 1.0, so a single diverging palette reads cleanly.
        fmt: 'pdf' (recommended for the thesis, vector) or 'png'.
        text_thresh: label turns white once |lift - center| exceeds this, else black.
            a.plot_phase_heatmap(save_dir="figures", show=False)
        """
        import seaborn as sns
        df = self.all_meta_lifts(verbose=False)
        m = df[["lift1", "lift2", "lift3"]].copy()
        m.columns = ["KK-Early", "KK-Late", "Özel"]

        if sort_by_centroid:
            phase_idx = np.array([1, 2, 3])
            centroid = (m.values * phase_idx).sum(axis=1) / m.values.sum(axis=1)
            m = m.iloc[np.argsort(centroid)]

        fig, ax = plt.subplots(figsize=figsize)
        sns.heatmap(m, annot=False, cmap=cmap, center=center,
                    linewidths=0.5, linecolor="white",
                    cbar_kws={"label": "phase lift"}, ax=ax)

        # draw the numbers manually: white past the distance threshold, black otherwise
        for yi in range(m.shape[0]):
            for xi in range(m.shape[1]):
                val = m.values[yi, xi]
                color = "white" if abs(val - center) >= text_thresh else "black"
                ax.text(xi + 0.5, yi + 0.5, f"{val:.1f}",
                        ha="center", va="center",
                        color=color, fontsize=label_fs)
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
        ax.tick_params(labelsize=label_fs)             # axis labels (cluster / phase)
        cbar = ax.collections[0].colorbar               # colour bar
        cbar.ax.tick_params(labelsize=label_fs - 2)
        cbar.ax.yaxis.label.set_size(label_fs - 1)
        ax.set_ylabel(""); ax.set_xlabel("")
        plt.tight_layout()

        if save_dir:
            targets = [save_dir]
            if LATEX_FIG_DIR and os.path.abspath(LATEX_FIG_DIR) != os.path.abspath(save_dir):
                targets.append(LATEX_FIG_DIR)
            for d in targets:
                os.makedirs(d, exist_ok=True)
                path = f"{d}/{self.term}_phase_heatmap.{fmt}"
                fig.savefig(path, dpi=dpi, bbox_inches="tight")
                print("saved:", path)
        if show:
            plt.show()
        else:
            plt.close(fig)
        return self
    
    def meta_words(self, cluster_ids, top=30, root_len=None, verbose=True):
        """Combined frequency list for a cluster group, summing over its members.
        Words are grouped by stem and frequencies are SUMMED across sub-clusters.
        Same logic as the single-cluster frequency display (commonest inflection +
        stem total), but over a group.
            a.meta_words([19,29,30,31,32,33,34,35,36], top=25)
        Returns: DataFrame [form, freq]; prints one line when verbose.
        """
        self._ensure_fit()
        if not hasattr(self, "_meta_raw"):
            raise RuntimeError("re-run fit for meta_words (the old fit output did not "
                               "store meta data)")
        rl = root_len if root_len is not None else getattr(self, "_root_len", 5)
        ids = [int(c) for c in cluster_ids]
        rows_idx = [self._meta_rowmap[c] for c in ids if c in self._meta_rowmap]
        if not rows_idx:
            print("no rows found for these ids:", ids)
            return pd.DataFrame(columns=["form", "freq"])

        # within-group total: each word's summed frequency over the selected clusters
        summed = self._meta_raw[rows_idx].sum(axis=0)   # combined freq per word

        # group by stem: stem total + that stem's commonest inflection
        root_freq, root_best = defaultdict(int), {}
        for i in np.where(summed > 0)[0]:
            w = self._meta_words[i]
            root = to_stem(w, rl, self.use_stems)
            f = int(summed[i])
            root_freq[root] += f
            if root not in root_best or f > root_best[root][1]:
                root_best[root] = (w, f)

        roots = sorted(root_freq, key=root_freq.get, reverse=True)[:top]
        out_rows = [(root_best[r][0], root_freq[r]) for r in roots]
        df = pd.DataFrame(out_rows, columns=["form", "freq"])
        if verbose:
            print(f"meta ({', '.join('c'+str(c) for c in ids)}) combined freq:")
            print("   " + ", ".join(f"{f}({n})" for f, n in out_rows))
        return df

    def meta_ctfidf(self, top=30, groups=None, verbose=True, reduce_frequent=False,
                    shared_min_share=0.8, shared_penalty=0.5, keep=None):
        """Words that distinguish each meta-cluster from the OTHER meta-clusters.
        The corpus is the set of meta-clusters; c-TF-IDF is recomputed over those
        documents.
        reduce_frequent:
          False    -> no suppression (raw c-TF-IDF).
          True     -> square-root of tf for EVERY word (suppresses all).
          'shared' -> multiplies only SHARED stems by shared_penalty. Only stems
                      present in most clusters (parti, cumhuriyet, devlet) are
                      suppressed; cluster-specific stems (lira, çiftçi, kul) are
                      untouched.
                      shared_min_share: share of clusters at which a stem counts
                      as shared.
                      shared_penalty: score multiplier for shared stems (0.5 = halved).
        keep: words exempt from the penalty even when shared (e.g. ['erdoğan']), to
              protect analytically central but widespread proper nouns.
        Returns: {group: DataFrame[form, ctfidf, freq]}"""
        self._ensure_fit()
        if not hasattr(self, "_meta_raw"):
            raise RuntimeError("re-run fit for meta_ctfidf")
        groups = groups or self.cluster_groups
        names = list(groups)
        rl = getattr(self, "_root_len", 5)

        # 1) combined raw frequency vector per meta-cluster (word level)
        M = np.zeros((len(names), self._meta_raw.shape[1]))
        for gi, name in enumerate(names):
            rows = [self._meta_rowmap[int(c)] for c in groups[name]
                    if int(c) in self._meta_rowmap]
            M[gi] = self._meta_raw[rows].sum(axis=0)

        # 2) c-TF-IDF: tf (within-row share) * idf (how many meta-clusters it occurs in)
        #    The +1 in idf keeps widespread stems (türkiye, lira) from dropping to
        #    zero: a stem keeps a high tf in the cluster where it dominates and a low
        #    c-TF-IDF in marginal ones. The most widespread markers (türkiye,
        #    cumhuriyet) are additionally collected by shared_roots into the
        #    'shared across clusters' row.
        tf = M / np.clip(M.sum(axis=1, keepdims=True), 1, None)
        if reduce_frequent is True:
            tf = np.sqrt(tf)          # square-root for every word
        df_count = (M > 0).sum(axis=0)
        idf = np.log((len(names) + 1) / (df_count + 1)) + 1
        ctfidf = tf * idf

        if reduce_frequent == "shared":
            # stem level: how many clusters each stem occurs in (inflections merged)
            root_cluster = defaultdict(set)
            for gi in range(M.shape[0]):
                for i in np.where(M[gi] > 0)[0]:
                    root_cluster[to_stem(self._meta_words[i], rl, self.use_stems)].add(gi)
            thresh = shared_min_share * len(names)
            shared_roots_set = {r for r, cs in root_cluster.items()
                                if len(cs) >= thresh}
            # reduce keep words to stems and exempt them from the penalty
            keep_roots = {to_stem(w, rl, self.use_stems) for w in (keep or [])}
            shared_roots_set -= keep_roots
            # penalise ONLY shared stems; leave the rest untouched
            shared_cols = np.array([to_stem(self._meta_words[i], rl, self.use_stems) in shared_roots_set
                                    for i in range(M.shape[1])])
            ctfidf[:, shared_cols] *= shared_penalty

        # 3) highest c-TF-IDF stems per group
        result = {}
        for gi, name in enumerate(names):
            root_score, root_best, root_freq = defaultdict(float), {}, defaultdict(int)
            for i in np.where(M[gi] > 0)[0]:
                w = self._meta_words[i]; root = to_stem(w, rl, self.use_stems)
                root_score[root] += ctfidf[gi, i]
                root_freq[root] += int(M[gi, i])
                if root not in root_best or M[gi, i] > root_best[root][1]:
                    root_best[root] = (w, int(M[gi, i]))
            roots = sorted(root_score, key=root_score.get, reverse=True)[:top]
            result[name] = pd.DataFrame(
                [(root_best[r][0], round(root_score[r], 3), root_freq[r]) for r in roots],
                columns=["form", "ctfidf", "freq"])
            if verbose:
                print(f"{name}: " + ", ".join(f"{f}" for f in result[name]['form'][:12]))
        return result

    def shared_roots(self, groups=None, min_share=0.9, top=10):
        """Stems present in most meta-clusters (the widespread markers that c-TF-IDF
        suppresses: türkiye, cumhuriyet, millet...). Feeds the 'shared across
        clusters' row under the table. min_share: share of groups at which a stem
        counts as shared. Returns: DataFrame[form, freq, n_groups].
            a.shared_roots(min_share=0.8)"""
        self._ensure_fit()
        if not hasattr(self, "_meta_raw"):
            raise RuntimeError("re-run fit for shared_roots")
        groups = groups or self.cluster_groups
        names = list(groups)
        rl = getattr(self, "_root_len", 5)

        # combined frequency per group (the same M as meta_ctfidf)
        M = np.zeros((len(names), self._meta_raw.shape[1]))
        for gi, name in enumerate(names):
            rows = [self._meta_rowmap[int(c)] for c in groups[name]
                    if int(c) in self._meta_rowmap]
            M[gi] = self._meta_raw[rows].sum(axis=0)

        # per stem: how many groups it occurs in + total freq + commonest inflection
        root_groups, root_freq, root_best = defaultdict(set), defaultdict(int), {}
        for gi in range(len(names)):
            for i in np.where(M[gi] > 0)[0]:
                w = self._meta_words[i]; root = to_stem(w, rl, self.use_stems)
                root_groups[root].add(gi)
                root_freq[root] += int(M[gi, i])
                if root not in root_best or M[gi, i] > root_best[root][1]:
                    root_best[root] = (w, int(M[gi, i]))

        thresh = min_share * len(names)
        shared = [(root_best[r][0], root_freq[r], len(gs))
                  for r, gs in root_groups.items() if len(gs) >= thresh]
        # exclude the signifier itself
        shared = [row for row in shared
                  if to_stem(row[0], rl, self.use_stems) != to_stem(self.term, rl, self.use_stems)]
        shared.sort(key=lambda x: x[1], reverse=True)
        return pd.DataFrame(shared[:top], columns=["form", "freq", "n_groups"])

    def group_members(self, name=None, words=None, top=30):
        """Without name: the full {group: [id...]} map. With name: that group's id list.
        words="freq"   -> (ids, meta_words DataFrame)
        words="ctfidf" -> (ids, that group's meta_ctfidf DataFrame)
        words="both"   -> (ids, freq_df, ctfidf_df)"""
        if name is None:
            return dict(self.cluster_groups)
        ids = self.cluster_groups[name]
        if words is None:
            return ids
        if words == "freq":
            return ids, self.meta_words(ids, top=top, verbose=False)
        if words == "ctfidf":
            return ids, self.meta_ctfidf(top=top, verbose=False)[name]
        if words == "both":
            return (ids,
                    self.meta_words(ids, top=top, verbose=False),
                    self.meta_ctfidf(top=top, verbose=False)[name])
        raise ValueError("words: None | 'freq' | 'ctfidf' | 'both'")

    def latex_group_table(self, top_words=20, reduce_frequent=False,
                          shared_penalty=0.5, keep=None):
        """LaTeX skeleton to be filled in by hand (Turkish terms kept).
        You add the translations, \\ant{} and \\midrule. Ids go in the appendix."""
        cti = self.meta_ctfidf(top=top_words, verbose=False, reduce_frequent=reduce_frequent,
                               shared_penalty=shared_penalty, keep=keep)
        order = []
        for name, ids in self.cluster_groups.items():
            r = self.meta_lift(ids, verbose=False)
            lifts = [r["lift1"], r["lift2"], r["lift3"]]
            dom = int(np.argmax(lifts))
            # phase centre of gravity: low = KK-Early, high = Özel
            phase_idx = np.array([1, 2, 3])
            centroid = (np.array(lifts) * phase_idx).sum() / sum(lifts)
            order.append((name, ids, lifts, dom, centroid))
        order.sort(key=lambda x: x[4])   # by centroid: KK-Early on top, Özel at the bottom

        header = (
            "\\footnotesize\n"
            "\\begin{longtable}{l r r r p{7cm}}\n"
            "\\caption{Contextual clusters of \\textit{hak}...} \\\\\n"
            "\\toprule\n"
            "\\textbf{Cluster (n)} & \\textbf{KK-E} & \\textbf{KK-L} & "
            "\\textbf{Özel} & \\textbf{Representative terms (freq.)} \\\\\n"
            "\\midrule\n"
            "\\endfirsthead\n\n"
            "\\multicolumn{5}{c}{\\tablename\\ \\thetable\\ \\textit{(continued)}} \\\\\n"
            "\\toprule\n"
            "\\textbf{Cluster (n)} & \\textbf{KK-E} & \\textbf{KK-L} & "
            "\\textbf{Özel} & \\textbf{Representative terms (freq.)} \\\\\n"
            "\\midrule\n"
            "\\endhead\n\n"
            "\\bottomrule\n"
            "\\endlastfoot\n"
        )

        lines = []
        for name, ids, lifts, dom, _ in order:
            n_total = int(self.out[self.out["cluster"].isin(ids)]["n"].sum())
            lift_cells = [f"\\textbf{{{lv:.1f}}}" if i == dom else f"{lv:.1f}"
                        for i, lv in enumerate(lifts)]
            tbl = cti[name].head(top_words)          # select and order by c-TF-IDF score
            terms = []
            for j, (_, row) in enumerate(tbl.iterrows()):
                form, freq = row["form"], int(row["freq"])
                terms.append(f"{form} ({freq})")
            lines.append(
                f"\\raggedright \\textbf{{{name}}} \\newline ({n_total})\n"
                f"  & {lift_cells[0]} & {lift_cells[1]} & {lift_cells[2]}\n"
                f"  & {', '.join(terms)} \\\\[8pt]"
            )

        body = "\n".join(lines)

        # --- shared across clusters row ---
        shared = self.shared_roots(min_share=0.8, top=15)
        if not shared.empty:
            shared_terms = ", ".join(f"{r['form']} ({int(r['freq'])})"
                                     for _, r in shared.iterrows())
            body += (f"\n\\midrule\n"
                     f"\\raggedright \\textit{{Shared across clusters}}\n"
                     f"  & & &\n"
                     f"  & {shared_terms} \\\\")

        footer = "\n\\end{longtable}"
        return header + "\n" + body + footer

    def latex_group_table_short(self, picked=None, top_words=6, signifier="hak",
                                reduce_frequent=False, shared_penalty=0.5, keep=None):
        """Short main-text version: only hand-picked clusters, fewer terms.
        'picked' = list of cluster_group names to include, in the order you want.
        If None, uses all groups. Appendix keeps the full longtable."""
        cti = self.meta_ctfidf(top=top_words, verbose=False, reduce_frequent=reduce_frequent,
                               shared_penalty=shared_penalty, keep=keep)

        names = picked if picked is not None else list(self.cluster_groups.keys())

        header = (
            "\\begin{table}[H]\n"
            "\\centering\n"
            "\\footnotesize\n"
            f"\\caption{{Selected contextual clusters of \\sig{{{signifier}}} "
            "and their phase-specific lift. "
            "Terms are selected and ordered by c-TF-IDF score; "
            "parenthetical counts are raw frequencies (inflections merged).}\n"
            f"\\label{{tab:clusters-{signifier}}}\n"
            "\\vspace{0.5em}\n"
            "\\begin{tabular}{p{2.5cm} r r r p{8cm}}\n"
            "\\toprule\n"
            "\\textbf{Cluster (n)} & \\textbf{KK-E} & \\textbf{KK-L} & "
            "\\textbf{Özel} & \\textbf{Representative terms (freq.)} \\\\\n"
            "\\midrule\n"
        )

        lines = []
        for name in names:
            ids = self.cluster_groups[name]
            r = self.meta_lift(ids, verbose=False)
            lifts = [r["lift1"], r["lift2"], r["lift3"]]
            dom = int(np.argmax(lifts))
            n_total = int(self.out[self.out["cluster"].isin(ids)]["n"].sum())
            lift_cells = [f"\\textbf{{{lv:.1f}}}" if i == dom else f"{lv:.1f}"
                        for i, lv in enumerate(lifts)]
            tbl = cti[name].head(top_words)          # select and order by c-TF-IDF score
            terms = []
            for j, (_, row) in enumerate(tbl.iterrows()):
                form, freq = row["form"], int(row["freq"])
                terms.append(f"{form} ({freq})")
            lines.append(
                f"\\raggedright \\textbf{{{name}}} ({n_total})\n"
                f"  & {lift_cells[0]} & {lift_cells[1]} & {lift_cells[2]}\n"
                f"  & {', '.join(terms)} \\\\\n\\addlinespace"
            )

        body = "\n".join(lines)

        # --- shared across clusters row: widespread markers suppressed by c-TF-IDF ---
        shared = self.shared_roots(min_share=0.8, top=15)
        if not shared.empty:
            shared_terms = ", ".join(f"{r['form']} ({int(r['freq'])})"
                                     for _, r in shared.iterrows())
            shared_row = (f"\n\\midrule\n"
                          f"\\raggedright \\textit{{Shared across clusters}}\n"
                          f"  & & &\n"
                          f"  & {shared_terms} \\\\")
        else:
            shared_row = ""

        footer = shared_row + "\n\\bottomrule\n\\end{tabular}\n\\end{table}"
        return header + body + footer

    def print_group(self, name, ranking="both"):
        """Print each sub-cluster of a meta-cluster one by one, in the same format
        as the fit output (n, phase lifts, freq/ctfidf words, verbs).
            a.print_group("Moral Node")
        """
        self._ensure_fit()
        ids = [int(c) for c in self.cluster_groups[name]]
        sub = self.out[self.out["cluster"].isin(ids)].copy()
        if sub.empty:
            print(f"no clusters found for '{name}' (is the fit current?)")
            return

        # group summary on top
        print(f"=== {name} | {len(sub)} clusters | total n={int(sub['n'].sum())} ===")
        for _, r in sub.iterrows():
            dph = str(r["dom_phase"]).lstrip("f")
            head = (f"c{r['cluster']:>2} n={r['n']:>4} "
                    f"f1={r['lift1']:.1f} f2={r['lift2']:.1f} f3={r['lift3']:.1f} "
                    f"[f{dph} {r['dom_lift']:.1f}]")
            if ranking == "both":
                print(f"{head}\n   freq: {r.get('freq_words','')}\n"
                    f"   ctfidf: {r.get('ctfidf_words','')}\n"
                    f"   verbs: {r.get('verb_words','')}")
            elif ranking == "freq":
                print(f"{head} | {r.get('freq_words','')}\n   verbs: {r.get('verb_words','')}")
            else:
                print(f"{head} | {r.get('ctfidf_words','')}\n   verbs: {r.get('verb_words','')}")

    def forms_in_cluster(self, cluster, keyword=None, root_len=5, top=30, verbose=True):
        """All inflections of one stem within a cluster (or meta-cluster), by
        frequency. Use it to decide which inflection to put in the table.

        cluster: a single cluster id (int) OR a list of ids (meta-cluster;
        inflections are SUMMED across sub-clusters).
        keyword: the stem (defaults to self.term). root_len = first N characters.
            forms_in_cluster(5, "demokrasi")            # single cluster
            forms_in_cluster([19,29,30], "demokrat")    # meta-cluster (summed)
        Returns: DataFrame [form, freq]. NOTE: only inflections that passed the
        fit's min_word_len and content-word filters are visible (same source as
        meta_words).
        """
        self._ensure_fit()
        if not hasattr(self, "_meta_raw"):
            raise RuntimeError("re-run fit for forms_in_cluster "
                               "(the old fit output did not store inflection data)")
        ids = [int(cluster)] if isinstance(cluster, (int, np.integer)) else [int(c) for c in cluster]
        rows_idx = [self._meta_rowmap[c] for c in ids if c in self._meta_rowmap]
        if not rows_idx:
            print("no rows found for these ids:", ids)
            return pd.DataFrame(columns=["form", "freq"])

        key = clean_context(keyword if keyword is not None else self.term)
        grp = key[:root_len]

        # total frequency of each word across the selected cluster(s)
        summed = self._meta_raw[rows_idx].sum(axis=0)   # combined freq per word
        # keep only inflections starting with this stem
        rows = []
        for i in np.where(summed > 0)[0]:
            w = self._meta_words[i]
            if w[:root_len] == grp:                      # same stem group
                rows.append((w, int(summed[i])))
        rows.sort(key=lambda x: x[1], reverse=True)
        rows = rows[:top]

        df = pd.DataFrame(rows, columns=["form", "freq"])
        if verbose:
            tag = f"c{ids[0]}" if len(ids) == 1 else f"meta({','.join('c'+str(c) for c in ids)})"
            if rows:
                print(f"{tag} '{grp}' inflections:  " + ",  ".join(f"{f}({n})" for f, n in rows))
            else:
                print(f"no '{grp}' stem in {tag}")
        return df

    def forms_across_clusters(self, form="hakkını", exact=True):
        """How often one surface form occurs in each cluster.
        exact=True : exact match (hakkını == hakkını)
        exact=False: substring (hakkını, hakkınızı, hakkınıza all count)
        """
        self._ensure_fit()
        f = clean_context(form)
        # surface may hold several forms for merged occurrences
        # ("hakları | hakları | hakkını"); each mention is checked separately.
        def _match(s):
            parts = [p.strip() for p in clean_context(str(s)).split("|")]
            if exact:
                return any(p == f for p in parts)
            return any(f in p for p in parts)
        hit = self.m["surface"].map(_match)

        rows = []
        for c in sorted(x for x in self.m.cluster.unique() if x != -1):
            mask = (self.m.cluster == c)
            n_tot = int(mask.sum())
            n_form = int((mask & hit).sum())
            rows.append({
                "cluster": c,
                "n_total": n_tot,
                f"n_{f}": n_form,
                "ratio": round(n_form / n_tot, 3) if n_tot else 0.0,
            })
        df = pd.DataFrame(rows).sort_values(f"n_{f}", ascending=False).reset_index(drop=True)
        total_form = int(hit.sum())
        print(f"'{f}' occurs {total_form} times across {(df[f'n_{f}']>0).sum()} clusters")
        return df


    # ----------------------------------------------------------------
    # SUB-CLUSTERING   (splitting a large syntactic cluster into discursive axes)
    # ----------------------------------------------------------------
    @staticmethod
    def fmt_id(cid):
        """Readable form of a sub-cluster id. Scheme: base = 1000 + parent*100.
        1503 -> '5;3', 1000 -> '0;0', 6012 -> '50;12'. Ordinary ids (< 1000) are
        unchanged. For reports and tables."""
        cid = int(cid)
        if cid >= 1000:
            b = cid - 1000
            return f"{b // 100};{b % 100}"
        return str(cid)

    def subcluster(self, cluster_id, n_components=12, min_cluster_size=15,
                   min_samples=5, show=None):
        """Split a cluster and READ it. Does NOT change self.m permanently:
        sub-clusters are written temporarily, printed via _summarize in the fit
        format (ctfidf + freq + lift + verbs, against the WHOLE corpus), then
        self.m is restored.
        The point is to judge which sub-clusters are meaningful by the same measure
        as the fit, then pass keep=[...] to split_cluster with the same parameters."""
        mask = (self.m.cluster == cluster_id).values
        sub_v = self.v[mask]
        idx = self.m.index[mask]
        print(f"c{cluster_id}: sub-clustering {mask.sum()} occurrences")
        if len(sub_v) < min_cluster_size * 2:
            print("too small, sub-clustering is meaningless")
            return None

        clu = umap.UMAP(n_neighbors=15, min_dist=0.0, n_components=n_components,
                        metric="cosine", random_state=42).fit_transform(sub_v)
        labels = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size,
                                 min_samples=min_samples).fit_predict(clu)
        n_sub = len(set(labels)) - (1 if -1 in labels else 0)
        noise = (labels == -1).mean()
        print(f"  -> {n_sub} sub-clusters, noise {noise:.0%}\n")

        # --- write temporarily: all sub-clusters under 1000+ ids ---
        base = 1000 + cluster_id * 100
        saved_out = self.out                      # save the current table
        saved_labels = self.m.loc[idx, "cluster"].copy()   # original labels
        for orig_i, lab in zip(idx, labels):
            self.m.at[orig_i, "cluster"] = (base + lab) if lab != -1 else -1

        # --- summarise in fit format and print ONLY the sub-clusters ---
        self._summarize(ranking="both", verbose=False)
        sub_ids = sorted(c for c in self.out.cluster.unique() if c >= base and c < base + 100)
        # largest first
        sub_rows = self.out[self.out.cluster.isin(sub_ids)].sort_values("n", ascending=False)
        order = sub_rows if show is None else sub_rows.head(show)
        for _, r in order.iterrows():
            raw = int(r["cluster"]) - base       # raw label for keep=[...]
            phase_counts = self.m[self.m.cluster == r["cluster"]]["phase"].value_counts().sort_index().to_dict()
            print(f"  sub-cluster {raw} (n={r['n']}, phase={phase_counts}) "
                  f"f1={r['lift1']:.1f} f2={r['lift2']:.1f} f3={r['lift3']:.1f} "
                  f"[{r['dom_phase']} {r['dom_lift']:.1f}]")
            if r.get("freq_words"):
                print(f"     freq: {r['freq_words']}")
            if r.get("ctfidf_words"):
                print(f"     ctfidf: {r['ctfidf_words']}")
            print(f"     verbs: {r['verb_words']}\n")

        # --- restore self.m and self.out ---
        self.m.loc[idx, "cluster"] = saved_labels
        self.out = saved_out
        return None

    def split_cluster(self, cluster_id, keep, n_components=12,
                      min_cluster_size=15, min_samples=5, resummarize=True):
        """Split a cluster and write ONLY the sub-clusters in keep back to the main
        cluster column under new ids. Everything not kept goes to noise (-1). Then
        _summarize refreshes every table (ctfidf/verbs/lift/self.out), so the
        sub-clusters appear in the same table and the same universe as ordinary
        clusters and can be fed to a meta-cluster like any other cluster.

        keep: sub-cluster numbers to keep. Call with the SAME parameters as
              subcluster, otherwise the numbers shift.
        Returns: the new id list, e.g. [503, 504, 509].

        Id scheme: base = 1000 + cluster_id*100 (c5 -> 1500, 1501...). Ordinary
        cluster ids are always < 1000 and sub-clusters always >= 1000, so collisions
        are impossible (cluster_id=0 included). fmt_id renders this as '5;3'.
        """
        base = 1000 + cluster_id * 100
        mask = (self.m.cluster == cluster_id).values
        idx = self.m.index[mask]
        sub_v = self.v[mask]
        if len(sub_v) < min_cluster_size * 2:
            print("too small, split cancelled")
            return []

        clu = umap.UMAP(n_neighbors=15, min_dist=0.0, n_components=n_components,
                        metric="cosine", random_state=42).fit_transform(sub_v)
        labels = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size,
                                 min_samples=min_samples).fit_predict(clu)

        keep = set(keep)
        for orig_i, lab in zip(idx, labels):
            self.m.at[orig_i, "cluster"] = (base + lab) if lab in keep else -1

        label_set = set(labels)
        kept = sorted(base + l for l in keep if l in label_set)
        kept_occ = sum(1 for lab in labels if lab in keep)
        dropped = int(mask.sum()) - kept_occ
        print(f"c{cluster_id} ({int(mask.sum())} occ) -> kept: "
              f"{[self.fmt_id(k) for k in kept]} ({kept_occ} occ), "
              f"moved to noise: {dropped} occ")
        if resummarize:
            self._summarize(ranking="both", verbose=False)
            # show the new sub-clusters in fit format
            new = self.out[self.out.cluster.isin(kept)]
            for _, r in new.iterrows():
                head = (f"c{self.fmt_id(r['cluster'])} n={r['n']:>4} "
                        f"f1={r['lift1']:.1f} f2={r['lift2']:.1f} f3={r['lift3']:.1f} "
                        f"[{r['dom_phase']} {r['dom_lift']:.1f}]")
                print(head)
                if r.get('freq_words'):
                    print(f"   freq: {r['freq_words']}")
                if r.get('ctfidf_words'):
                    print(f"   ctfidf: {r['ctfidf_words']}")
                print(f"   verbs: {r['verb_words']}")
        return kept

    # ----------------------------------------------------------------
    # IN-CLUSTER KWIC   (co-occurrence -> articulation check)
    # ----------------------------------------------------------------
    def kwic_in_cluster(self, cluster, keyword=None, context_words=None,
                    mode="root", root_len=3, phase=None, venue=None,
                    max_rows=None, newest_first=True):
        """KWIC over a cluster's occurrences, using the metadata context only.
        keyword: keep occurrences whose surface matches this word.
        context_words: keep occurrences whose context also contains these words (AND).
        venue: a single venue (str) or a list of venues.
        """
        self._ensure_fit()

        sub = self.m[self.m.cluster == cluster].copy()

        # phase filter
        if phase is not None:
            if isinstance(phase, str):
                phase = int(phase.lstrip("fF"))
            sub = sub[sub["phase"] == phase]

        # venue filter (single value or list)
        if venue is not None:
            venues = [venue] if isinstance(venue, str) else list(venue)
            sub = sub[sub["venue"].isin(venues)]

        # keyword filter (over surface)
        if keyword:
            kw = clean_context(keyword)
            def _kw_match(s):
                # surface may hold several forms (merged occurrences): match each
                # mention separately; the occurrence passes if any of them matches.
                parts = [p.strip(".,;:!?\"'()[] ") for p in clean_context(str(s)).split("|")]
                if mode == "exact":
                    return any(t == kw for t in parts)
                rl = min(root_len, len(kw))
                return any(t[:rl] == kw[:rl] for t in parts)
            sub = sub[sub["surface"].apply(_kw_match)]

        # context_words filter (AND)
        if context_words:
            cwords = [clean_context(w) for w in context_words]
            def _has_cwords(ctx):
                toks = [clean_context(t).strip(".,;:!?\"'()[]") for t in str(ctx).split()]
                for cw in cwords:
                    rl = min(root_len, len(cw)) if mode != "exact" else len(cw)
                    if not any((t == cw) if mode == "exact" else (t[:rl] == cw[:rl])
                            for t in toks):
                        return False
                return True
            sub = sub[sub["context"].apply(_has_cwords)]

        # order by date (used for sorting only; the date is not reported)
        if newest_first and "date" in sub.columns:
            sub = sub.sort_values("date", ascending=False)

        print(f"'{clean_context(keyword or self.term)}' (c{cluster}): {len(sub)} occurrences")

        out = sub[["phase", "venue", "context"]].rename(columns={"context": "match"})
        if max_rows:
            out = out.head(max_rows)
        return out
    # ----------------------------------------------------------------
    # CONTEXTUAL DRIFT
    # ----------------------------------------------------------------

    
    # ----------------- ARTICULATION CHANGE (JSD)  -----------------
    # APD/dispersion measures raw embedding GEOMETRY; this one measures
    # REDISTRIBUTION across articulations ((meta-)clusters) — which maps directly
    # onto the RQ, is robust to idiolect at sub-cluster level, bounded in [0,1],
    # decomposable, and drawn from the same data as the phase-lift heatmap (two
    # resolutions). It uses the object's own fit and cluster_groups, so no export,
    # no re-fit and no cluster-id mismatch.


    # Okabe-Ito colour-blind safe palette (for mover lines)


    def top_clusters_in_quarter(self, quarter, top=5, n_words=15, verbose=True):
        """The densest raw HDBSCAN clusters in one QUARTER (the same unit as APD).
        When the quarterly APD plot shows a peak or trough, pass that quarter here,
        see which clusters hold the most occurrences, then read them with
        kwic_in_cluster.

        quarter: a pandas Period string like '2024Q1', or any date falling in that
            quarter ('2024-02-15'); both are converted to the quarter.
        top: how many clusters. n_words: c-TF-IDF words per cluster (by score).
        Returns: DataFrame (cluster, n, share, ctfidf_words).
        """
        import re as _re
        per = pd.Period(quarter, freq="Q")
        d = self.m["date"]
        mask = (d.dt.to_period("Q") == per).values
        labs = self.m.loc[mask, "cluster"]
        labs = labs[labs != -1]                       # drop noise
        vc = labs.value_counts().head(top)
        total = int(mask.sum())
        out = pd.DataFrame({"cluster": vc.index.astype(int), "n": vc.values})
        out["share"] = (out["n"] / total).round(3)

        # c-TF-IDF words from self.out (already score-selected); strip the counts
        def _clean_words(s):
            if not isinstance(s, str):
                return ""
            # "word(345; 148)" / "word(345)" -> "word", first n_words
            ws = [_re.sub(r"\(.*?\)", "", w).strip().lstrip("·")
                for w in s.split(",")]
            ws = [w for w in ws if w]
            return ", ".join(ws[:n_words])
        wmap = {}
        if hasattr(self, "out") and self.out is not None:
            wmap = dict(zip(self.out["cluster"], self.out.get("ctfidf_words", "")))
        out["ctfidf_words"] = out["cluster"].map(lambda c: _clean_words(wmap.get(c, "")))

        if verbose:
            print(f"{per}  ({total} occurrences, noise excluded)")
            for _, r in out.iterrows():
                print(f"   c{int(r['cluster']):<4} n={int(r['n']):3d}  {r['share']:.1%}   {r['ctfidf_words']}")
        return out.reset_index(drop=True)
    
