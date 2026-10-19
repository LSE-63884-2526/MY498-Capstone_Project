"""
PREPROCESSING PIPELINE
REPUBLICAN PEOPLE'S PARTY (TR) 2015-2026 CORPUS

Usage in notebook:
    from preprocessing_pipeline import PreprocessingPipeline

    pipeline = PreprocessingPipeline("corpus_main.csv")

    # -- Cleaning steps --
    pipeline.clean_parens()             # 1. remove transcriber parentheses
    pipeline.strip_spot_blocks()        # 2. delete ALL-CAPS editor spot blocks
    pipeline.strip_editorial_intro()    # 3. remove 3rd-person editor intros
    pipeline.strip_editor_attribution() # 4. strip attribution tags around quotes
    pipeline.clean_tv_speeches()        # 5. drop non-leader turns (tv/press)
    pipeline.remove_short_speeches()    # 6. remove speeches under 50 words

    # -- Text columns (all derived from the cleaned working text) --
    pipeline.build_text_clean()         # 7. text_clean (lowercase, no punct, stopwords kept)
    pipeline.build_text_bert()          # 8. text_bert  (cased, punctuation kept, for BERTurk)
    pipeline.build_text_lemma(JAR)      # 9. text_lemma (Zemberek - slow, needs JVM)

    # -- Single output CSV --
    pipeline.save()                     # -> corpus_main_processed.csv (the ONLY corpus file)

    # -- Inspection --
    pipeline.show_freq(col="text_lemma", n=200)
    pipeline.kwic("emekci", pipeline.df["text_bert"], dates=pipeline.df["month"])

Output columns (single CSV):
    id, date, title, url, venue, speaker, format, year, month, quarter,
    text_raw, text_clean, text_lemma, text_bert
"""

import re
import unicodedata
from collections import Counter
from pathlib import Path

import pandas as pd

from tqdm.auto import tqdm
tqdm.pandas()

from default_stems import DEFAULT_STEMS as DEFAULT_MANUAL_STEMS

ROOT           = Path(__file__).resolve().parent.parent
DATA_DIR       = ROOT / "Data"
STOPWORDS_PATH = ROOT / "stopwords_tr.txt"

# -----------------------------------------------------------------------------
# PIPELINE CLASS
# -----------------------------------------------------------------------------

class PreprocessingPipeline:

    # columns written to the single output CSV (in this order)
    FINAL_COLS = ["id", "date", "title", "url", "venue", "speaker", "format",
                  "year", "month", "quarter",
                  "text_raw", "text_clean", "text_lemma", "text_bert"]

    def __init__(
        self,
        input_csv: str,
        output_csv: str = DATA_DIR / "corpus_main_processed.csv",
    ):
        self.input_csv  = input_csv
        self.output_csv = output_csv

        self.df             = pd.read_csv(input_csv)
        self.df["text_raw"] = self.df["text"].copy()  # preserve original text

        with open(STOPWORDS_PATH, encoding="UTF-8") as f:
            manual_stopwords = {line.strip() for line in f if line.strip()}
        self.stopwords = manual_stopwords

        self.manual_stems   = DEFAULT_MANUAL_STEMS.copy()
        self.shorts_df      = pd.DataFrame()
        self._text_cleaned  = False

        print(f"Corpus loaded: {len(self.df)} rows")


    # -- INTERNAL HELPERS -----------------------------------------------------

    def _split_caps_boundary(self, text: str) -> str:
        """
        Insert a space where an ALL-CAPS heading block runs into the lowercase
        body: "LAZIMGencler" -> "LAZIM Gencler". With 2+ capitals followed by
        [capital + lowercase], the last capital goes to the NEW word.
        Abbreviation+suffix (apostrophised) is unaffected. A trapped quote mark
        is split off too ("YAŞANIR”Cumhuriyet"), so strip_spot_blocks can clean
        the block's final word and expose the intro behind it.
        Both the lemma and the bert build rely on this.
        """
        if not isinstance(text, str):
            return text
        return re.sub(r"([A-ZÇĞİÖŞÜ]{2,}[”\"’'‘“]*)([A-ZÇĞİÖŞÜ][a-zçğıöşü])", r"\1 \2", text)

    def _fix_punct_glue(self, text: str) -> str:
        """
        Split sentence-final punctuation glued to the next sentence:
        "soylesin.Bakin" -> "soylesin. Bakin". Requires at least 2 lowercase
        letters before the period, so abbreviations (M.I.T, A.B.D) are
        preserved — they have a single capital before the dot.
        Matters for cased BERT: two glued sentences formed one token block and
        corrupted the embedding. _split_caps_boundary handles capital-to-capital
        glue; this handles lowercase-dot-CAPITAL glue.
        """
        if not isinstance(text, str):
            return text
        return re.sub(r"([a-zçğıöşü]{2,}[.!?])([A-ZÇĞİÖŞÜ])", r"\1 \2", text)

    def _clean_text(self, text: str) -> str:
        """Remove URLs, social media handles, newlines, and punctuation."""
        text = self._split_caps_boundary(text)
        text = self._fix_punct_glue(text)      # must run last
        text = re.sub(r"http\S+|www\.\S+", " ", text)  # URLs
        text = re.sub(r"[@#]\w+", " ", text)            # @ mentions and hashtags
        text = re.sub(r"\n|\r|\t", " ", text)           # newlines and tabs
        text = re.sub(r"[^\w\s]", " ", text)            # punctuation
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"<[^>]+>", " ", text)                                  # html tags
        text = re.sub(r"[\(\[]\s*alkış[^\)\]]*[\)\]]", " ", text, flags=re.I) # (Alkışlar)
        text = re.sub(r"\balkışlar?\b|\btezahürat\b", " ", text, flags=re.I)  # bare stage direction
        return text.strip()

    def _turkish_lower(self, text: str) -> str:
        """Lowercase with Turkish-specific I/i dot handling."""
        text = text.replace("İ", "i").replace("I", "ı")
        return text.lower()

    def _normalize(self, text: str) -> str:
        return self._turkish_lower(self._clean_text(text))

    def _light_clean(self, text: str) -> str:
        """
        Minimal clean for BERT input. Keeps case, punctuation, stopwords,
        and morphological suffixes. Only strips URLs, handles, and fixes
        whitespace / unicode form. Do NOT lowercase if using cased BERTurk.

        ALL-CAPS news headlines (the leader's own words, printed in capitals)
        are first split off the glued body, then lowercased: the content
        survives but the casing signal that distorts cased BERT is removed.
        """
        if not isinstance(text, str):
            return ""
        text = unicodedata.normalize("NFC", text)
        text = self._split_caps_boundary(text)
        text = self._fix_punct_glue(text)      # must run last
        text = re.sub(r"http\S+|www\.\S+", " ", text)  # URLs
        text = re.sub(r"[@#]\w+", " ", text)            # @ mentions and hashtags
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def _manual_stem(self, text: str) -> str:
        for pattern, base in self.manual_stems.items():
            text = re.sub(pattern, base, text)
        return text


    # -- CLEANING STEPS (run before build_text_lemma) -------------------------
    def clean_parens(self):
        """
        Remove ALL parenthetical content (transcription/editor insertions).

        In speech transcripts parentheses are never the speaker's own; they are
        added by transcribers/editors to mark host questions, context, or asides:
          (Multecilerle ilgili soru uzerine)
          (Partinizin bu konuda onerileri var mi)
          (Cumhurbaskani adayi olur musunuz?)
          (Ahmet HAKAN- Neden?)
        All removed since none belong to the speaker's discourse.
        """
        def _clean(text):
            if not isinstance(text, str):
                return text
            text = re.sub(r"\([^)]*\)", " ", text)   # all parenthetical content
            text = re.sub(r"\s+", " ", text)
            return text.strip()

        self.df["text"] = self.df["text"].apply(_clean)
        print("Parenthesis cleaning complete (all parentheses removed).")

    def strip_editor_attribution(self,
                             attr_csv: str = DATA_DIR / "corpus_attributions_removed.csv"):
        """
        Removes editorial attribution TAGS while KEEPING the quoted speech.
        A reporting verb is required: "Kılıçdaroğlu'na oy vereceksiniz" (no
        reporting verb) is untouched; "Kılıçdaroğlu, sorularını yanıtladı:
        'söz'" loses the tag and keeps the speech. Quote marks are stripped too.
        """
        reporting_verb = (r"(şunları söyledi|şunları kaydetti|şunları ifade etti|"
                 r"şu (?:mesajları|ifadeleri|sözleri|cevabı|yanıtı) verdi|"
                 r"şöyle (?:devam etti|konuştu|dedi|yanıt verdi)|"
                 r"diye (?:konuştu|ekledi|belirtti|devam etti)|"
                 r"sorularını yanıtladı|soruları(?:nı)? (?:yanıtladı|cevapladı)|"
                 r"mesaj(?:ları)? verdi|ceva(?:bı|p) verdi|yanıtladı|"
                 r"dedi|belirtti|vurguladı|kaydetti|ifade etti|açıkladı|söyledi|konuştu|anlattı|seslendi)")
        name = (r"(özgür özel|kemal kılıçdaroğlu|kılıçdaroğlu|"
                r"(?-i:Özel)(?!\s*(?:sektör|likle|leştir|ekonomi|hayat|sohbet|le\b|de\b|durum|okul|güvenlik)))"
                r"(?!\s*[-–—])")
        title = r"(cumhuriyet halk partisi|chp)['\u2019]?\s*(genel başkanı|lideri|genel başkan)"

        pat1 = re.compile(
            title + r"[^.\u201C\u201D\"]*?" + name +
            r"[^.\u201C\u201D\"]*?" + reporting_verb + r"\s*:?\s*(?=[\u201C\u201D\"])",
            re.IGNORECASE)
        pat2 = re.compile(
            r"\b" + name + r"['\u2019]?\s*,?\s*[^.\u201C\u201D\"]*?" + reporting_verb +
            r"\s*:?\s*(?=[\u201C\u201D\"])",
            re.IGNORECASE)
        pat3 = re.compile(
            r"(?<=[\u201C\u201D\"])\s*,?\s*" + reporting_verb + r"(?:\s+" + name + r")?[^.\u201C\u201D\"]*",
            re.IGNORECASE)
        pat4 = re.compile(
            r"\b" + name + r"['\u2019]?\s*,?\s*[^.:\u201C\u201D\"]*?" + reporting_verb +
            r"[^.:\u201C\u201D\"]*[.:]",
            re.IGNORECASE)

        attr_records = []

        def _strip(idx, row):
            text = row["text"]
            if not isinstance(text, str):
                return text
            for pat in (pat1, pat2, pat3, pat4):
                for match in pat.finditer(text):
                    attr_records.append({
                        "speech_id": idx,
                        "id": row.get("id"),
                        "speaker": row.get("speaker"),
                        "date": row.get("date"),
                        "venue": row.get("venue"),
                        "removed_attribution": match.group(),
                    })
            text = pat1.sub(" ", text)
            text = pat2.sub(" ", text)
            text = pat3.sub(" ", text)
            text = pat4.sub(" ", text)
            text = re.sub(r"[\u201C\u201D\"]", " ", text)
            text = re.sub(r"\s+([.,;:!?])", r"\1", text)
            text = re.sub(r"\s+", " ", text)
            return text.strip()

        self.df["text"] = self.df.apply(lambda row: _strip(row.name, row), axis=1)

        attr_df = pd.DataFrame(attr_records)
        attr_df.to_csv(attr_csv, index=False)

        n_speeches = attr_df["speech_id"].nunique() if len(attr_df) else 0
        print(f"Editor attributions: {len(attr_df)} tags removed from {n_speeches} speeches")
        print(f"Saved: {attr_csv}")

    def strip_editorial_intro(self, max_iter: int = 6,
                          intro_csv: str = DATA_DIR / "corpus_intros_removed.csv"):
        """
        Iteratively removes editor-written third-person framing from the start of
        a speech. Five cutters, in priority order D->B->C->A->T:
          D  ALL-CAPS headline   : "CHP GENEL BAŞKANI ... BİR ARAYA GELDİ (date)"
          B  genitive (-nun)     : "... Kılıçdaroğlu'nun ... yaptığı konuşma şöyle:"
          C  title + colon       : "CHP Genel Başkanı Kemal Kılıçdaroğlu:"
          A  nominative + comma  : "(CHP) Genel Başkanı ... Kılıçdaroğlu, ... söyledi:"
          T  transition verb     : "... soruları yanıtladı / gündeme ilişkin ..."
        One cut per round; stops as soon as the text opens with a quote, which
        protects both the leader's speech and any leftover ALL-CAPS spot block.
        NOTE: spot blocks must be removed first, otherwise a leading block hides
        the intro behind it. Glued punctuation is opened first (_fix_punct_glue);
        A/T cuts stop at the EARLIEST quote / colon / sentence boundary and are
        capped at 450 characters, so they never swallow the speech.
        """
        LOW        = "a-zçğıöşü"
        QUOTE_OPEN = re.compile(r'^\s*["“‘]')
        NAME_QUOTE = re.compile(r'["“‘]([^"”’“‘]{0,60})["”’]')       # short quoted name/title
        def is_name_quote(text, i):
            """Is the opening quote at i a SHORT name/title rather than speech?
            <=5 words inside -> a programme or venue name ("Sümen 6"), not speech."""
            m = NAME_QUOTE.match(text[i:])
            return bool(m) and len(m.group(1).split()) <= 5
        def first_speech_quote(text):
            """Position of the first quote that actually opens speech; skips
            short-name quote PAIRS entirely. For straight quotes (") the opening
            and closing marks are identical, so we jump past the pair to avoid
            reading the closing mark as a new opening."""
            skip_to = 0
            for m in re.finditer(r'["“‘]', text):
                if m.start() < skip_to:
                    continue
                nm = NAME_QUOTE.match(text[m.start():])
                if nm and len(nm.group(1).split()) <= 5:
                    skip_to = m.start() + nm.end()      # skip the name's closing quote too
                    continue
                return m.start()                        # long or unclosed quote = speech
            return None
        D_START    = re.compile(r"^\s*(?:CHP\s+)?GENEL\s+BAŞKAN")     # ALL-CAPS news headline
        C_PAT = re.compile(
            r"^\s*(?:cumhuriyet halk partisi|chp)\s*(?:genel başkanı|genel başkan|lideri)\s+"
            r"(?:kemal\s+)?(?:kılıçdaroğlu|özgür\s+özel|özel)['’]?\s*:\s*[-–—]*\s*", re.I)
        B_PAT = re.compile(
            r"^\s*(?:.{0,40}?)?(?:cumhuriyet halk partisi|chp|genel başkan\w*)"
            r"[^\"“‘.]{0,150}?(?:kılıçdaroğlu|özel)['’]n[ıu]n\b[^\"“‘]{0,190}?"
            r"(?:yaptığı konuşma|konuşması|konuşma şöyle|açıklaması|değerlendirmeleri|"
            r"grup toplantısı konuşması|toplantısı konuşması|açıklama)\b"
            r"(?:\s+şöyle)?(?:\s*[;:]?\s*\(?\s*\d[\d .a-zçğıöşü]*\)?)?\s*[:]?\s*", re.I)
        # A_HEAD: the CHP/Cumhuriyet prefix is optional; the text may open directly
        # with "Genel Başkan(ı)/Lider" (e.g. "Genel Başkan Özel," exposed after spot removal).
        A_HEAD = re.compile(
            r"^\s*(?:(?:cumhuriyet halk partisi|chp)[^.]{0,30}?)?"
            r"(?:genel başkan\w*|lider\w*|cumhurbaşkanı aday\w*)[^.]{0,60}?"
            r"(?:kılıçdaroğlu|özel)\s*,", re.I)
        transitions = [
            r"şunları (söyledi|kaydetti)", r"şu açıklamayı yaptı",
            r"şöyle (konuştu|dedi|sürdürdü)", r"şu (ifadeleri|sözleri) (kullandı|söyledi)",
            r"toplantısında (konuştu|konuşma)", r"yaptığı konuşma şöyle", r"özetle şöyle",
            r"soruları yanıtlayan", r"yaptığı konuşma", r"grup toplantısı konuşması",
            r"ortak açıklaması", r"sorularını yanıt(?:ladı|layan)", r"soruları(?:nı)? yanıtladı",
            r"değerlendirmelerde bulundu", r"açıklamalarda bulundu",
            r"programına katıl(?:dı|arak)", r"yayınına katıl(?:dı|arak)",
            r"konuk oldu", r"konuğu oldu", r"canlı yayın konuğu oldu",
            r"katılarak soruları yanıtladı", r"gündeme ilişkin",
        ]
        TRANS_PAT = re.compile("|".join(transitions), re.I)

        def d_cut(s):                                   # leading ALL-CAPS block (no quote)
            if not D_START.match(s):
                return 0
            mq = re.search(r'["“‘]', s); qpos = mq.start() if mq else len(s)
            lpos = len(s)
            for tok in re.finditer(r"\S+", s):          # first lowercase token = end of the headline
                if re.search(rf"[{LOW}]", tok.group()):
                    lpos = tok.start(); break
            cut = min(qpos, lpos)
            return cut if cut >= 12 else 0

        def a_cut(s):                                   # nominative+comma: cut at the earliest boundary
            if not A_HEAD.match(s):
                return 0
            head = s[:500]; cands = []
            qpos = first_speech_quote(head)             # skip short-name quotes ("Sümen 6")
            if qpos is not None: cands.append(qpos)     # cut BEFORE the real speech quote (quote survives)
            mc = re.search(r":", head)
            if mc: cands.append(mc.end())               # cut AFTER the colon (speech survives)
            me = re.search(r"[.!?](?:\s|$)", head)
            if me: cands.append(me.end())               # cut AFTER the sentence end
            if not cands:
                return 0
            cut = min(cands)
            return cut if cut <= 450 else 0

        def t_cut(s):                                   # bounded transition verb (never crosses a quote)
            head = s[:500]
            mt = TRANS_PAT.search(head)
            if not mt:
                return 0
            end = mt.end()
            if end > 450:
                return 0
            qpos = first_speech_quote(head)             # only a real speech quote blocks the transition
            if qpos is not None and qpos < end:         # speech precedes the transition -> keep, do not cut
                return 0
            m2 = re.match(r"\s*:", s[end:])             # absorb a trailing ":"
            return end + (m2.end() if m2 else 0)

        intro_records = []

        def _strip(idx, row):
            text = row["text"]
            if not isinstance(text, str):
                return text
            text = self._fix_punct_glue(text)           # "yaptı.İki" -> "yaptı. İki"
            for _ in range(max_iter):
                s = text.lstrip(" .:\t-–—")              # leading orphan punctuation (incl. the "." left by spot removal)
                mopen = QUOTE_OPEN.match(s)
                if mopen and not is_name_quote(s, mopen.end() - 1):
                    break                                # leading real speech quote (not a short name) -> STOP
                cut, which = 0, None
                dc = d_cut(s)
                if dc: cut, which = dc, "D"
                if not cut:
                    m = B_PAT.match(s)
                    if m and m.end() > 12: cut, which = m.end(), "B"
                if not cut:
                    m = C_PAT.match(s)
                    if m: cut, which = m.end(), "C"
                if not cut:
                    ac = a_cut(s)
                    if ac: cut, which = ac, "A"
                if not cut:
                    tc = t_cut(s)
                    if tc: cut, which = tc, "T"
                if not cut:
                    break
                removed = s[:cut].strip()
                intro_records.append({
                    "speech_id": idx, "speaker": row.get("speaker"), "date": row.get("date"),
                    "venue": row.get("venue"), "removed_intro": removed,
                    "removed_chars": len(removed), "variant": which,
                })
                text = s[cut:]
            return text.strip()

        self.df["text"] = self.df.apply(lambda row: _strip(row.name, row), axis=1)
        intros_df = pd.DataFrame(intro_records)
        intros_df.to_csv(intro_csv, index=False)
        n_speeches = intros_df["speech_id"].nunique() if len(intros_df) else 0
        print(f"Editorial intro: {len(intros_df)} blocks removed from {n_speeches} speeches"
              + (f" | variant {intros_df['variant'].value_counts().to_dict()}" if len(intros_df) else ""))
        print(f"Saved: {intro_csv}")

    def strip_spot_blocks(self,
                          spot_csv: str = DATA_DIR / "corpus_spot_blocks_removed.csv"):
        """
        Removes every ALL-CAPS "spot block" (capitalised pull-quotes inserted by
        the editor) and saves them to a separate CSV.

        A spot block is a run of >=4 consecutive ALL-CAPS alphabetic tokens,
        counted only in set-off position (start of text / after . ! ? : ; /
        after a quote / opening with a quote). Short capitalised sequences inside
        a sentence (ASELSAN, HAVELSAN, TAI) and editorial D-headlines are
        therefore NOT removed. Each block is logged with speech_id / speaker /
        date / venue. Body-repetition matching is not computed here.
        """
        def caps_class(tok):
            core = re.sub(r"[^A-Za-zÇĞİÖŞÜçğıöşü]", "", tok.replace("’", "").replace("'", ""))
            if not core:
                return None                                  # neutral (digit/punctuation)
            return core == core.upper() and core != core.lower()

        def norm_words(s):
            s = s.replace("’", "").replace("'", "").replace("‘", "")
            s = re.sub(r"[^\w\s]", " ", self._turkish_lower(s))
            return [w for w in s.split() if len(w) > 2]

        D_HEADER = re.compile(r'^\s*(?:cumhuriyet halk partisi\s+)?(?:chp\s+)?genel başkan', re.I)
        BOUNDARY = set('.!?:;…”"\'')
        OPEN_Q   = ('“', '"', '‘')

        spot_records = []

        # When the spot block's first capitalised word is glued to the body by
        # sentence-final punctuation and/or a quote ("...sayıyorlar.“MESELE"), that
        # word merged into a lowercase token, the block did not look set-off, and it
        # was SKIPPED. Since quotes are stripped later (in strip_editor_attribution),
        # the block un-glued too late and leaked into text_bert. The de-glue below
        # runs before tokenisation and only inserts spaces — content and quotes are
        # unchanged. The pattern needs 2+ lowercase + [.!?:;…]/quote + 2+ capitals,
        # so abbreviations (A.B.D) and ordinary text are unaffected.
        LOW, CAPS, Q = "a-zçğıöşü", "A-ZÇĞİÖŞÜ", "\"“”‘’"
        def _deglue_caps_start(t):
            t = re.sub(rf'([{LOW}]{{2,}}[.!?:;…]+)([{Q}]*[{CAPS}]{{2,}})', r'\1 \2', t)  # end.“HEADING / end.TIP
            t = re.sub(rf'([{LOW}]{{2,}})([{Q}][{CAPS}]{{2,}})',          r'\1 \2', t)  # word“HEADING
            return t

        def _strip(idx, row):
            text = row["text"]
            if not isinstance(text, str):
                return text
            text = self._split_caps_boundary(text)   # "YAŞANIR”Cumhuriyet" -> split, so the whole block is removed
            text = _deglue_caps_start(text)          # "...sayiyorlar.“MESELE" -> "...sayiyorlar. “MESELE"
            toks = [(m.group(), m.start(), m.end()) for m in re.finditer(r"\S+", text)]
            n = len(toks); i = 0; spans = []
            while i < n:
                if caps_class(toks[i][0]) is True:
                    j = i; last = i; caps = 0
                    while j < n:
                        cj = caps_class(toks[j][0])
                        if cj is True:  caps += 1; last = j; j += 1
                        elif cj is None: j += 1                # digits/punctuation do not break the run
                        else: break                            # lowercase -> run ends
                    a, b = toks[i][1], toks[last][2]
                    run  = text[a:b].strip()
                    prev = text[:a].rstrip()
                    set_off = (not prev) or (prev[-1] in BOUNDARY) or run[:1] in OPEN_Q
                    if caps >= 4 and set_off and not D_HEADER.match(run) and len(norm_words(run)) >= 4:
                        spans.append((a, b, run))
                    i = last + 1
                else:
                    i += 1
            if not spans:
                return text
            for a, b, run in spans:
                spot_records.append({
                    "speech_id": idx, "id": row.get("id"), "speaker": row.get("speaker"),
                    "date": row.get("date"), "venue": row.get("venue"),
                    "spot_block": run, "n_words": len(run.split()),
                })
            for a, b, _ in sorted(spans, key=lambda x: -x[0]):
                text = text[:a] + " " + text[b:]
            return re.sub(r"\s+", " ", text).strip()

        self.df["text"] = self.df.apply(lambda row: _strip(row.name, row), axis=1)
        spot_df = pd.DataFrame(spot_records)
        spot_df.to_csv(spot_csv, index=False)
        n_sp = spot_df["speech_id"].nunique() if len(spot_df) else 0
        print(f"Spot blocks removed: {len(spot_df)} blocks from {n_sp} speeches")
        print(f"Saved: {spot_csv}")

    def clean_tv_speeches(self):
        """
        Keeps only the leader's (KK/Özel) turns in TV/interview transcripts.
        Journalist and other-speaker turns are dropped entirely; the leader's
        turn LABEL ("Kemal Kılıçdaroğlu -") is removed but the speech is kept.

        Turn boundaries are matched with a Turkish-aware pattern recognising all
        three dash types. Unrecognised "Name -" patterns (incidental uses like
        "Türkiye - Rusya") are not treated as turns, so the text is not damaged.
        """
        TV_HOST_NAMES = [
            "sunucu", "ismail küçükkaya", "hakan çelik", "fatih altaylı", "şirin payzın",
            "veyis ateş", "ahmet hakan", "taha akyol", "suat toktaş",
            "murat yetkin", "ahmed arpat", "orhan uğuroğlu", "hande fırat",
            "ece üner", "nazlı çelik", "tolga ışık", "seda öğretir",
            "oğuz haksever", "ufuk karcı", "deniz zeyrek",
            "mücahit enes turbil", "emre özgül", "doğan şentürk",
            "serap belet", "kürşad oğuz", "özlem gürses",
            "irfan değirmenci", "gökhan kayış", "ipek özbey",
            "hülya hökenek", "muharrem sarıkaya", "nagehan alçı", "simge fıstıkoğlu",
            "necmi uçar", "bengü şap babaeker", "ruşen çakır", "enver aysever", "ilker karagöz",
            "sacit ak", "uğur dündar", "esra toptaş", "fevzi çakır", "armağan çağlayan",
            "alican türkoğlu", "özay şendir", "jülide ateş", "ahmet arpad", "ali can türkoğlu",
            "ali bayramoğlu", "yaşar taşkın koç", "çağlar cilara", "özlem akarsu çelik",
            "hasan basri akdemir", "elif çakır", "murat sabuncu", "sorel dağıstanlı",
            "serhan asker", "mehmet tezkan", "savaş kerimoğlu", "kerem kırçuval",
            "merdan yanardağ", "cüneyt akman", "ahmet arpat", "nazlı çelik", "haldun altılar",
            "duygu demirdağ", "evren özalkuş", "zeynel lüle", "oğuzhan uğur", "mehmet akif ersoy",
            "alişer delek", "selen yalaz", "yavuz oğhan", "saygı öztürk",
            "babala", "candaş tolga ışık", "bülent aydemir", "gazeteci",
            "yavuz semerci", "ismail saymaz", "emre kongar", "mustafa yılmaz", "serpil çevikcan", "abdulkadir selvi",
            "sedat bozkurt", "fikret bila", "erdem gül", "kürşat oğuz", "mehmet ali kayacı", "mustafa deniz", "ibrahim kahveci",
            "binali yıldırım", "soru", "yavuz selim demirağ", "ali haydar fırat", "kadın", "çocuk", "öğrenci",
            "ev emekçisi kadın", "meral akşener", "ali babacan", "temel karamollaoğlu", "ahmet davutoğlu", "gültekin uysal",
            "ergün atalay", "okan konuralp", "nejla kurul",
            # press speakers + names missed earlier
            "ahmet arpat", "mustafa destici", "ekrem imamoğlu", "mansur yavaş", "erol bilecik", "güllü demir, oya berberoğlu, "
        ]

        def tr_lower(s):
            return s.replace("İ", "i").replace("I", "ı").lower()

        host_names = set(tr_lower(h) for h in TV_HOST_NAMES)

        tv_intro = re.compile(
            r"^.*?"
            r"(?:soruları(?:nı)? (?:yanıtla(?:dı|yan)|cevapla)|"
            r"değerlendirmelerde bulundu|programına katıldı|konuk oldu|"
            r"konuğu oldu|şunları (?:söyledi|dile getirdi)|çok sayıda konuya değindi|"
            r"gündeme ilişkin değerlendirmelerde bulundu)"
            r"[^.:]*[:.]\s*",
            re.IGNORECASE | re.DOTALL)

        # turn start: (optional CHP/title) + Name (1-3 words) + dash (any of three)
        turn_start = re.compile(
            r"(?:(?:cumhuriyet halk partisi|chp)['\u2019]?\s*(?:genel başkanı|lideri)?\s*)?"
            r"([A-ZÇĞİÖŞÜ][a-zçğıöşüA-ZÇĞİÖŞÜ]+(?:\s+[A-ZÇĞİÖŞÜ][a-zçğıöşüA-ZÇĞİÖŞÜ]+){0,2})\s*[-–—]\s*"
        )

        def extract_turns(text):
            if not isinstance(text, str):
                return text
            text = tv_intro.sub("", text, count=1)
            text = re.sub(r'^[\s.,;:!?\-–—]+', '', text)   # leading orphan punctuation
            # then split into turns
            parts = turn_start.split(text)
            result = [parts[0]]
            for i in range(1, len(parts) - 1, 2):
                speaker = tr_lower(parts[i].strip())
                speech = parts[i + 1]
                if "kılıçdaroğlu" in speaker or "özel" in speaker:
                    result.append(speech)                    # leader: keep the speech, drop the label
                elif any(host in speaker for host in host_names):
                    pass                                   # host/other speaker: drop the turn
                else:
                    result.append(parts[i] + " – " + speech)  # unrecognised: leave untouched
            return re.sub(r"\s+", " ", " ".join(result)).strip()

        # --- SAFE turn cleaner for press ---
        # Only RECOGNISED speaker labels are boundaries: "Name SURNAME-" (surname in
        # capitals) or the leader (Kılıçdaroğlu/Özel, any spelling). Non-leader turns
        # are dropped. Hyphenated phrases like "Türk-İş" or "AK PARTİ-" do not match,
        # so the text is not damaged. The pre-segment (a leader/neutral opening before
        # the first marker) is KEPT. Press transcripts differ structurally from TV, so
        # extract_turns is not used here: it swallowed natural transitions and
        # hyphenated phrases.
        PREC = re.compile(
            r"((?:[A-ZÇĞİÖŞÜ][a-zçğıöşü'’]+\s+){1,2}[A-ZÇĞİÖŞÜ]{3,}"
            r"(?:['’][A-ZÇĞİÖŞÜa-zçğıöşü]+)?)\s*[-–—]\s")
        LEAD = re.compile(r"((?:kemal\s+)?kılıçdaroğlu|(?:özgür\s+)?özel)\s*[-–—]\s", re.I)

        def remove_nonleader_turns(text):
            if not isinstance(text, str):
                return text
            marks = {}
            for m in PREC.finditer(text):
                marks[m.start()] = (m.end(), tr_lower(m.group(1)))
            for m in LEAD.finditer(text):
                marks.setdefault(m.start(), (m.end(), tr_lower(m.group(1))))
            ms = sorted((s, e, n) for s, (e, n) in marks.items())
            if not ms:
                return text                                  # no multi-speaker structure -> leave untouched
            out = [text[:ms[0][0]].strip()]                  # pre-segment KEPT (leader's opening)
            for i, (s, e, n) in enumerate(ms):
                end = ms[i + 1][0] if i + 1 < len(ms) else len(text)
                seg = text[e:end]
                if ("kılıçdaroğlu" in n) or ("özel" in n):
                    out.append(seg.strip())                  # leader turn: keep the speech, drop the label
                # otherwise a non-leader turn -> dropped
            return re.sub(r"\s+", " ", " ".join(x for x in out if x)).strip()

        for speaker_key in ["kilicdaroglu", "ozel"]:
            tv_mask = (self.df["venue"] == "tv") & (self.df["speaker"] == speaker_key)
            tv_before = self.df.loc[tv_mask, "text"]
            tv_after  = tv_before.apply(extract_turns)
            tv_changed = int((tv_before != tv_after).sum())     # count of speeches actually changed
            self.df.loc[tv_mask, "text"] = tv_after.values

            press_mask = (self.df["venue"] == "press") & (self.df["speaker"] == speaker_key)
            press_before = self.df.loc[press_mask, "text"]
            press_after  = press_before.apply(remove_nonleader_turns)
            press_changed = int((press_before != press_after).sum())
            self.df.loc[press_mask, "text"] = press_after.values

            print(f"Turn extraction: {speaker_key} - tv {tv_changed}/{tv_mask.sum()}, "
                  f"press {press_changed}/{press_mask.sum()} (changed/checked)")

    def remove_short_speeches(self, min_words: int = 50):
        """
        Remove speeches below min_words threshold from the main corpus.
        Short entries are saved to self.shorts_df for later inspection.
        """
        self.df["word_count"] = self.df["text"].str.split().str.len()
        short_mask = self.df["word_count"] < min_words

        short_records               = self.df[short_mask].copy()
        short_records["speech_id"]  = short_records.index
        short_records["intro_text"] = short_records["text"]
        short_records["reason"]     = "too_short"

        keep_cols        = ["speech_id", "speaker", "date", "venue", "intro_text", "reason"]
        self.shorts_df   = short_records[keep_cols].reset_index(drop=True)

        self.df = self.df[~short_mask].drop(columns=["word_count"]).reset_index(drop=True)
        print(f"Short speeches removed: {short_mask.sum()}")
        print(f"Remaining speeches: {len(self.df)}")

    def save(self):
        """Write the single consolidated corpus CSV (the only corpus output file).

        Selects FINAL_COLS (drops the working `text` and any leftover columns
        such as text_morpho / text_phrases / text_length / word_count)."""
        cols = [c for c in self.FINAL_COLS if c in self.df.columns]
        self.df[cols].to_csv(self.output_csv, index=False)
        self._text_cleaned = True
        print(f"Saved: {self.output_csv} ({len(self.df)} rows) cols={cols}")

    def load(self):
        """Reload the saved single CSV (skip re-running Zemberek)."""
        self.df            = pd.read_csv(self.output_csv)
        self._text_cleaned = True
        print(f"Loaded: {self.output_csv} ({len(self.df)} rows)")


    # -- PIPELINE STEPS -------------------------------------------------------

    def build_text_lemma(self, zemberek_jar_path):
        """
        Normalize -> Zemberek lemmatize -> apply MANUAL_STEMS.
        Replaces Stanza-based lemmatization with Zemberek for better Turkish coverage.
        """
        import jpype
        
        if not jpype.isJVMStarted():
            jpype.startJVM(classpath=[zemberek_jar_path])
        
        TurkishMorphology = jpype.JClass('zemberek.morphology.TurkishMorphology')
        morphology = TurkishMorphology.createWithDefaults()

        def _zemberek_lemmatize(text):
            if not isinstance(text, str) or not text.strip():
                return ""

            sentences = re.split(r'[.!?]+', text)

            lemmas = []
            for sentence in sentences:
                if not sentence.strip():
                    continue
                
                normalized = self._normalize(sentence)
                if not normalized.strip():
                    continue

                results = morphology.analyzeAndDisambiguate(normalized).bestAnalysis()
                for analysis in results:
                    lemma_list = list(analysis.getLemmas())
                    if lemma_list:
                        lemma = str(lemma_list[0])
                        # UNK and length only here — stopwords come later
                        if lemma != 'UNK' and len(lemma) > 2:
                            lemmas.append(lemma)

            # 1. apply MANUAL_STEMS
            text_with_stems = self._manual_stem(" ".join(lemmas))
            
            # 2. stopword filter (last)
            final_tokens = [
                t for t in text_with_stems.split() 
                if t not in self.stopwords
            ]
            
            return " ".join(final_tokens)
        
        print(f"Zemberek lemmatizes ({len(self.df)} speeches)...")
        
        from tqdm import tqdm
        tqdm.pandas()
        
        self.df["text_lemma"] = (
            self.df["text"]
            .dropna()
            .progress_apply(_zemberek_lemmatize)
        )
        print("text_lemma built.")

    def build_text_clean(self):
        """
        Normalize -> text_clean.

        Turkish-lowercased, URLs/handles/punctuation/applause markers removed,
        stopwords KEPT, NOT lemmatized. Distinct from text_bert (which keeps
        case + punctuation for BERTurk).
        """
        self.df["text_clean"] = self.df["text"].apply(
            lambda t: self._normalize(t) if isinstance(t, str) else "")
        print("text_clean built.")

    def build_text_bert(self):
        """
        Light clean -> text_bert, for BERTurk contextual embeddings.

        Keeps case, punctuation, stopwords, and suffixes so the model sees
        natural sentences. Derived from the `text` column AFTER the cleaning
        steps (clean_parens, strip_spot_blocks, strip_editorial_intro,
        strip_editor_attribution, clean_tv_speeches).
        """
        self.df["text_bert"] = self.df["text"].apply(self._light_clean)
        print("text_bert built.")


    # -- INSPECTION -----------------------------------------------------------

    def show_freq(self, col: str = "text_lemma", n: int = 100):
        """Print frequency list for a given column. Stopwords and short tokens excluded."""
        tokens = " ".join(self.df[col].dropna()).split()
        freq   = Counter(tokens).most_common(n)
        print(f"\n-- {col.upper()} FREQUENCY (top {n}) --")
        for word, count in freq:
            if len(word) > 3 and word not in self.stopwords:
                print(f"{word:40} {count}")

    def kwic(self, term: str, texts, dates=None, venue=None, window: int = 5,
         start_date=None, end_date=None, max_results=None, collocate=None) -> pd.DataFrame:
        """
        Key Word In Context (KWIC) concordance.

        Inspection tool - NOT part of the pipeline.
        Call after TF-IDF or conText to examine the immediate context of a term.
        Use raw corpus (text column) rather than text_lemma to preserve
        morphological suffixes and natural reading context.

        Reports overall frequency (occurrences and documents) across all texts.
        If a date range is supplied, also reports frequency within that range and
        restricts the concordance to it.

        Parameters
        ----------
        term        : search term (substring match)
        texts       : pd.Series of texts to search
        dates       : optional pd.Series of dates aligned with texts (e.g. df["month"])
        venue       : optional pd.Series of speech types aligned with texts
        window      : number of tokens on each side of the keyword
        start_date  : optional range start (e.g. "2019-04-01"); enables range frequency
        end_date    : optional range end
        max_results : optional cap on number of concordance rows returned

        Returns
        -------
        pd.DataFrame with columns: left, term, right, (date), (type)
        """
        texts = pd.Series(texts).reset_index(drop=True)
        if dates is not None:
            dates = pd.to_datetime(pd.Series(dates).reset_index(drop=True),
                                dayfirst=True, errors="coerce")
        if venue is not None:
            venue = pd.Series(venue).reset_index(drop=True)

        def count_hits(text_series):
            occ, docs = 0, 0
            for doc in text_series:
                if not isinstance(doc, str):
                    continue
                n = sum(1 for w in doc.split() if term in w)
                if n:
                    occ += n
                    docs += 1
            return occ, docs

        # overall frequency (all texts)
        total_occ, total_docs = count_hits(texts)

        # date range filter
        if start_date is not None and dates is not None:
            rng_mask = (dates >= pd.Timestamp(start_date))
            if end_date is not None:
                rng_mask &= (dates < pd.Timestamp(end_date))
            search_texts = texts[rng_mask]
            search_dates = dates[rng_mask]
            search_venue = venue[rng_mask] if venue is not None else None
            rng_occ, rng_docs = count_hits(search_texts)
        else:
            search_texts, search_dates, search_venue = texts, dates, venue
            rng_occ, rng_docs = total_occ, total_docs

        # summary
        print(f"{'='*60}")
        print(f"KWIC: '{term}'")
        print(f"Overall: {total_occ} occurrences in {total_docs} documents")
        if start_date is not None and dates is not None:
            print(f"Range ({start_date} - {end_date}): {rng_occ} occurrences in {rng_docs} documents")
        print(f"{'='*60}")

        # concordance (range only if range supplied)
        results = []
        for idx, doc in search_texts.items():
            if not isinstance(doc, str):
                continue
            words = doc.split()
            for i, word in enumerate(words):
                if term in word:
                    left_words  = words[max(0, i - window):i]
                    right_words = words[i + 1:i + window + 1]
                    if collocate is not None:
                        ctx_words = left_words + right_words
                        if not any(collocate in w for w in ctx_words):
                            continue   # skip if the collocate is absent from the window
                    row = {
                        "left":  " ".join(left_words),
                        "term":  word,
                        "right": " ".join(right_words),
                    }
                    if search_dates is not None:
                        row["date"] = search_dates.loc[idx]
                    if search_venue is not None:
                        row["venue"] = search_venue.loc[idx]
                    results.append(row)

        df = pd.DataFrame(results)
        if dates is not None and "date" in df.columns:
            df = df.sort_values("date").reset_index(drop=True)
        if max_results:
            df = df.head(max_results)
        return df