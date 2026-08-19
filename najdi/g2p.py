#!/usr/bin/env python3
"""
Najdi Arabic G2P front-end for Kokoro / StyleTTS2
=================================================
Extends Nabra's MSA front-end (normalize -> espeak-ng ar -> clean) with the
dialect rewrites that make the model speak Najdi instead of fusha.

The MSA layer is kept byte-compatible with Nabra on purpose: same
`ARABIC_PHONEME_FIXUPS`, same `EXTRA_SYMBOLS` (ʕ->7, ħ->8), same syllable-dot
stripping. We are fine-tuning Nabra's embeddings, so every symbol we emit has
to land on the slot Nabra trained.

What this adds on top:

  1. ق -> /ɡ/     Najdi realizes ق as /g/. The corpus measures this directly:
                  13,648 of 18,030 tashkeel-transfer failures are exactly ق
                  heard as ج by an acoustic diacritizer. The transcripts keep
                  the letter ق, so without this rule the phonemes say /q/ while
                  the audio says /g/ — the aligner then has to absorb the
                  mismatch, and duration/F0 prediction pays for it.

  2. Exception lexicon: Qur'anic and high-register words keep /q/. Matched on
     the undiacritized skeleton so tashkeel variants collapse together.

  3. strip_final_harakat(): the corpus strips i'rab and pausal sukun (only 4.8%
     of marked words carry a word-final mark). Inference text must match that
     convention or it is off-distribution — the dataset card says so explicitly.

Deliberately NOT done:
  - ث/ذ are left as θ/ð. Najdi preserves the interdentals; there is no rule.
  - Emphatics stay collapsed (sˤ->s, ð̪ˤ->ð) via Nabra's fixups. Changing this
    would move symbols off the slots Nabra trained.
  - Kashkasha (ك -> /ts/) is not applied. It is bound to the 2nd-person
    feminine suffix; a blanket rule would corrupt every other ك. Candidate
    contexts are counted so it can be revisited with evidence.

Usage:
    from najdi.g2p import NajdiG2P, strip_final_harakat
    g2p = NajdiG2P()
    phonemes = g2p("قَالَ لِي أَحْمَد")      # -> 'ɡˈaːla liː ʔˈaħmad'
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from pathlib import Path

from najdi.normalize_tashkeel import normalize as normalize_marks

# ── MSA layer, carried over from Nabra's arabic_g2p.py verbatim ──────────────
ARABIC_PHONEME_FIXUPS = {
    "̪": "",  # combining bridge below (dental) -> strip
    "ˤ": "",  # pharyngealization marker -> strip
    "[": "",
    "]": "",
    "{": "",
    "}": "",
}

EXTRA_SYMBOLS = {
    "ʕ": 7,  # ع voiced pharyngeal fricative (distinct from ء/ʔ)
    "ħ": 8,  # ح voiceless pharyngeal fricative (distinct from ه/h)
}

_SYLLABLE_DOT = re.compile(r"(?<=\S)\.(?=\S)")
_LATIN_RUN = re.compile(r"[A-Za-z]+")
_CITATION = re.compile(r"\[[^\]]*\]")
_WS = re.compile(r"\s+")

# ── Arabic diacritics ────────────────────────────────────────────────────────
SHADDA = "ّ"
# Everything strippable at word end. Shadda is excluded: it marks gemination,
# not a case ending, and the corpus keeps it.
_FINAL_MARKS = {
    "ً",  # fathatan
    "ٌ",  # dammatan
    "ٍ",  # kasratan
    "َ",  # fatha
    "ُ",  # damma
    "ِ",  # kasra
    "ْ",  # sukun
    "ٰ",  # superscript alef
}
_ALL_HARAKAT = _FINAL_MARKS | {SHADDA}

_ALEF_VARIANTS = str.maketrans({"أ": "ا", "إ": "ا", "آ": "ا", "ٱ": "ا", "ة": "ه", "ى": "ي"})


def normalize_text(text: str) -> tuple[str, int]:
    """Drop citation markers + Latin-script loanwords and tidy whitespace."""
    text = _CITATION.sub(" ", text)
    # Tatweel/kashida (U+0640) is a decorative glyph-stretcher with no phonetic
    # value, but espeak does not recognise it bare and falls back to reading the
    # character BY NAME: "الـ" comes out as altatwˈiːl, i.e. the model literally
    # says the word "تطويل". Writers use it for trailing-off speech ("بكثر الـ..")
    # so it does reach us in real text. Strip it before phonemizing; a token
    # left dangling as a lone article is then simply dropped as empty.
    text = text.replace("ـ", "")
    latin = _LATIN_RUN.findall(text)
    if latin:
        text = _LATIN_RUN.sub(" ", text)
    text = _WS.sub(" ", text).strip()
    return text, len(latin)


def clean_phonemes(ph: str) -> str:
    """Strip espeak syllable-boundary dots and remap out-of-vocab phonemes."""
    ph = _SYLLABLE_DOT.sub("", ph)
    for old, new in ARABIC_PHONEME_FIXUPS.items():
        ph = ph.replace(old, new)
    return ph


def strip_final_harakat(text: str) -> str:
    """Remove word-final short vowels / tanwin / sukun, keeping shadda.

    Matches the corpus convention: i'rab and pausal sukun are stripped, shadda
    survives. Apply to inference text so it sits on the training distribution.

        رَجُلٌ  -> رَجُل
        حَقٌّ   -> حَقّ     (dammatan goes, shadda stays)

    Shadda and the vowel it carries appear in both orders in real text, so pop
    the whole trailing mark-run and filter it rather than stopping at the first
    shadda.
    """
    out = []
    for word in text.split(" "):
        i = len(word)
        while i > 0 and word[i - 1] in _ALL_HARAKAT:
            i -= 1
        out.append(word[:i] + "".join(c for c in word[i:] if c == SHADDA))
    return " ".join(out)


def undiacritize(word: str) -> str:
    """Letter skeleton, normalized — the key the exception lexicon matches on."""
    w = "".join(c for c in word if c not in _ALL_HARAKAT)
    w = w.translate(_ALEF_VARIANTS)
    return "".join(c for c in w if unicodedata.category(c) != "Mn")


# ── Words that keep /q/ despite the Najdi ق -> /ɡ/ default ───────────────────
# Qur'anic and high-register vocabulary, where speakers code-switch to the
# fusha realization. Seeded conservatively: a false entry here makes the model
# say /q/ in ordinary speech, which is worse than missing one. Grow it from
# evidence — section 3.5 of the plan recovers exactly these cases, since a word
# the acoustic diacritizer wrote with ق (not ج) is a word actually said /q/.
# Keys are undiacritized skeletons after alef/ta-marbuta normalization.
_Q_EXCEPTION_SEED = """
قران القران قرانا بالقران للقران والقران قرانيه
قيامه القيامه
قبله القبله
فقه الفقه فقهاء الفقهاء فقهي
تقوي التقوي
القدس
قرءان القرءان
مقرئ المقرئ
الفرقان المنافقون الانشقاق القارعه
"""

# Words that make the NEXT word a Qur'anic proper noun, which keeps /q/.
#
# Sura names cannot go in the flat lexicon because most double as ordinary
# words that Najdi genuinely says with /ɡ/: البقرة is "the cow", القصص is
# "stories", الطلاق is "divorce", القدر is "fate" or "a pot". A blanket entry
# would force /q/ in everyday speech — the exact false positive the seed list
# above warns about. The preceding word is what disambiguates them:
#     سورة البقرة  -> suːrat albaqarat   (the sura)
#     ذبحنا البقرة -> ðabaħnaː albaɡarat (the cow)
_Q_NEXT_TRIGGERS = {"سوره", "سور"}


def load_q_exceptions(path: str | Path | None = None) -> set[str]:
    """Undiacritized skeletons that keep /q/. Optional file extends the seed."""
    lex = {undiacritize(w) for w in _Q_EXCEPTION_SEED.split() if w}
    if path:
        p = Path(path)
        if p.exists():
            for line in p.read_text(encoding="utf-8").splitlines():
                line = line.split("#", 1)[0].strip()
                if line:
                    lex.add(undiacritize(line))
    return lex


class NajdiG2P:
    """Diacritized Najdi text -> Kokoro-compatible IPA.

    Set `dialect=False` to get Nabra's plain MSA behaviour, which is what the
    A/B comparison in the eval plan needs.
    """

    def __init__(
        self,
        dialect: bool = True,
        q_exceptions: str | Path | None = None,
        strip_final: bool = False,
    ):
        from misaki import espeak

        self._g2p = espeak.EspeakG2P(language="ar")
        self.dialect = dialect
        self.strip_final = strip_final
        self.q_exceptions = load_q_exceptions(q_exceptions)

        # Diagnostics — read these after a corpus pass, they are how you find
        # out the front-end is quietly misbehaving.
        self.latin_dropped = 0
        self.n_words = 0
        self.n_q_words = 0
        self.n_q_kept = 0
        self.n_misaligned = 0
        self.oov = Counter()
        self.kashkasha_candidates = Counter()

    # ── Najdi rewrites ──────────────────────────────────────────────────────
    def _rewrite_word(self, phonemes: str, word: str, prev: str | None = None) -> str:
        """Apply dialect rules to one word's phonemes.

        `prev` is the preceding word, needed for the sura-name rule: a word is
        only a Qur'anic proper noun when something like سورة introduces it.
        """
        if "ق" in word:
            self.n_q_words += 1
            if undiacritize(word) in self.q_exceptions:
                self.n_q_kept += 1
                return phonemes  # high-register: keep /q/
            if prev is not None and undiacritize(prev) in _Q_NEXT_TRIGGERS:
                self.n_q_kept += 1
                return phonemes  # sura name: سورة البقرة, not "the cow"
        # espeak only ever emits `q` for ق, so this is safe to apply globally
        # within the word once the exception check has passed.
        return phonemes.replace("q", "ɡ")

    def _note_kashkasha(self, word: str) -> None:
        """Count 2fs-suffix candidates so the ك->/ts/ rule can be revisited."""
        skel = undiacritize(word)
        if len(skel) > 2 and skel.endswith("ك"):
            self.kashkasha_candidates[skel] += 1

    # ── Main entry point ────────────────────────────────────────────────────
    def phonemize(self, text: str) -> str:
        text, n_latin = normalize_text(text)
        self.latin_dropped += n_latin
        if not text:
            return ""
        # harakat written BEFORE a shadda makes espeak drop the vowel silently
        # (الشَّمْس -> ʔaʃʃms instead of ʔaʃʃˈams). Only 0.7 % of the corpus is
        # affected, but a diacritizer feeding this at inference could emit the
        # wrong order everywhere, so canonicalize on the way in.
        text = normalize_marks(text)
        if self.strip_final:
            text = strip_final_harakat(text)

        raw, _ = self._g2p(text)
        phon = clean_phonemes(raw)

        if not self.dialect:
            self._record_oov(phon)
            return phon

        words = text.split(" ")
        parts = phon.split(" ")
        self.n_words += len(words)

        if len(parts) == len(words):
            for w in words:
                self._note_kashkasha(w)
            out = " ".join(
                self._rewrite_word(p, w, words[i - 1] if i else None)
                for i, (p, w) in enumerate(zip(parts, words))
            )
        else:
            # espeak merged or split something (proclitics, numerals). Fall back
            # to phonemizing each word alone so rules stay word-scoped. Slower,
            # and stress marks can differ slightly from the sentence rendering,
            # so it is worth watching this counter rather than ignoring it.
            self.n_misaligned += 1
            chunks = []
            for i, w in enumerate(words):
                self._note_kashkasha(w)
                wp = clean_phonemes(self._g2p(w)[0])
                chunks.append(self._rewrite_word(wp, w, words[i - 1] if i else None))
            out = " ".join(chunks)

        self._record_oov(out)
        return out

    __call__ = phonemize

    def _record_oov(self, phonemes: str) -> None:
        for ch in phonemes:
            if ch not in _VOCAB:
                self.oov[ch] += 1

    def stats(self) -> dict:
        return {
            "words": self.n_words,
            "q_words": self.n_q_words,
            "q_kept_as_q": self.n_q_kept,
            "q_rewritten": self.n_q_words - self.n_q_kept,
            "misaligned_sentences": self.n_misaligned,
            "latin_runs_dropped": self.latin_dropped,
            "oov_symbols": dict(self.oov),
        }


def _load_vocab() -> set[str]:
    """The 178 Kokoro symbols, as patched with Nabra's ʕ/ħ at 7/8."""
    import sys

    st2 = Path(__file__).resolve().parent.parent / "kikiri" / "StyleTTS2"
    sys.path.insert(0, str(st2))
    from kokoro_symbols import symbols  # noqa: E402

    return set(symbols)


_VOCAB = _load_vocab()


if __name__ == "__main__":
    g = NajdiG2P()
    samples = [
        "قَالَ لِي أَحْمَد",
        "الْقُرْآن الْكَرِيم",
        "طَبْعَا أَنَا مَا كَان عِنْدِي أَخْوَان",
        "النَّاس تَعَوَّدَت عَلَيْهِم يَعْنِي",
        "وِشْ رَايَك نَقْعُد فِي الْقَهْوَة",
    ]
    for s in samples:
        print(f"{s}\n  -> {g(s)}")
    print("\nstats:", g.stats())
    print("\nstrip_final_harakat:")
    for w in ["رَجُلٌ", "حَقٌّ", "بَيْتِهِ", "طَبْعَاً"]:
        print(f"  {w} -> {strip_final_harakat(w)}")
