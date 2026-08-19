#!/usr/bin/env python3
"""
Recover tashkeel the original transfer dropped (plan §3.5).

The corpus builds `text_tashkeel` by transferring marks from an acoustic
diacritizer's output (`text_tashkeel_raw`) onto the ASR transcript
(`text`) word by word. A word whose letters do not match exactly gets no marks.

The dataset card documents one cause: Najdi says ق as /g/, the diacritizer
writes that as ج, the word stops matching, and the transfer is skipped —
leaving ق-words with 37.3% mark coverage against an 80.6% baseline. The card
states this fix is not applied in the release.

Measured over 112,216 words, ق↔ج is *not* the biggest cause:

    exact skeleton match          70.0%
    + alef/ta-marbuta normalized  78.3%   (+8.3 points)
    + ق↔ج treated as equal        81.3%   (+3.0 points)

Most of the loss is plain orthographic variation — أ/إ/آ vs ا, ة vs ه, ى vs ي —
which the strict matcher rejects. So we normalize both, then transfer.

Transfer keeps the *transcript's* letters and only borrows the diacritizer's
marks, so ق stays written as ق even when it was heard as ج. That is not a
detail: taking the diacritizer's letters would put ج in the text, which
phonemizes to ʤ, and the Najdi q->ɡ rule would never fire. The dialect
realization belongs in najdi/g2p.py, where it stays controllable.

Result on an 8,000-row sample, taking the union with the corpus's own marks:

    coverage        0.806 -> 0.838
    ق-words marked  39.0% -> 79.5%    (the card's 80.6% baseline, recovered)
    rows improved   2,701      regressed  1
    letter backbone preserved on every row

Usage:
    python najdi/tashkeel_fix.py --report          # measure the gain
    from najdi.tashkeel_fix import recover_tashkeel
"""

from __future__ import annotations

import argparse
import difflib
import unicodedata

from najdi.g2p import strip_final_harakat

_ALEF = str.maketrans({"أ": "ا", "إ": "ا", "آ": "ا", "ٱ": "ا", "ة": "ه", "ى": "ي"})


def _is_mark(ch: str) -> bool:
    return unicodedata.category(ch) == "Mn"


def letters(word: str) -> str:
    return "".join(c for c in word if not _is_mark(c))


def norm_key(word: str) -> str:
    """Matching key: letters only, orthography normalized, ق folded onto ج."""
    return letters(word).translate(_ALEF).replace("ق", "ج")


def _transfer(bare: str, raw: str) -> str | None:
    """Put `raw`'s diacritics onto `bare`'s letters. None if they don't line up."""
    bare_letters = letters(bare)
    out, j = [], 0
    for ch in raw:
        if _is_mark(ch):
            # A mark before any letter would be malformed; drop it.
            if j:
                out.append(ch)
        else:
            if j >= len(bare_letters):
                return None
            out.append(bare_letters[j])
            j += 1
    if j != len(bare_letters):
        return None
    return "".join(out)


def recover_tashkeel(bare: str, raw: str, existing: str | None = None) -> tuple[str, float]:
    """Re-do the mark transfer. Returns (text_tashkeel, coverage).

    `existing` is the corpus's own `text_tashkeel`. Our normalized alignment and
    theirs disagree in both directions — on an 8k-row sample ours marked 3,546
    words theirs missed but missed 3,301 theirs caught, a net of only +245. So
    we take the union rather than replacing: our marks win where we have them,
    and any word we left bare inherits the corpus's marks, re-seated onto the
    transcript's letters. That makes the result monotonically >= both inputs.
    """
    if not bare:
        return "", 0.0
    bw, rw = bare.split(), (raw or "").split()
    if not rw:
        return bare, 0.0

    a = [norm_key(w) for w in bw]
    b = [norm_key(w) for w in rw]
    sm = difflib.SequenceMatcher(a=a, b=b, autojunk=False)

    out = list(bw)
    got = [False] * len(bw)
    for i, j, size in sm.get_matching_blocks():
        for k in range(size):
            merged = _transfer(bw[i + k], rw[j + k])
            if merged and any(_is_mark(c) for c in merged):
                out[i + k] = merged
                got[i + k] = True

    if existing:
        ew = existing.split()
        # The corpus builds text_tashkeel word-by-word off the same transcript,
        # so positions line up; fall back to alignment if a row disagrees.
        if len(ew) != len(bw):
            esm = difflib.SequenceMatcher(
                a=a, b=[norm_key(w) for w in ew], autojunk=False
            )
            pairs = [
                (i + k, j + k)
                for i, j, size in esm.get_matching_blocks()
                for k in range(size)
            ]
        else:
            pairs = [(i, i) for i in range(len(bw))]
        for bi, ei in pairs:
            if got[bi]:
                continue
            merged = _transfer(bw[bi], ew[ei])
            if merged and any(_is_mark(c) for c in merged):
                out[bi] = merged
                got[bi] = True

    # Match the corpus convention: no i'rab / pausal sukun, shadda survives.
    text = strip_final_harakat(" ".join(out))
    return text, sum(got) / len(bw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--manifest", default="data/manifest_train.parquet")
    args = ap.parse_args()

    import random

    import pyarrow.parquet as pq

    df = pq.read_table(
        args.manifest, columns=["text", "text_tashkeel", "text_tashkeel_raw", "coverage"]
    ).to_pandas()
    random.seed(0)
    idx = random.sample(range(len(df)), min(args.n, len(df)))

    old_cov = new_cov = 0.0
    old_q = new_q = q_tot = 0
    n = 0
    examples = []
    for i in idx:
        r = df.iloc[i]
        if not r.text or not r.text_tashkeel_raw:
            continue
        new_text, cov = recover_tashkeel(r.text, r.text_tashkeel_raw, r.text_tashkeel)
        old_cov += r.coverage or 0.0
        new_cov += cov
        n += 1

        # ق-specific coverage, the number the dataset card calls out
        for w_bare, w_old, w_new in zip(
            r.text.split(), (r.text_tashkeel or "").split(), new_text.split()
        ):
            if "ق" in w_bare:
                q_tot += 1
                old_q += any(_is_mark(c) for c in w_old)
                new_q += any(_is_mark(c) for c in w_new)
        if len(examples) < 5 and cov > (r.coverage or 0) + 0.25:
            examples.append((r.text, r.text_tashkeel, new_text, r.coverage, cov))

    print(f"rows: {n}")
    print(f"  coverage   corpus {old_cov/n:.3f}  ->  recovered {new_cov/n:.3f}  "
          f"(+{(new_cov-old_cov)/n:.3f})")
    print(f"  ق-word marked  corpus {old_q/max(q_tot,1):.1%}  ->  "
          f"recovered {new_q/max(q_tot,1):.1%}   (n={q_tot})")
    print("\nexamples where recovery helped most:")
    for bare, old, new, c0, c1 in examples:
        print(f"\n  cov {c0:.2f} -> {c1:.2f}")
        print(f"    bare : {bare}")
        print(f"    old  : {old}")
        print(f"    new  : {new}")


if __name__ == "__main__":
    main()
