#!/usr/bin/env python3
"""
Canonical ordering for Arabic combining marks.

Unicode allows a vowel mark either side of a shadda, and both orderings occur
in real text. espeak-ng only honours the canonical one:

    consonant + shadda + harakat   ->  الشَّمْس  ʔaʃʃˈams   (correct)
    consonant + harakat + shadda   ->  الشَّمْس  ʔaʃʃms     (vowel SILENTLY dropped)

The wrong order does not error, it just deletes a vowel from the phoneme
string. Applied across a corpus that is a systematic pronunciation defect that
no symbol-map assertion would catch, so normalize before phonemizing.
"""

import re

SHADDA = "ّ"
HARAKAT = "ً-ْٰ"

_SWAP = re.compile(f"([{HARAKAT}])({SHADDA})")
_DUP = re.compile(f"([{HARAKAT}])\\1+")


def normalize(text: str) -> str:
    """Reorder harakat+shadda -> shadda+harakat and collapse duplicated marks."""
    prev = None
    while prev != text:  # a mark may need to migrate past several neighbours
        prev = text
        text = _SWAP.sub(r"\2\1", text)
    return _DUP.sub(r"\1", text)


def count_misordered(text: str) -> int:
    return len(_SWAP.findall(text))


if __name__ == "__main__":
    import sys

    for path in sys.argv[1:]:
        with open(path, encoding="utf-8") as fh:
            src = fh.read()
        n = count_misordered(src)
        out = normalize(src)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(out)
        print(f"{path}: reordered {n} harakat+shadda pairs")
