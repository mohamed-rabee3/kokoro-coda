#!/usr/bin/env python3
"""
G2P gate — verification checkpoint #2.

Runs the Najdi front-end over real corpus rows and asserts every emitted symbol
lands in the Kokoro vocab. A symbol that is not in the table gets silently
dropped by TextCleaner, which is how a broken front-end turns into a model that
trains fine and speaks nonsense. Fail loudly here instead.

Also reports whether the corpus text already follows the pausal convention, so
we know `strip_final_harakat` is reproducing it rather than fighting it.

Usage:
    python najdi/g2p_gate.py [--n 500]
"""

import argparse
import glob
import sys
from collections import Counter

import pyarrow.parquet as pq

from najdi.g2p import NajdiG2P, strip_final_harakat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--parquet", default=None)
    args = ap.parse_args()

    path = args.parquet or glob.glob(
        "/home/ai2/.cache/huggingface/hub/datasets--Eman-Fouda*/snapshots/*/data/val-*.parquet"
    )[0]
    d = pq.read_table(path, columns=["text", "text_tashkeel", "coverage"]).to_pydict()
    n = min(args.n, len(d["text_tashkeel"]))
    print(f"{path}\nrows: {n}\n")

    # Does the corpus already strip word-final marks?
    unchanged = sum(
        1
        for t in d["text_tashkeel"][:n]
        if t and strip_final_harakat(t) == t
    )
    print(f"pausal convention: {unchanged}/{n} rows unchanged by strip_final_harakat "
          f"({unchanged/n:.1%}) — expect high; confirms we match the corpus\n")

    g_dialect = NajdiG2P(dialect=True)
    g_msa = NajdiG2P(dialect=False)

    empty = 0
    for t in d["text_tashkeel"][:n]:
        if not t:
            empty += 1
            continue
        g_dialect.phonemize(t)
        g_msa.phonemize(t)

    s = g_dialect.stats()
    print("dialect front-end")
    for k, v in s.items():
        print(f"  {k:22} {v}")
    print(f"\n  q rewrite rate: {s['q_rewritten']}/{s['q_words']} "
          f"({s['q_rewritten']/max(s['q_words'],1):.1%}) of ق-words -> /ɡ/")
    print(f"  empty rows: {empty}")

    top = g_dialect.kashkasha_candidates.most_common(8)
    print(f"\n  kashkasha candidates (ك-final words), top: {top}")

    ok = True
    if s["oov_symbols"]:
        ok = False
        print("\nFAIL: out-of-vocab symbols emitted by the dialect front-end:")
        for ch, c in sorted(s["oov_symbols"].items(), key=lambda kv: -kv[1]):
            print(f"    {ch!r}  U+{ord(ch):04X}  x{c}")
    oov_msa = g_msa.stats()["oov_symbols"]
    if oov_msa:
        ok = False
        print(f"\nFAIL: MSA front-end OOV: {oov_msa}")

    misaligned = s["misaligned_sentences"]
    if misaligned > 0.05 * n:
        print(f"\nWARN: {misaligned}/{n} sentences fell back to per-word phonemization")

    print("\nGATE:", "PASS — every symbol is in the Kokoro vocab" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
