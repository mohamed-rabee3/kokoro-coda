#!/usr/bin/env python3
"""
Transcribe-back intelligibility (plan §9).

Synthesize -> ASR -> compare against the input text. This is the objective
"is it actually saying the words" metric, and unlike listening it is
comparable across checkpoints.

Comparison is on the undiacritized letter skeleton with orthographic variants
normalized (أ/إ/آ -> ا, ة -> ه, ى -> ي). Whisper does not reliably emit
tashkeel, and grading a TTS model on whether an ASR model guessed the same
hamza spelling measures the ASR, not the TTS.

The ق caveat: our model deliberately says /g/ for ق. An MSA-trained ASR often
writes that back as ج or ق inconsistently, so ق-words are reported separately
rather than being allowed to silently inflate the error rate.

Usage:
    python najdi/wer_eval.py --dirs eval/nabra eval/s1_najdivoice
"""

import argparse
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
_ALEF = str.maketrans({"أ": "ا", "إ": "ا", "آ": "ا", "ٱ": "ا", "ة": "ه", "ى": "ي"})
_PUNCT = "،؟!.,?؛:\"'()[]{}—…"


def norm(text: str) -> str:
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")
    text = "".join(" " if c in _PUNCT else c for c in text)
    return " ".join(text.translate(_ALEF).split())


def edit(a: list, b: list) -> int:
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", nargs="+", required=True)
    ap.add_argument("--model", default="openai/whisper-large-v3")
    ap.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="print every ref/hyp pair (unreadable at 200 sentences)",
    )
    args = ap.parse_args()

    import torch
    from transformers import pipeline

    asr = pipeline(
        "automatic-speech-recognition",
        model=args.model,
        device=0 if torch.cuda.is_available() else -1,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    )

    print(f"{'dir':24} {'WER':>7} {'CER':>7}   {'WER(no-ق)':>10}")
    rows = {}
    for d in args.dirs:
        dp = ROOT / d
        refs = (dp / "sentences.txt").read_text(encoding="utf-8").splitlines()
        # categories.txt is optional — the 7-sentence smoke set has none.
        cat_file = dp / "categories.txt"
        cats = (
            cat_file.read_text(encoding="utf-8").splitlines()
            if cat_file.exists()
            else ["all"] * len(refs)
        )
        we = wn = ce = cn = 0
        we_q = wn_q = 0
        per_cat = {}
        details = []
        for i, ref in enumerate(refs):
            wav = dp / f"{i:02d}.wav"
            if not wav.exists():
                continue
            hyp = asr(str(wav), generate_kwargs={"language": "arabic"})["text"]
            r, h = norm(ref), norm(hyp)
            rw, hw = r.split(), h.split()
            e = edit(rw, hw)
            we += e
            wn += len(rw)
            ce += edit(list(r), list(h))
            cn += len(r)
            if "ق" not in r:
                we_q += e
                wn_q += len(rw)
            c = cats[i] if i < len(cats) else "all"
            pe, pn = per_cat.get(c, (0, 0))
            per_cat[c] = (pe + e, pn + len(rw))
            details.append((ref, hyp, e / max(len(rw), 1)))
        rows[d] = (we / max(wn, 1), ce / max(cn, 1), we_q / max(wn_q, 1))
        print(f"{d:24} {rows[d][0]:7.3f} {rows[d][1]:7.3f}   {rows[d][2]:10.3f}")
        if len(per_cat) > 1:
            for c in sorted(per_cat, key=lambda k: -per_cat[k][0] / max(per_cat[k][1], 1)):
                pe, pn = per_cat[c]
                print(f"      {c:10} {pe / max(pn, 1):7.3f}  ({pn} words)")
        if args.verbose:
            for ref, hyp, w in details:
                print(f"    [{w:4.2f}] ref: {ref}")
                print(f"           hyp: {hyp.strip()}")
    return rows


if __name__ == "__main__":
    main()
