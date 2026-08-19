#!/usr/bin/env python3
"""
Build the StyleTTS2 file lists and the OOD text pool.

Format is `path|phonemes|speaker_id`, paths relative to data_params.root_path.
Phonemes come from the recovered tashkeel (najdi/tashkeel_fix.py) run through
the Najdi front-end (najdi/g2p.py), so what the model is trained on matches
what inference will produce, symbol for symbol.

OOD_texts.txt feeds Stage 2's SLM adversarial loss. It has to be Najdi — the
upstream recipe ships German, and leaving that in place would have the WavLM
discriminator scoring our model against a different language's prosody.

Espeak is not thread-safe in this binding, so G2P is spread over processes.

Usage:
    python najdi/build_lists.py [--procs 12]
"""

import argparse
import os
from multiprocessing import Pool
from pathlib import Path

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent.parent
_G2P = None


def _init():
    global _G2P
    from najdi.g2p import NajdiG2P

    _G2P = NajdiG2P(dialect=True)


def _phonemize(args):
    key, speaker, bare, raw, existing = args
    from najdi.tashkeel_fix import recover_tashkeel

    text, cov = recover_tashkeel(bare, raw, existing)
    try:
        ph = _G2P.phonemize(text)
    except Exception:
        return key, speaker, "", 0.0
    return key, speaker, ph, cov


def rows_from(sel):
    for r in sel.itertuples():
        yield (r.audio_file, r.speaker, r.text or "",
               r.text_tashkeel_raw or "", r.text_tashkeel or "")


def write_list(path, entries, spk_ids, min_phonemes):
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for key, speaker, ph, _cov in entries:
            if len(ph) < min_phonemes:
                continue
            rel = f"{speaker}/{Path(key).stem}.wav"
            f.write(f"{rel}|{ph}|{spk_ids[speaker]}\n")
            n += 1
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--procs", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    ap.add_argument("--selection", default="selection.parquet")
    ap.add_argument("--val-selection", default="selection_val.parquet")
    ap.add_argument("--min-phonemes", type=int, default=20)
    ap.add_argument("--ood-min", type=int, default=50)
    ap.add_argument("--ood-max", type=int, default=20000)
    args = ap.parse_args()

    sel = pq.read_table(ROOT / "data" / args.selection).to_pandas()
    spk_ids = {s: i for i, s in enumerate(sorted(sel.speaker.unique()))}
    print(f"train rows {len(sel)}  speakers {len(spk_ids)}  procs {args.procs}")

    with Pool(args.procs, initializer=_init) as p:
        entries = p.map(_phonemize, rows_from(sel), chunksize=256)

    n = write_list(ROOT / "data" / "train_list.txt", entries, spk_ids, args.min_phonemes)
    print(f"train_list.txt: {n} lines")

    val_path = ROOT / "data" / args.val_selection
    if val_path.exists():
        vsel = pq.read_table(val_path).to_pandas()
        # Validation speakers are different videos, so they get ids too; the
        # style encoder takes its reference from the clip's own mel, not the id.
        for s in sorted(vsel.speaker.unique()):
            spk_ids.setdefault(s, len(spk_ids))
        with Pool(args.procs, initializer=_init) as p:
            ventries = p.map(_phonemize, rows_from(vsel), chunksize=64)
        n = write_list(ROOT / "data" / "val_list.txt", ventries, spk_ids, args.min_phonemes)
        print(f"val_list.txt: {n} lines")
    else:
        print(f"WARN: {val_path} missing — no validation list written")

    # OOD pool: phoneme strings the model is scored on without matching audio.
    #
    # Drawn from train rows the selection REJECTED, not from the `long` split.
    # The long split's text is deliberately bare — no tashkeel — so espeak would
    # guess its short vowels and the OOD phoneme distribution would drift away
    # from what the model is actually trained on. Rejected train rows fail on
    # audio quality, not text, so their recovered tashkeel is just as good.
    full = pq.read_table(
        ROOT / "data" / "manifest_train.parquet",
        columns=["audio_file", "speaker", "text", "text_tashkeel", "text_tashkeel_raw"],
    ).to_pandas()
    rejected = full[~full.audio_file.isin(set(sel.audio_file))]
    print(f"OOD source: {len(rejected)} rejected train rows")
    take = rejected.sample(n=min(len(rejected), args.ood_max * 2), random_state=0)
    with Pool(args.procs, initializer=_init) as p:
        res = p.map(_phonemize, rows_from(take), chunksize=256)

    ood, seen = [], set()
    for _k, _s, ph, _c in res:
        if len(ph) >= args.ood_min and ph not in seen:
            seen.add(ph)
            ood.append(ph)
        if len(ood) >= args.ood_max:
            break

    # One phoneme string per line, no pipes. meldataset.py picks the field with
    # `idx = 1 if ".wav" in tl[0].split("|")[0] else 0`, so a leading label
    # column would be read AS the text. kikiri's own file is bare lines too.
    with open(ROOT / "data" / "OOD_texts.txt", "w", encoding="utf-8") as f:
        for ph in ood[: args.ood_max]:
            f.write(f"{ph}\n")
    print(f"OOD_texts.txt: {min(len(ood), args.ood_max)} lines")


if __name__ == "__main__":
    main()
