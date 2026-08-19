#!/usr/bin/env python3
"""
Long-form synthesis by chunking (plan §9, "document the chunking requirement").

Two hard limits make this necessary, and neither one errors — they just
degrade:

  510 tokens   Kokoro's positional limit. synth.py skips anything longer;
               there is no graceful truncation.
  ~12.4 s      the corpus duration ceiling. Nothing longer was trained on, and
               measured WER on this data climbs 0.10-0.13 at 13 s -> 0.39 at
               24 s -> 0.50-0.63 at 31 s. A single long utterance does not
               fail loudly, it just gets progressively less intelligible.

So we split on punctuation, synthesize each piece separately, and rejoin with
silences sized to the boundary that produced them. Punctuation is the right
split point because it is where a speaker would breathe anyway — splitting on
a token budget alone would cut mid-clause and the prosody would audibly break.

Usage:
    python najdi/synth_long.py --model M.pth --voice V.pt \
        --text-file para.txt --out-dir eval/para
"""

import argparse
import re
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

ROOT = Path(__file__).resolve().parent.parent
REPO_ID = "oddadmix/Nabra-82M-v0.1"
SR = 24000

# Pause after each boundary type, in seconds. Ellipsis gets the longest gap:
# in this dialect it marks a hesitation//thinking beat, not a list separator.
PAUSE = {"ellipsis": 0.55, "stop": 0.42, "comma": 0.22, "forced": 0.12}

_SPLIT = re.compile(r"(\.\.+|[.؟?!]|[،,])")

# The decoder emits ~0.21 s of leading and ~0.18 s of trailing silence on every
# utterance. Concatenating raw chunks therefore stacks that onto the pause we
# actually want: 0.21 + 0.42 + 0.18 = 0.8 s of dead air at every boundary, which
# does not sound like a pause, it sounds like separate recordings spliced
# together. Trim the model's padding first, then insert the intended gap, so the
# PAUSE table above means what it says.
TRIM_THRESH = 0.02  # fraction of chunk peak
KEEP_MARGIN = 0.020  # s of silence retained either side, so onsets breathe
FADE = 0.005  # s, guards against a click if a trim lands mid-waveform


def trim_silence(a: np.ndarray, sr: int = SR) -> np.ndarray:
    """Strip the decoder's leading/trailing silence, keeping a small margin."""
    peak = float(np.abs(a).max())
    if peak <= 0:
        return a
    idx = np.where(np.abs(a) > peak * TRIM_THRESH)[0]
    if len(idx) == 0:
        return a
    m = int(sr * KEEP_MARGIN)
    a = a[max(0, idx[0] - m) : min(len(a), idx[-1] + m)].copy()
    n = int(sr * FADE)
    if len(a) > 2 * n:
        a[:n] *= np.linspace(0.0, 1.0, n, dtype=np.float32)
        a[-n:] *= np.linspace(1.0, 0.0, n, dtype=np.float32)
    return a


def chunk(text: str, max_tokens: int, count_tokens):
    """Split into (text, pause_after) pairs, each under max_tokens."""
    parts = _SPLIT.split(text)
    units = []  # (text, boundary_kind)
    buf = ""
    for p in parts:
        if not p:
            continue
        if _SPLIT.fullmatch(p):
            kind = ("ellipsis" if p.startswith("..")
                    else "comma" if p in "،," else "stop")
            if buf.strip():
                units.append((buf.strip(), kind))
            buf = ""
        else:
            buf += p
    if buf.strip():
        units.append((buf.strip(), "stop"))

    # Merge neighbours while they fit, so we do not chop at every comma and
    # lose the phrase-level prosody the predictor was trained to produce.
    out = []
    cur, cur_kind = "", "stop"
    for t, kind in units:
        cand = (cur + " " + t).strip() if cur else t
        if cur and count_tokens(cand) > max_tokens:
            out.append((cur, cur_kind))
            cur, cur_kind = t, kind
        else:
            cur, cur_kind = cand, kind
    if cur:
        out.append((cur, cur_kind))

    # Anything still too long has no punctuation to split on; break on words.
    final = []
    for t, kind in out:
        if count_tokens(t) <= max_tokens:
            final.append((t, kind))
            continue
        words, acc = t.split(), ""
        for w in words:
            cand = (acc + " " + w).strip()
            if acc and count_tokens(cand) > max_tokens:
                final.append((acc, "forced"))
                acc = w
            else:
                acc = cand
        if acc:
            final.append((acc, kind))
    return final


def token_reader(env_path: str | None = None):
    env = ROOT / ".env"
    if not env.exists():
        return None
    for line in env.read_text().splitlines():
        if line.strip().startswith("HF_TOKEN"):
            return line.split("=", 1)[1].strip().strip('"').strip("'")
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None)
    ap.add_argument("--voice", default=None)
    ap.add_argument("--text-file", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--max-tokens", type=int, default=120,
                    help="~17 tok/s, so 120 lands near 7 s — inside the trained "
                         "range with margin, not at the 510 hard cap")
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--allow-base-voice", action="store_true")
    args = ap.parse_args()

    if args.model and not args.voice and not args.allow_base_voice:
        raise SystemExit("pass --voice matched to this checkpoint (see synth.py)")

    from huggingface_hub import hf_hub_download
    from kokoro import KModel

    from najdi.g2p import NajdiG2P

    tok = token_reader()
    config = hf_hub_download(REPO_ID, "config.json", token=tok)
    model_path = args.model or hf_hub_download(REPO_ID, "kokoro_arabic.pth", token=tok)
    voice_path = args.voice or hf_hub_download(REPO_ID, "af_msa.pt", token=tok)

    model = KModel(repo_id=REPO_ID, config=config, model=model_path,
                   disable_complex=True).eval().to(args.device)
    model.vocab.update({"ʕ": 7, "ħ": 8})
    voice = torch.load(voice_path, map_location=args.device, weights_only=True)
    g2p = NajdiG2P(dialect=True)

    def n_tokens(t):
        ph = g2p.phonemize(t)
        return len([c for c in ph if model.vocab.get(c) is not None])

    text = Path(args.text_file).read_text(encoding="utf-8").strip()
    pieces = chunk(text, args.max_tokens, n_tokens)

    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"model : {model_path}")
    print(f"voice : {voice_path}")
    print(f"chunks: {len(pieces)}\n")

    audio_all, rtfs, rows = [], [], []
    for i, (t, kind) in enumerate(pieces):
        ph = g2p.phonemize(t)
        n = len([c for c in ph if model.vocab.get(c) is not None])
        t0 = time.time()
        with torch.no_grad():
            out = model(ph, voice[len(ph) - 1], speed=args.speed, return_output=True)
        el = time.time() - t0
        a = out.audio.detach().cpu().numpy().astype(np.float32)
        raw = len(a) / SR
        a = trim_silence(a)
        dur = len(a) / SR
        rtfs.append(el / max(raw, 1e-6))
        audio_all.append(a)
        audio_all.append(np.zeros(int(SR * PAUSE[kind]), dtype=np.float32))
        sf.write(out_dir / f"{i:02d}.wav", a, SR)
        rows.append((i, dur, n, kind, t))
        flag = "  <-- LONG" if dur > 12.4 else ""
        print(f"  [{i:02d}] {dur:5.2f}s {n:4d} tok  {kind:8} {t[:46]}{flag}")

    full = np.concatenate(audio_all)
    sf.write(out_dir / "FULL.wav", full, SR)
    (out_dir / "chunks.txt").write_text(
        "\n".join(f"{i:02d}\t{d:.2f}s\t{k}\t{t}" for i, d, n, k, t in rows),
        encoding="utf-8",
    )
    print(f"\nFULL.wav  {len(full)/SR:.1f}s  ({len(pieces)} chunks)")
    print(f"longest chunk: {max(r[1] for r in rows):.2f}s  (ceiling 12.4 s)")
    print(f"median RTF ({args.device}): {float(np.median(rtfs)):.3f}")


if __name__ == "__main__":
    main()
