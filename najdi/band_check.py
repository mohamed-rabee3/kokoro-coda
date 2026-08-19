#!/usr/bin/env python3
"""
Gate #8 — did the unsupervised high band survive?

Training is band-limited-safe (plan §5): the mel loss stops at 8 kHz and both
discriminators see lowpassed audio. That removes the pressure to zero out the
8-12 kHz band, but it also leaves that band with NO supervision, so the decoder
is free to drift there. Three outcomes:

    zeroed     the muffling we were trying to prevent — band-safe patch failed
    hissing    drift into broadband noise
    tonal      drift into a narrowband whine — looks BRIGHT on an energy-only
               metric, which is how it slipped past the first version of this
               check. Stage 1 ended at 9.3x peak/mean around 9.6-10.2 kHz
               against Nabra's 2.6x, while the genuinely useful 7-9 kHz band
               went 12-15 dB *dimmer*. Aggregate energy said "brighter than
               Nabra"; the shape said "ringing".
    structured dim but following the harmonic/fricative structure — what we want

Hence per-subband peak/mean against the reference, not just total energy. Real
fricative energy is broadband (low peak/mean); an artifact is concentrated.

Compares a checkpoint's output against stock Nabra on the same sentences, since
Nabra's decoder is the reference for what "correct full band" sounds like here.

Usage:
    python najdi/band_check.py --wav-dir eval/stage1_e2 --ref-dir eval/nabra
"""

import argparse
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parent.parent
SR = 24000


def band_profile(wav: np.ndarray, sr: int = SR) -> dict:
    """Energy split around 8 kHz plus a flatness read on the upper band."""
    n = 2048
    hop = 512
    if len(wav) < n:
        return {}
    frames = np.lib.stride_tricks.sliding_window_view(wav, n)[::hop]
    win = np.hanning(n)
    S = np.abs(np.fft.rfft(frames * win, axis=-1)) ** 2
    freqs = np.fft.rfftfreq(n, 1 / sr)

    low = S[:, (freqs >= 300) & (freqs < 8000)]
    high = S[:, (freqs >= 8200) & (freqs < 11500)]
    if high.size == 0:
        return {}

    low_e = low.sum(1).mean()
    high_e = high.sum(1).mean()
    ratio_db = 10 * np.log10((high_e + 1e-20) / (low_e + 1e-20))

    # Spectral flatness of the upper band: ~1 is white noise (hiss),
    # well below 1 means structure (harmonics, shaped fricatives).
    hp = high + 1e-20
    flatness = float(np.exp(np.log(hp).mean(1)).mean() / hp.mean(1).mean())

    # How much of the time the upper band is essentially dead.
    frame_hi = high.sum(1)
    dead = float((frame_hi < 1e-8 * max(low.sum(1).mean(), 1e-20)).mean())

    out = {
        "high_low_db": float(ratio_db),
        "flatness": flatness,
        "dead_frac": dead,
        "centroid_hz": float((S * freqs).sum() / max(S.sum(), 1e-20)),
    }

    # Per-subband shape. Tonal drift concentrates energy in a narrow peak, so
    # peak/mean separates a real fricative band from a whine even when both
    # carry the same total energy.
    avg = S.mean(0)
    for lo, hi in [(7000, 8000), (8000, 9000), (9000, 10000),
                   (10000, 11000), (11000, 12000)]:
        m = (freqs >= lo) & (freqs < hi)
        b = avg[m]
        if b.size:
            out[f"pkmean_{lo//1000}_{hi//1000}k"] = float(b.max() / max(b.mean(), 1e-20))
            out[f"db_{lo//1000}_{hi//1000}k"] = float(10 * np.log10(b.mean() / max(avg.max(), 1e-20) + 1e-20))
    return out


def summarize(d: Path) -> dict:
    rows = []
    for w in sorted(d.glob("*.wav")):
        wav, sr = sf.read(w, dtype="float32")
        if wav.ndim > 1:
            wav = wav.mean(1)
        p = band_profile(wav, sr)
        if p:
            rows.append(p)
    if not rows:
        return {}
    return {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav-dir", required=True)
    ap.add_argument("--ref-dir", default="eval/nabra")
    args = ap.parse_args()

    got = summarize(ROOT / args.wav_dir)
    ref = summarize(ROOT / args.ref_dir)
    if not got:
        raise SystemExit(f"no wavs in {args.wav_dir}")

    print(f"{'metric':16} {'ours':>10} {'nabra':>10}")
    for k in got:
        r = f"{ref.get(k, float('nan')):10.3f}" if ref else "         -"
        print(f"{k:16} {got[k]:10.3f} {r}")

    print()
    # Tonal drift: any subband above 8 kHz far more peaked than the reference.
    tonal = [
        k for k in got
        if k.startswith("pkmean_")
        and not k.startswith("pkmean_7_")
        and ref.get(k)
        and got[k] > max(3.0 * ref[k], 6.0)
    ]

    verdict = "STRUCTURED — high band intact"
    if got["dead_frac"] > 0.5 or got["high_low_db"] < -45:
        verdict = ("ZEROED — the band-safe patch is not doing its job; check "
                   "NAJDI_BAND_LIMIT_HZ was set during training")
    elif tonal:
        verdict = (f"TONAL — narrowband drift in {', '.join(sorted(tonal))}; "
                   "enable high-band self-distillation (plan §5 patch 3). Note "
                   "total energy can look FINE or even bright in this state")
    elif got["flatness"] > 0.6 and got["high_low_db"] > (ref.get("high_low_db", -99) + 6):
        verdict = ("HISSING — decoder drifted into noise above 8 kHz; enable "
                   "high-band self-distillation (plan §5 patch 3) or lower ft_lr")
    print("GATE 8:", verdict)


if __name__ == "__main__":
    main()
