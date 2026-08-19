#!/usr/bin/env python3
"""
Median F0 per video — a channel-robust check on the single-speaker claim.

ECAPA cosine is inflated when every clip shares one recording chain (same
platform, same codec, same 16 kHz resample), so a high similarity is not by
itself evidence of one speaker. Median F0 does not have that problem: an adult
male sits near 100-130 Hz and an adult female near 180-220 Hz, whatever the
channel. If videos land in both bands, the corpus is definitively multi-speaker.

Usage:
    python najdi/f0_probe.py
"""

import glob
import io
from collections import defaultdict

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf
import torch
import torchaudio


def median_f0(wav: np.ndarray, sr: int) -> float:
    """Voiced-frame median F0 via torchaudio's autocorrelation detector."""
    x = torch.from_numpy(wav).float().unsqueeze(0)
    try:
        f0 = torchaudio.functional.detect_pitch_frequency(
            x, sample_rate=sr, freq_low=60, freq_high=400
        ).squeeze()
    except Exception:
        return float("nan")
    f0 = f0[(f0 > 60) & (f0 < 400)]
    return float(f0.median()) if f0.numel() > 10 else float("nan")


def main():
    path = glob.glob(
        "/home/ai2/.cache/huggingface/hub/datasets--Eman-Fouda*/snapshots/*/data/val-*.parquet"
    )[0]
    d = pq.read_table(path).to_pydict()

    per_video = defaultdict(list)
    for i in range(len(d["audio"])):
        wav, sr = sf.read(io.BytesIO(d["audio"][i]["bytes"]), dtype="float32")
        f = median_f0(wav, sr)
        if not np.isnan(f):
            per_video[d["video_id"][i]].append(f)

    print(f"{'video':16} {'n':>4} {'medF0':>7} {'p25':>7} {'p75':>7}   guess")
    rows = []
    for v, fs in sorted(per_video.items(), key=lambda kv: np.median(kv[1])):
        m = float(np.median(fs))
        guess = "male" if m < 155 else "female"
        rows.append((v, m, guess))
        print(f"{v:16} {len(fs):4d} {m:7.1f} {np.percentile(fs,25):7.1f} "
              f"{np.percentile(fs,75):7.1f}   {guess}")

    med = np.array([r[1] for r in rows])
    males = sum(1 for r in rows if r[2] == "male")
    print(f"\nvideos: {len(rows)}   male-band: {males}   female-band: {len(rows)-males}")
    print(f"per-video median F0: min={med.min():.1f} max={med.max():.1f} "
          f"spread={med.max()-med.min():.1f} Hz")

    if males and len(rows) - males:
        print("\nVERDICT: MULTI-SPEAKER — videos fall in both the male and female F0 bands.")
    elif med.max() - med.min() > 50:
        print("\nVERDICT: SUSPICIOUS — same band, but >50 Hz spread across videos.")
    else:
        print("\nVERDICT: consistent with a single speaker.")


if __name__ == "__main__":
    main()
