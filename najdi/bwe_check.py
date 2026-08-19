#!/usr/bin/env python3
"""
Judge bandwidth-extended audio before training on it (plan §8.2).

The source is genuinely empty above 8 kHz (16 kHz recordings upsampled to
24 kHz). BWE invents that band. Two ways it goes wrong, and only one is
audible as "brightness":

    buzzing   diffusion SR adds tonal/harmonic junk on fricatives. Shows up
              as a HIGH peak/mean ratio in the invented band — the same
              signature najdi/band_check.py was rewritten to catch, because
              total energy alone reads it as "bright and healthy".
    dead      the model refuses and leaves the band empty; nothing gained.

Reference is Nabra's own output, which is genuine full-band 24 kHz speech —
the distribution we actually want the decoder to learn.
"""
import argparse, glob, os
import numpy as np, soundfile as sf


def bands(path, target_sr=24000):
    a, sr = sf.read(path, dtype="float32")
    if a.ndim > 1:
        a = a.mean(1)
    if sr != target_sr:  # AudioSR emits 48 kHz; compare at the model's rate
        import math
        g = math.gcd(sr, target_sr)
        from scipy.signal import resample_poly
        a = resample_poly(a, target_sr // g, sr // g).astype(np.float32)
        sr = target_sr
    n = 2048
    hop = 512
    if len(a) < n:
        return None
    win = np.hanning(n).astype(np.float32)
    frames = 1 + (len(a) - n) // hop
    S = np.empty((frames, n // 2 + 1), dtype=np.float32)
    for i in range(frames):
        S[i] = np.abs(np.fft.rfft(a[i * hop : i * hop + n] * win))
    freqs = np.fft.rfftfreq(n, 1 / sr)
    out = {}
    for lo, hi in [(1000, 4000), (4000, 8000), (8000, 10000), (10000, 12000)]:
        m = (freqs >= lo) & (freqs < hi)
        if not m.any():
            continue
        b = S[:, m]
        e = float(b.mean())
        pk = float(b.max(axis=1).mean() / max(b.mean(), 1e-12))
        out[f"db_{lo//1000}_{hi//1000}k"] = 20 * np.log10(max(e, 1e-12))
        out[f"pkmean_{lo//1000}_{hi//1000}k"] = pk
    tot = S.mean(1)
    out["centroid_hz"] = float((S * freqs).sum() / max(S.sum(), 1e-12))
    return out


def summarize(d):
    rows = [bands(f) for f in sorted(glob.glob(os.path.join(d, "*.wav")))]
    rows = [r for r in rows if r]
    if not rows:
        return {}
    return {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", nargs="+", required=True)
    ap.add_argument("--ref", default=None, help="genuine full-band reference")
    args = ap.parse_args()
    ref = summarize(args.ref) if args.ref else {}
    stats = {d: summarize(d) for d in args.dirs}
    keys = list(next(iter(stats.values())).keys())
    hdr = f"{'metric':<18}" + "".join(f"{os.path.basename(d.rstrip('/')):>14}" for d in args.dirs)
    if ref:
        hdr += f"{'REF':>14}"
    print(hdr)
    for k in keys:
        line = f"{k:<18}" + "".join(f"{stats[d].get(k, float('nan')):>14.3f}" for d in args.dirs)
        if ref:
            line += f"{ref.get(k, float('nan')):>14.3f}"
        print(line)
    if ref:
        print()
        for d in args.dirs:
            s = stats[d]
            hi = [k for k in s if k.startswith("pkmean_") and ("8_10" in k or "10_12" in k)]
            worst = max((s[k] / max(ref.get(k, 1e-9), 1e-9) for k in hi), default=0)
            gain = s.get("db_8_10k", -99) - ref.get("db_8_10k", -99)
            if worst > 2.0:
                v = f"BUZZ RISK (peak/mean {worst:.1f}x reference)"
            elif gain < -20:
                v = "DEAD (invented band far below reference)"
            else:
                v = "OK"
            print(f"{os.path.basename(d.rstrip('/')):<20} {v}")


if __name__ == "__main__":
    main()
