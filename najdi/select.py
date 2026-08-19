#!/usr/bin/env python3
"""
Turn the manifest + features into a training set (plan §3).

Three decisions, in order:

  1. Row quality filter — SNR, SQUIM, music, transcript agreement, tashkeel
     coverage, duration. Nulls mean "not measured", not "bad": 19% of rows have
     no cer_yt_vs_cohere and no boundary_clipped, and dropping those silently
     would throw away a fifth of the corpus for no stated reason.

  2. Gender gate — per-speaker median F0, female band only. Per *speaker*, not
     per clip: single-clip F0 octave-errors, a median over dozens does not.

  3. Voice core — the subset that will carry the shipped voice. Grown from the
     best female speaker outward, requiring both a close ECAPA centroid and a
     close median F0. ECAPA alone is not trustworthy on this corpus (within-video
     cosine 0.738 vs across-video 0.679 — the shared YouTube recording chain
     saturates it), so F0 acts as the second, channel-robust key.

Outputs data/selection.parquet: every surviving row, its speaker_id, and an
`in_voice_core` flag.

Usage:
    python najdi/select.py --report      # retention curves, pick thresholds
    python najdi/select.py               # write the selection
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent.parent


def load(args) -> pd.DataFrame:
    man = pq.read_table(ROOT / "data" / args.manifest).to_pandas()
    feat = pq.read_table(ROOT / "data" / args.features).to_pandas()

    spk = feat.groupby("speaker").agg(
        f0_speaker=("f0_median", lambda s: np.nanmedian(s.astype(float))),
        n_f0=("f0_median", lambda s: int(np.sum(~np.isnan(s.astype(float))))),
    )
    cent = (
        feat.groupby("speaker")["ecapa"]
        .apply(lambda s: np.stack(s.to_numpy()).astype(np.float32).mean(0))
        .to_dict()
    )
    C = {k: v / (np.linalg.norm(v) + 1e-9) for k, v in cent.items()}
    df = man.merge(spk, left_on="speaker", right_index=True, how="left")
    return df, C


def row_quality(df: pd.DataFrame, a) -> pd.Series:
    """Null-tolerant row filter. NaN => not measured => not disqualifying."""
    def ok(col, fn):
        v = df[col]
        return v.isna() | fn(v)

    return (
        (df.duration >= a.min_dur)
        & (df.duration <= a.max_dur)
        & ok("snr_db", lambda v: v >= a.min_snr)
        & ok("squim_mos", lambda v: v >= a.min_squim)
        & ok("music_score", lambda v: v <= a.max_music)
        & ok("cer_yt_vs_cohere", lambda v: v <= a.max_cer)
        & ok("coverage", lambda v: v >= a.min_coverage)
        & (df.boundary_clipped != True)  # noqa: E712 — NaN must survive this
    )


def report(df: pd.DataFrame, a):
    tot_h = df.duration.sum() / 3600
    print(f"corpus: {len(df)} rows, {tot_h:.1f} h, {df.speaker.nunique()} speakers\n")

    print("individual filters (hours retained):")
    checks = [
        ("duration in range", (df.duration >= a.min_dur) & (df.duration <= a.max_dur)),
        (f"snr_db >= {a.min_snr}", df.snr_db.isna() | (df.snr_db >= a.min_snr)),
        (f"squim >= {a.min_squim}", df.squim_mos.isna() | (df.squim_mos >= a.min_squim)),
        (f"music <= {a.max_music}", df.music_score.isna() | (df.music_score <= a.max_music)),
        (f"cer <= {a.max_cer}", df.cer_yt_vs_cohere.isna() | (df.cer_yt_vs_cohere <= a.max_cer)),
        (f"coverage >= {a.min_coverage}", df.coverage.isna() | (df.coverage >= a.min_coverage)),
        ("not boundary_clipped", df.boundary_clipped != True),  # noqa: E712
    ]
    for name, m in checks:
        print(f"  {name:24} {df.duration[m].sum()/3600:7.1f} h  ({m.mean():5.1%})")

    q = row_quality(df, a)
    print(f"\n  ALL COMBINED           {df.duration[q].sum()/3600:7.1f} h  ({q.mean():5.1%})")

    f0 = df.groupby("speaker").f0_speaker.first().dropna()
    print(f"\nper-speaker median F0 (n={len(f0)}):")
    for lo, hi in [(0, 130), (130, 145), (145, 155), (155, 170), (170, 190), (190, 400)]:
        sel = f0[(f0 >= lo) & (f0 < hi)]
        h = df[df.speaker.isin(sel.index)].duration.sum() / 3600
        tag = "  <- boundary, needs a listen" if (lo, hi) in [(145, 155), (155, 170)] else ""
        print(f"  {lo:3d}-{hi:3d} Hz  {len(sel):4d} speakers  {h:6.1f} h{tag}")

    fem = f0[f0 >= a.female_f0].index
    both = q & df.speaker.isin(fem)
    print(f"\nfemale (F0 >= {a.female_f0}) + quality: "
          f"{df.duration[both].sum()/3600:.1f} h, {df[both].speaker.nunique()} speakers")


def build_voice_core(df: pd.DataFrame, C: dict, a) -> set:
    """Grow the voice core from the strongest female speaker outward."""
    g = df.groupby("speaker").agg(
        hours=("duration", lambda s: s.sum() / 3600),
        squim=("squim_mos", "median"),
        snr=("snr_db", "median"),
        f0=("f0_speaker", "first"),
    )
    g = g[g.index.isin(C)]
    if g.empty:
        return set()

    seed = (g.hours * g.squim.fillna(0)).idxmax()
    v0, f0_0 = C[seed], g.f0[seed]
    print(f"\nvoice-core seed: {seed}  {g.hours[seed]:.2f} h  F0 {f0_0:.0f} Hz")

    cand = g.assign(cos=[float(C[s] @ v0) for s in g.index])
    cand = cand[(cand.f0 - f0_0).abs() <= a.core_f0_tol].sort_values("cos", ascending=False)

    core, hours = [], 0.0
    for s, row in cand.iterrows():
        if hours >= a.core_hours:
            break
        core.append(s)
        hours += row.hours
    print(f"voice core: {len(core)} speakers, {hours:.1f} h, "
          f"cos range {cand.cos[core].min():.3f}-{cand.cos[core].max():.3f}, "
          f"F0 {cand.f0[core].min():.0f}-{cand.f0[core].max():.0f} Hz")
    return set(core)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="manifest_train.parquet")
    ap.add_argument("--features", default="features.parquet")
    ap.add_argument("--no-core", action="store_true",
                    help="skip voice-core selection (validation split)")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--min-dur", type=float, default=1.5)
    ap.add_argument("--max-dur", type=float, default=12.0)
    ap.add_argument("--min-snr", type=float, default=20.0)
    ap.add_argument("--min-squim", type=float, default=4.2)
    ap.add_argument("--max-music", type=float, default=0.05)
    ap.add_argument("--max-cer", type=float, default=0.10)
    ap.add_argument("--min-coverage", type=float, default=0.75)
    ap.add_argument("--female-f0", type=float, default=165.0)
    ap.add_argument("--core-hours", type=float, default=25.0)
    ap.add_argument("--core-f0-tol", type=float, default=15.0)
    ap.add_argument("--out", default="selection.parquet")
    args = ap.parse_args()

    df, C = load(args)
    if args.report:
        report(df, args)
        return

    keep = row_quality(df, args)
    f0 = df.groupby("speaker").f0_speaker.first()
    female = set(f0[f0 >= args.female_f0].index)
    keep &= df.speaker.isin(female)

    sel = df[keep].copy()
    core = set() if args.no_core else build_voice_core(sel, C, args)
    sel["in_voice_core"] = sel.speaker.isin(core)

    ids = {s: i for i, s in enumerate(sorted(sel.speaker.unique()))}
    sel["speaker_id"] = sel.speaker.map(ids)

    out = ROOT / "data" / args.out
    cols = ["audio_file", "speaker", "speaker_id", "video_id", "duration",
            "text", "text_tashkeel", "text_tashkeel_raw", "coverage",
            "squim_mos", "snr_db", "in_voice_core"]
    pq.write_table(pa.Table.from_pandas(sel[cols], preserve_index=False),
                   out, compression="zstd")
    print(f"\nwrote {out}")
    print(f"  rows {len(sel)}  hours {sel.duration.sum()/3600:.1f}  "
          f"speakers {sel.speaker.nunique()}")
    print(f"  voice core: {sel.in_voice_core.sum()} rows, "
          f"{sel[sel.in_voice_core].duration.sum()/3600:.1f} h")


if __name__ == "__main__":
    main()
