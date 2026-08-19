#!/usr/bin/env python3
"""
Is this corpus really one speaker?
==================================
The `speaker` column is `<video_id>_SPK<n>` — a *per-video* diarization label.
It says nothing about whether the person in video A is the person in video B,
so the single-speaker claim cannot be checked from metadata. This checks it
acoustically with ECAPA-TDNN embeddings.

The discriminating comparison is within-video vs across-video cosine similarity:

  one speaker    -> across-video ≈ within-video, both high, one tight cluster
  many speakers  -> across-video collapses well below within-video

Usage:
    python najdi/speaker_probe.py [--parquet PATH] [--max-rows N]
"""

import argparse
import glob
import io
from collections import defaultdict

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf
import torch


def load_encoder(device):
    from speechbrain.inference.speaker import EncoderClassifier

    return EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir=".cache/ecapa",
        run_opts={"device": device},
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", default=None)
    ap.add_argument("--max-rows", type=int, default=414)
    args = ap.parse_args()

    path = args.parquet or glob.glob(
        "/home/ai2/.cache/huggingface/hub/datasets--Eman-Fouda*/snapshots/*/data/val-*.parquet"
    )[0]
    d = pq.read_table(path).to_pydict()
    n = min(args.max_rows, len(d["audio"]))
    print(f"{path}\nrows: {n}\n")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    enc = load_encoder(device)

    embs, vids = [], []
    for i in range(n):
        wav, sr = sf.read(io.BytesIO(d["audio"][i]["bytes"]), dtype="float32")
        assert sr == 16000, f"expected 16 kHz, got {sr}"
        with torch.no_grad():
            e = enc.encode_batch(torch.from_numpy(wav).unsqueeze(0).to(device))
        e = e.squeeze().cpu().numpy()
        embs.append(e / np.linalg.norm(e))
        vids.append(d["video_id"][i])

    E = np.stack(embs)
    vids = np.array(vids)
    uniq = sorted(set(vids.tolist()))
    print(f"embeddings: {E.shape}, videos: {len(uniq)}\n")

    S = E @ E.T
    same = np.equal.outer(vids, vids)
    off = ~np.eye(len(E), dtype=bool)

    within = S[same & off]
    across = S[~same]
    print("cosine similarity")
    print(f"  within-video  n={within.size:6d}  mean={within.mean():.3f}  p10={np.percentile(within,10):.3f}  p90={np.percentile(within,90):.3f}")
    print(f"  across-video  n={across.size:6d}  mean={across.mean():.3f}  p10={np.percentile(across,10):.3f}  p90={np.percentile(across,90):.3f}")
    print(f"  gap (within - across) = {within.mean() - across.mean():.3f}")

    # Per-video centroids: if it is one person, these all sit on top of each other.
    cent = np.stack([E[vids == v].mean(0) for v in uniq])
    cent /= np.linalg.norm(cent, axis=1, keepdims=True)
    C = cent @ cent.T
    iu = np.triu_indices(len(uniq), 1)
    print(f"\nper-video centroid similarity: mean={C[iu].mean():.3f}  min={C[iu].min():.3f}  max={C[iu].max():.3f}")

    print("\ncentroid matrix (rows/cols = videos)")
    print("        " + " ".join(f"{v[:6]:>6}" for v in uniq))
    for i, v in enumerate(uniq):
        print(f"{v[:6]:>6}  " + " ".join(f"{C[i,j]:6.2f}" for j in range(len(uniq))))

    # ECAPA/VoxCeleb convention: ~0.25 is the usual same/different decision region.
    thr = 0.35
    frac = float((C[iu] > thr).mean())
    print(f"\nvideo pairs above {thr}: {frac:.1%}")
    if frac > 0.9:
        verdict = "ONE SPEAKER — all videos cluster together"
    elif frac < 0.2:
        verdict = "MANY SPEAKERS — videos are acoustically distinct people"
    else:
        verdict = "MIXED — a dominant speaker plus others, or heavy channel variation"
    print(f"VERDICT: {verdict}")


if __name__ == "__main__":
    main()
