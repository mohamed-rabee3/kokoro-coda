#!/usr/bin/env python3
"""
Pass B — per-clip acoustic features, no audio kept.

Streams every shard, decodes the embedded 16 kHz WAVs, and records two things
per clip:

  f0_median   voiced-frame median F0. Feeds the gender gate. Channel-robust,
              which matters because ECAPA is not (see najdi/speaker_probe.py:
              within-video cosine 0.738 vs across-video 0.679 on this corpus —
              the shared YouTube recording chain saturates the embedding).
  ecapa       192-d speaker embedding, float16. Feeds female-speaker
              clustering and per-clip outlier rejection.

Audio is discarded as we go, so disk stays flat. Both features are capped at
--f0-per-speaker clips per speaker: a per-speaker median F0 and a per-speaker
ECAPA centroid both converge in a few dozen clips, and the cap is checked before
decoding, which is what keeps this pass to ~35 min instead of ~3 h. Per-clip
embeddings for the clips we actually keep are computed later during
materialization, where the audio is already decoded.

Output: data/features.parquet

Usage:
    python najdi/audio_features.py [--workers 8] [--batch 64]
"""

import argparse
import io
import queue
import threading
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import soundfile as sf
import torch
import torchaudio
from huggingface_hub import HfFileSystem

REPO = "Eman-Fouda/youtube-001-part02-tashkeel"
ROOT = Path(__file__).resolve().parent.parent
SR = 16000


def token() -> str:
    for line in (ROOT / ".env").read_text().splitlines():
        if line.strip().startswith("HF_TOKEN"):
            return line.split("=", 1)[1].strip().strip('"').strip("'")
    raise SystemExit("HF_TOKEN not found in .env")


def median_f0(wav: torch.Tensor, sr: int = SR) -> float:
    """Voiced-frame median F0 on GPU. Returns nan when too few voiced frames."""
    try:
        f0 = torchaudio.functional.detect_pitch_frequency(
            wav.unsqueeze(0), sample_rate=sr, freq_low=65, freq_high=350
        ).squeeze()
    except Exception:
        return float("nan")
    f0 = f0[(f0 > 65) & (f0 < 350)]
    return float(f0.median()) if f0.numel() >= 10 else float("nan")


def shard_reader(fs, paths, out_q, cols):
    """Download + parse shards in the background so the GPU never waits."""
    for p in paths:
        try:
            with fs.open(p, "rb") as fh:
                tbl = pq.ParquetFile(fh).read(columns=cols)
            out_q.put((p, tbl))
        except Exception as e:
            out_q.put((p, e))
    out_q.put(None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--f0-per-speaker", type=int, default=40)
    ap.add_argument("--prefetch", type=int, default=3)
    ap.add_argument("--limit-shards", type=int, default=0)
    ap.add_argument("--split", default="train", choices=["train", "val", "long"])
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if args.out is None:
        suffix = "" if args.split == "train" else f"_{args.split}"
        args.out = str(ROOT / "data" / f"features{suffix}.parquet")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    from speechbrain.inference.speaker import EncoderClassifier

    enc = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir=str(ROOT / ".cache" / "ecapa"),
        run_opts={"device": device},
    )
    enc.eval()

    fs = HfFileSystem(token=token())
    prefix = f"datasets/{REPO}/data"
    paths = sorted(f for f in fs.ls(prefix, detail=False) if f"/{args.split}-" in f)
    if args.limit_shards:
        paths = paths[: args.limit_shards]
    print(f"shards: {len(paths)}  device: {device}")

    cols = ["audio_file", "speaker", "duration", "audio"]
    q: queue.Queue = queue.Queue(maxsize=args.prefetch)
    # A few readers share the queue; 8 parallel streams gave ~32 MB/s vs 4 MB/s
    # for one, and the network is the bottleneck here, not the GPU.
    chunks = [paths[i :: args.workers] for i in range(args.workers)]
    threads = [
        threading.Thread(target=shard_reader, args=(fs, c, q, cols), daemon=True)
        for c in chunks
    ]
    for t in threads:
        t.start()

    keys, spks, f0s, embs = [], [], [], []
    f0_count: Counter = Counter()
    done = finished = 0
    t0 = time.time()

    while finished < args.workers:
        item = q.get()
        if item is None:
            finished += 1
            continue
        path, tbl = item
        if isinstance(tbl, Exception):
            print(f"  FAILED {path}: {tbl}", flush=True)
            continue

        d = tbl.to_pydict()
        n = len(d["audio_file"])
        batch, batch_meta = [], []

        def flush():
            if not batch:
                return
            lens = torch.tensor([w.shape[0] for w in batch], dtype=torch.float32)
            mx = int(lens.max())
            padded = torch.zeros(len(batch), mx)
            for i, w in enumerate(batch):
                padded[i, : w.shape[0]] = w
            with torch.no_grad():
                e = enc.encode_batch(
                    padded.to(device), wav_lens=(lens / mx).to(device)
                ).squeeze(1)
            e = torch.nn.functional.normalize(e, dim=-1).cpu().numpy().astype(np.float16)
            for i, (k, s, want_f0) in enumerate(batch_meta):
                keys.append(k)
                spks.append(s)
                embs.append(e[i])
                f0s.append(
                    median_f0(batch[i].to(device)) if want_f0 else float("nan")
                )
            batch.clear()
            batch_meta.clear()

        for i in range(n):
            spk = d["speaker"][i]
            # Cap before decoding. A per-speaker centroid and a per-speaker
            # median F0 both converge in a few dozen clips, and decoding is the
            # single most expensive thing in this loop — skipping it here is
            # what turns a ~3 h pass into a ~35 min one. Per-clip embeddings for
            # the clips we actually keep get computed during materialization,
            # where the audio is already decoded.
            if f0_count[spk] >= args.f0_per_speaker:
                continue
            try:
                wav, sr = sf.read(io.BytesIO(d["audio"][i]["bytes"]), dtype="float32")
            except Exception:
                continue
            if sr != SR or wav.ndim != 1 or wav.shape[0] < SR // 2:
                continue
            f0_count[spk] += 1
            want_f0 = True
            batch.append(torch.from_numpy(wav))
            batch_meta.append((d["audio_file"][i], spk, want_f0))
            if len(batch) >= args.batch:
                flush()
        flush()

        done += 1
        el = time.time() - t0
        print(f"  [{done:3d}/{len(paths)}] {el:6.1f}s  clips={len(keys)}  "
              f"eta {(len(paths)-done)*el/max(done,1):6.1f}s", flush=True)

    E = np.stack(embs)
    out = Path(args.out)
    pq.write_table(
        pa.table({
            "audio_file": keys,
            "speaker": spks,
            "f0_median": f0s,
            "ecapa": [row.tolist() for row in E],
        }),
        out,
        compression="zstd",
    )
    n_f0 = int(np.sum(~np.isnan(np.array(f0s))))
    print(f"\nwrote {out}  clips={len(keys)}  with-F0={n_f0}  "
          f"{out.stat().st_size/1e6:.1f} MB  in {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
