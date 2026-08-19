#!/usr/bin/env python3
"""
Pass C — materialize the selected clips as 24 kHz WAVs.

StyleTTS2 wants mono 24 kHz 16-bit WAV on disk. The corpus is 16 kHz embedded
in parquet, so every kept row gets decoded, resampled with soxr (VHQ), peak
normalized, and written under data/audio/<speaker>/.

Also computes a per-clip ECAPA embedding while the audio is already decoded —
that is the cheap moment to do it, and it feeds the per-clip outlier rejection
that the speaker-level centroids in pass B cannot do.

Disk is the binding constraint (75 GB total, 43.6 GB dataset), so this writes
only what selection.parquet kept and never caches a shard.

Usage:
    python najdi/build_audio.py [--workers 8] [--split train]
"""

import argparse
import io
import queue
import threading
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import soundfile as sf
import soxr
import torch
from huggingface_hub import HfFileSystem

REPO = "Eman-Fouda/youtube-001-part02-tashkeel"
ROOT = Path(__file__).resolve().parent.parent
SRC_SR, DST_SR = 16000, 24000


def token() -> str:
    for line in (ROOT / ".env").read_text().splitlines():
        if line.strip().startswith("HF_TOKEN"):
            return line.split("=", 1)[1].strip().strip('"').strip("'")
    raise SystemExit("HF_TOKEN not found in .env")


def reader(fs, paths, out_q, cols):
    for p in paths:
        try:
            with fs.open(p, "rb") as fh:
                out_q.put(pq.ParquetFile(fh).read(columns=cols))
        except Exception as e:
            out_q.put(e)
    out_q.put(None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--split", default="train")
    ap.add_argument("--selection", default="selection.parquet")
    ap.add_argument("--audio-dir", default="audio")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--no-ecapa", action="store_true")
    args = ap.parse_args()

    sel = pq.read_table(ROOT / "data" / args.selection,
                        columns=["audio_file", "speaker"]).to_pandas()
    want = dict(zip(sel.audio_file, sel.speaker))
    print(f"selection: {len(want)} clips")

    audio_root = ROOT / "data" / args.audio_dir
    audio_root.mkdir(parents=True, exist_ok=True)

    enc = None
    if not args.no_ecapa:
        from speechbrain.inference.speaker import EncoderClassifier

        enc = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir=str(ROOT / ".cache" / "ecapa"),
            run_opts={"device": "cuda" if torch.cuda.is_available() else "cpu"},
        )
        enc.eval()
    dev = next(enc.mods.parameters()).device if enc else None

    fs = HfFileSystem(token=token())
    paths = sorted(f for f in fs.ls(f"datasets/{REPO}/data", detail=False)
                   if f"/{args.split}-" in f)
    q: queue.Queue = queue.Queue(maxsize=3)
    chunks = [paths[i :: args.workers] for i in range(args.workers)]
    for c in chunks:
        threading.Thread(target=reader, args=(fs, c, q, ["audio_file", "audio"]),
                         daemon=True).start()

    keys, embs = [], []
    written = skipped = finished = done = 0
    t0 = time.time()

    while finished < args.workers:
        tbl = q.get()
        if tbl is None:
            finished += 1
            continue
        if isinstance(tbl, Exception):
            print(f"  FAILED: {tbl}", flush=True)
            continue

        d = tbl.to_pydict()
        batch, batch_keys = [], []

        def flush():
            if not batch or enc is None:
                batch.clear(); batch_keys.clear(); return
            lens = torch.tensor([w.shape[0] for w in batch], dtype=torch.float32)
            mx = int(lens.max())
            pad = torch.zeros(len(batch), mx)
            for i, w in enumerate(batch):
                pad[i, : w.shape[0]] = torch.from_numpy(w)
            with torch.no_grad():
                e = enc.encode_batch(pad.to(dev), wav_lens=(lens / mx).to(dev)).squeeze(1)
            e = torch.nn.functional.normalize(e, dim=-1).cpu().numpy().astype(np.float16)
            keys.extend(batch_keys)
            embs.extend(e)
            batch.clear(); batch_keys.clear()

        for i in range(len(d["audio_file"])):
            key = d["audio_file"][i]
            spk = want.get(key)
            if spk is None:
                continue
            try:
                wav, sr = sf.read(io.BytesIO(d["audio"][i]["bytes"]), dtype="float32")
            except Exception:
                skipped += 1
                continue
            if wav.ndim != 1 or sr != SRC_SR:
                skipped += 1
                continue

            out = soxr.resample(wav, SRC_SR, DST_SR, quality="VHQ")
            peak = float(np.abs(out).max())
            if peak < 1e-5:
                skipped += 1
                continue
            out = (out / peak) * 0.95

            dest = audio_root / spk / (Path(key).stem + ".wav")
            dest.parent.mkdir(parents=True, exist_ok=True)
            sf.write(dest, out.astype(np.float32), DST_SR, subtype="PCM_16")
            written += 1

            batch.append(wav)
            batch_keys.append(key)
            if len(batch) >= args.batch:
                flush()
        flush()

        done += 1
        el = time.time() - t0
        print(f"  [{done:3d}/{len(paths)}] {el:6.1f}s  written={written} "
              f"skipped={skipped}  eta {(len(paths)-done)*el/max(done,1):6.1f}s", flush=True)

    print(f"\nwrote {written} wavs to {audio_root}  (skipped {skipped})")
    if embs:
        out = ROOT / "data" / f"clip_ecapa_{args.split}.parquet"
        pq.write_table(
            pa.table({"audio_file": keys, "ecapa": [e.tolist() for e in embs]}),
            out, compression="zstd")
        print(f"wrote {out}  ({len(keys)} embeddings)")


if __name__ == "__main__":
    main()
