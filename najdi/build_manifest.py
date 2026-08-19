#!/usr/bin/env python3
"""
Corpus metadata manifest — everything except the audio.

The `audio` column is ~99% of the 43.6 GB. Parquet column projection over
HfFileSystem range reads pulls only the columns we ask for, so the whole
metadata pass costs a few hundred MB instead of a full download. That matters:
we have 75 GB of disk and a 43.6 GB dataset, so the filter decisions have to be
made *before* anything is materialized.

Output: data/manifest_raw.parquet — one row per clip, all quality signals and
all three text variants, ready for the gender gate and the ق-mark recovery.

Usage:
    python najdi/build_manifest.py [--workers 12] [--split train]
"""

import argparse
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import HfFileSystem

REPO = "Eman-Fouda/youtube-001-part02-tashkeel"
ROOT = Path(__file__).resolve().parent.parent

# Everything but `audio`. Keeping all three text variants here means the ق-mark
# recovery (plan §3.5) never has to touch the remote files again.
COLS = [
    "audio_file", "video_id", "speaker", "shard",
    "start", "end", "duration",
    "text", "text_tashkeel", "text_tashkeel_raw", "text_cohere",
    "coverage", "n_words", "n_marked",
    "cer_yt_vs_cohere", "squim_mos", "snr_db", "music_score", "boundary_clipped",
]


def token() -> str:
    for line in (ROOT / ".env").read_text().splitlines():
        if line.strip().startswith("HF_TOKEN"):
            return line.split("=", 1)[1].strip().strip('"').strip("'")
    raise SystemExit("HF_TOKEN not found in .env")


def read_shard(fs: HfFileSystem, path: str) -> pa.Table:
    with fs.open(path, "rb") as fh:
        pf = pq.ParquetFile(fh)
        have = set(pf.schema_arrow.names)
        return pf.read(columns=[c for c in COLS if c in have])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--split", default="train", choices=["train", "val", "long"])
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    fs = HfFileSystem(token=token())
    prefix = f"datasets/{REPO}/data"
    files = sorted(f for f in fs.ls(prefix, detail=False) if f"/{args.split}-" in f)
    print(f"{args.split}: {len(files)} shards")

    out = Path(args.out or ROOT / "data" / f"manifest_{args.split}.parquet")
    out.parent.mkdir(parents=True, exist_ok=True)

    tables, done, t0 = [], 0, time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(read_shard, fs, f): f for f in files}
        for fut in as_completed(futs):
            f = futs[fut]
            try:
                tables.append(fut.result())
            except Exception as e:
                print(f"  FAILED {f}: {e}")
                continue
            done += 1
            el = time.time() - t0
            rate = done / el
            print(f"  [{done:3d}/{len(files)}] {el:6.1f}s  "
                  f"eta {(len(files)-done)/max(rate,1e-9):6.1f}s", flush=True)

    tbl = pa.concat_tables(tables, promote_options="default")
    pq.write_table(tbl, out, compression="zstd")
    print(f"\nwrote {out}  rows={tbl.num_rows}  {out.stat().st_size/1e6:.1f} MB")

    d = tbl.to_pydict()
    hours = sum(x for x in d["duration"] if x) / 3600
    print(f"total audio: {hours:.1f} h across {len(set(d['speaker']))} speakers "
          f"/ {len(set(d['video_id']))} videos")


if __name__ == "__main__":
    main()
