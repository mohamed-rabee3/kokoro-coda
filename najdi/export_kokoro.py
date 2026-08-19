#!/usr/bin/env python3
"""
StyleTTS2 training checkpoint -> Kokoro/KModel weights.

Inverse of najdi/convert_weights.py. Training checkpoints hold every module
(including the discriminators, text aligner and pitch extractor, which exist
only to train with); Kokoro's KModel wants exactly five. This pulls those out
so any epoch checkpoint can be run through the real inference path.

Note the `module.` prefix goes the other way here: accelerate/DataParallel may
have added it during training, and KModel does not expect it.

Usage:
    python najdi/export_kokoro.py --ckpt runs/stage1/epoch_1st_00001.pth \
                                  --out training/najdi_e1.pth
"""

import argparse
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
KEEP = ["bert", "bert_encoder", "predictor", "text_encoder", "decoder"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    net = ck.get("net", ck)
    print(f"checkpoint modules: {sorted(net)}")
    if "epoch" in ck:
        print(f"epoch: {ck['epoch']}  iters: {ck.get('iters')}")

    out = {}
    for name in KEEP:
        if name not in net:
            raise SystemExit(f"ERROR: {name} missing from checkpoint")
        sd = {k.replace("module.", ""): v for k, v in net[name].items()}
        out[name] = sd
        n = sum(v.numel() for v in sd.values() if hasattr(v, "numel"))
        print(f"  {name:14} {len(sd):4d} tensors  {n/1e6:7.2f}M")

    dest = Path(args.out)
    dest.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, dest)
    total = sum(sum(v.numel() for v in sd.values() if hasattr(v, "numel"))
                for sd in out.values())
    print(f"\nwrote {dest}  {total/1e6:.2f}M params  "
          f"{dest.stat().st_size/1e6:.1f} MB")


if __name__ == "__main__":
    main()
