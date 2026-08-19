#!/usr/bin/env python3
"""
Nabra-82M -> StyleTTS2 base checkpoint
======================================
StyleTTS2 expects {'net': {module_name: state_dict}}. Nabra ships the Kokoro
KModel layout instead, so we lift out only the five modules StyleTTS2 will load
and drop the `module.` prefix DataParallel left behind.

Everything StyleTTS2 builds but Kokoro does not have (diffusion, SLM
discriminator, style/predictor encoders, text aligner, pitch extractor) is
absent on purpose — `load_only_params: true` makes the load non-strict so those
stay at their fresh init. That is expected and is NOT the "weights didn't load"
failure; the thing to watch is Stage 2 mel starting ~0.43.

Usage:
    python najdi/convert_weights.py
"""

import os
import sys
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download

REPO_ID = "oddadmix/Nabra-82M-v0.1"
ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "training" / "nabra_base.pth"

# The five modules Kokoro/Nabra actually carries, in StyleTTS2's naming.
KEEP = ["bert", "bert_encoder", "predictor", "text_encoder", "decoder"]


def load_env_token() -> str | None:
    env = ROOT / ".env"
    if not env.exists():
        return None
    for line in env.read_text().splitlines():
        if line.strip().startswith("HF_TOKEN"):
            return line.split("=", 1)[1].strip().strip('"').strip("'")
    return None


def strip_prefix(state_dict):
    return {k.replace("module.", ""): v for k, v in state_dict.items()}


def main():
    token = os.environ.get("HF_TOKEN") or load_env_token()

    weights = hf_hub_download(REPO_ID, "kokoro_arabic.pth", token=token)
    config = hf_hub_download(REPO_ID, "config.json", token=token)
    voice = hf_hub_download(REPO_ID, "af_msa.pt", token=token)
    print(f"nabra weights : {weights}")
    print(f"nabra config  : {config}")
    print(f"nabra voice   : {voice}")

    raw = torch.load(weights, map_location="cpu", weights_only=False)
    print(f"\ntop-level keys in checkpoint: {sorted(raw.keys())}")

    missing = [k for k in KEEP if k not in raw]
    if missing:
        sys.exit(f"ERROR: Nabra checkpoint is missing expected modules: {missing}")

    net = {}
    for name in KEEP:
        sd = strip_prefix(raw[name])
        n_params = sum(v.numel() for v in sd.values() if hasattr(v, "numel"))
        net[name] = sd
        print(f"  {name:14} {len(sd):4d} tensors  {n_params/1e6:7.2f}M params")

    total = sum(
        sum(v.numel() for v in sd.values() if hasattr(v, "numel")) for sd in net.values()
    )
    print(f"  {'TOTAL':14} {total/1e6:7.2f}M params")

    # The Arabic adaptation lives in the text-encoder embedding: 178 rows, with
    # rows 7/8 (ʕ/ħ) trained by Nabra where stock Kokoro had them unused.
    emb = None
    for k, v in net["text_encoder"].items():
        if k.endswith("embedding.weight"):
            emb = (k, tuple(v.shape))
            break
    print(f"\ntext_encoder embedding: {emb}")
    if emb and emb[1][0] != 178:
        sys.exit(f"ERROR: expected a 178-row token embedding, got {emb[1]}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"net": net}, OUT)
    print(f"\nwrote {OUT}  ({OUT.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
