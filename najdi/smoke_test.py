#!/usr/bin/env python3
"""
Pre-flight checks before committing GPU-days (plan §12, gates 1 and 5).

Checks, in the order they can bite you:

  1. Symbol map is the Kokoro 178-token table with Nabra's ʕ/ħ at 7/8. Get this
     wrong and training runs perfectly while learning nonsense — it is the most
     common failure in this recipe.
  2. Band-limit patch is live (mel f_max and the discriminator BandLimiter).
  3. Nabra's weights actually land on the model. Missing diffusion/SLM keys are
     expected — Kokoro has no diffusion network — but a missing decoder or
     text_encoder key means the conversion broke.
  4. The text-encoder embedding really is 178x512 and rows 7/8 are trained.

Run from inside kikiri/StyleTTS2 so its local imports resolve.

Usage:
    cd kikiri/StyleTTS2 && python ../../najdi/smoke_test.py \
        --config ../../configs/config_najdi_stage1.yml
"""

import argparse
import sys
from pathlib import Path

import torch

# Same shim train_first.py/train_second.py install at import time. torch >= 2.6
# defaults weights_only=True, which rejects the StyleTTS2 Utils checkpoints
# (JDC/ASR/PLBERT). Without this the smoke test fails where training would not.
if getattr(torch, "_original_load", None) is None:
    torch._original_load = torch.load
    torch.load = lambda *args, **kwargs: torch._original_load(
        *args, **{**kwargs, "weights_only": False}
    )

import yaml  # noqa: E402


def check_symbols() -> bool:
    from kokoro_symbols import TextCleaner, dicts, symbols

    ok = True
    print("[1] symbol map")
    if len(symbols) != 178:
        print(f"    FAIL len(symbols)={len(symbols)}, expected 178")
        ok = False
    for sym, idx in [("ʕ", 7), ("ħ", 8), ("ɡ", 92), ("ː", 158), ("q", 59), (" ", 16)]:
        got = dicts.get(sym)
        flag = "ok " if got == idx else "FAIL"
        if got != idx:
            ok = False
        print(f"    {flag} {sym!r:5} -> {got} (want {idx})")
    ids = TextCleaner()("ɡˈaːla ħˈatta ʕˈan")
    print(f"    sample tokens: {ids[:12]}...")
    return ok


def check_band_limit() -> bool:
    import losses

    print("\n[2] band-limited-safe training")
    print(f"    BAND_LIMIT_HZ = {losses.BAND_LIMIT_HZ}")
    if not losses.BAND_LIMIT_HZ:
        print("    NOTE disabled — correct only for the §8 full-band fine-tune")
        return True
    mrl = losses.MultiResolutionSTFTLoss()
    fmaxes = {s.to_mel.f_max for s in mrl.stft_losses}
    print(f"    mel loss f_max = {fmaxes}")
    b = losses.BandLimiter(24000, losses.BAND_LIMIT_HZ)
    t = torch.arange(48000) / 24000.0

    def retained(freq):
        x = torch.sin(2 * torch.pi * freq * t).unsqueeze(0)
        return float(b(x).pow(2).mean() / x.pow(2).mean())

    keep, edge, kill = retained(4000), retained(7500), retained(10000)
    print(f"    retained: 4 kHz={keep:.3f}  7.5 kHz={edge:.3f}  10 kHz={kill:.3f}")
    # 7.5 kHz has to survive: that is real fricative energy, and torchaudio's
    # default filter width would drop it to ~0.2.
    return (fmaxes == {float(losses.BAND_LIMIT_HZ)}
            and keep > 0.95 and edge > 0.9 and kill < 1e-3)


def check_weights(cfg_path: str) -> bool:
    from models import build_model
    from Modules.diffusion.sampler import DiffusionSampler  # noqa: F401  (import graph)
    from munch import Munch
    from Utils.PLBERT.util import load_plbert
    from utils import recursive_munch

    print("\n[3] model build + Nabra weight load")
    cfg = yaml.safe_load(open(cfg_path))
    mp = recursive_munch(cfg["model_params"])

    from models import load_ASR_models, load_F0_models

    text_aligner = load_ASR_models(cfg["ASR_path"], cfg["ASR_config"])
    pitch_extractor = load_F0_models(cfg["F0_path"])
    plbert = load_plbert(cfg["PLBERT_dir"])
    model = build_model(mp, text_aligner, pitch_extractor, plbert)
    print(f"    modules built: {len(model)}")

    ckpt = torch.load(cfg["pretrained_model"], map_location="cpu", weights_only=False)
    net = ckpt["net"]
    print(f"    checkpoint modules: {sorted(net)}")

    ok = True
    for name in ["bert", "bert_encoder", "predictor", "text_encoder", "decoder"]:
        if name not in net:
            print(f"    FAIL {name} absent from checkpoint")
            ok = False
            continue
        res = model[name].load_state_dict(net[name], strict=False)
        miss, unexp = list(res.missing_keys), list(res.unexpected_keys)
        status = "ok " if not miss and not unexp else "WARN"
        print(f"    {status} {name:14} missing={len(miss)} unexpected={len(unexp)}")
        if miss:
            print(f"         first missing: {miss[:3]}")
            ok = False
        if unexp:
            print(f"         first unexpected: {unexp[:3]}")

    print("\n[4] text-encoder embedding")
    emb = net["text_encoder"]["embedding.weight"]
    print(f"    shape {tuple(emb.shape)} (want (178, 512))")
    norms = emb.norm(dim=1)
    med = float(norms.median())
    for i in (7, 8):
        r = float(norms[i]) / med
        flag = "ok " if r > 0.5 else "FAIL"
        if r <= 0.5:
            ok = False
        print(f"    {flag} row {i} norm {float(norms[i]):.2f} = {r:.2f}x median "
              f"-> Nabra {'trained' if r > 0.5 else 'did NOT train'} this slot")
    return ok and tuple(emb.shape) == (178, 512)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="../../configs/config_najdi_stage1.yml")
    args = ap.parse_args()

    results = [check_symbols(), check_band_limit(), check_weights(args.config)]
    print("\n" + "=" * 60)
    if all(results):
        print("SMOKE TEST: PASS — safe to start training")
        return 0
    print("SMOKE TEST: FAIL — fix before training")
    return 1


if __name__ == "__main__":
    sys.exit(main())
