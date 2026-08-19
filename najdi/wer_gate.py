#!/usr/bin/env python3
"""
Per-epoch transcribe-back gate for training (plan §12, added after the fact).

Why this exists
---------------
Stage 2's adversarial phase destroyed this model twice, and BOTH times every
training loss looked healthy while it happened:

    six-day run   eval mel 0.751 and "improving"; output was pure noise
    400-step probe  mel 1.20 -> 0.58, dur 3.26 -> 0.91, F0 23.1 -> 5.9, no NaN
                    -> WER 0.028 -> 1.028, every sentence a Whisper hallucination

Reconstruction loss is computed against a style vector produced by the *live*
encoders, so the decoder and the encoders can co-drift and keep agreeing with
each other while the exported model produces noise. The loss cannot see that.
Transcribe-back can: it runs the real inference path end to end.

So this is not a metric, it is a circuit breaker. Nothing long-running should
train without it.

Runs on CPU by default and deliberately so — a GPU ASR model would contend with
training for memory and can OOM the run it is supposed to protect. Cost is a
few minutes per epoch against multi-hour epochs.
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from najdi.wer_eval import edit, norm  # same normalization as the offline eval

_ASR = None


def _asr():
    """Load the ASR pipeline once, lazily (it is ~3 GB and not always needed)."""
    global _ASR
    if _ASR is None:
        import torch
        from transformers import pipeline

        model = os.environ.get("NAJDI_WER_MODEL", "openai/whisper-large-v3")
        dev = os.environ.get("NAJDI_WER_DEVICE", "cpu")
        _ASR = pipeline(
            "automatic-speech-recognition",
            model=model,
            device=0 if dev == "cuda" else -1,
            torch_dtype=torch.float16 if dev == "cuda" else torch.float32,
        )
    return _ASR


def score(pairs, sr=24000):
    """
    pairs: [(reference_text, audio_float32_mono_ndarray), ...]
    returns (wer, cer, details)
    """
    import numpy as np

    asr = _asr()
    we = wn = ce = cn = 0
    details = []
    for text, audio in pairs:
        audio = np.asarray(audio, dtype=np.float32).squeeze()
        hyp = asr(
            {"raw": audio, "sampling_rate": int(sr)},
            generate_kwargs={"language": "arabic"},
        )["text"]
        r, h = norm(text), norm(hyp)
        rw, hw = r.split(), h.split()
        e = edit(rw, hw)
        we += e
        wn += len(rw)
        ce += edit(list(r), list(h))
        cn += len(r)
        details.append((text, hyp.strip(), e / max(len(rw), 1)))
    return we / max(wn, 1), ce / max(cn, 1), details


def enabled() -> bool:
    return os.environ.get("NAJDI_WER_GATE", "1") not in ("0", "", "false")


def abort_threshold() -> float:
    """
    Default 0.35. Healthy is ~0.03 and total collapse reads ~1.0, so 0.35 is far
    above normal epoch-to-epoch noise and far below the failure. It is a
    collapse detector, not a quality bar — do not tighten it to chase small
    regressions or it will kill runs that were merely having a bad epoch.
    """
    return float(os.environ.get("NAJDI_WER_ABORT", "0.35"))
