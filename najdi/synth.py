#!/usr/bin/env python3
"""
Synthesize with Nabra or a fine-tuned checkpoint.

Doubles as the inference-path smoke test and as the baseline generator for
evaluation. Text goes through the *same* Najdi front-end used for training
(najdi/g2p.py), so train and inference share a symbol set exactly — that
identity is the whole reason the front-end is a single module.

Usage:
    # Nabra baseline
    python najdi/synth.py --out-dir eval/nabra

    # our fine-tune
    python najdi/synth.py --model runs/stage2/epoch_2nd_00009.pth \
                          --voice voices/sf_najdi.pt --out-dir eval/najdi

    # MSA front-end instead of the Najdi rules, for the A/B
    python najdi/synth.py --no-dialect --out-dir eval/nabra_msa
"""

import argparse
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

ROOT = Path(__file__).resolve().parent.parent
REPO_ID = "oddadmix/Nabra-82M-v0.1"

# Najdi test sentences. Deliberately loaded with ق so the q->ɡ rule is audible,
# plus the exception lexicon (القرآن) which must stay /q/.
SENTENCES = [
    "حَيَّاك اللَّه، وِشْ أَخْبَارِك؟",
    "قَالَ لِي أَبُوي نَرُوح الْيَوْم",
    "وِشْ رَايِك نَقْعُد فِي الْقَهْوَة شْوَي؟",
    "أَنَا قُلْت لَك قَبْل، بَس مَا سَمِعْت",
    "الْقُرْآن الْكَرِيم",
    "الْحِين وَقْت الْغَدَا، تَعَال",
    "مَا عِنْدِي وَقْت الْيَوْم، بَاجِر إِن شَاء اللَّه",
]


def token() -> str | None:
    env = ROOT / ".env"
    if not env.exists():
        return None
    for line in env.read_text().splitlines():
        if line.strip().startswith("HF_TOKEN"):
            return line.split("=", 1)[1].strip().strip('"').strip("'")
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None, help="checkpoint (default: Nabra)")
    ap.add_argument("--voice", default=None, help="voicepack .pt (default: af_msa)")
    ap.add_argument(
        "--allow-base-voice",
        action="store_true",
        help="permit Nabra's af_msa voicepack with our own checkpoint",
    )
    ap.add_argument("--out-dir", default="eval/nabra")
    ap.add_argument(
        "--sentences",
        default=None,
        help="category<TAB>text TSV (e.g. najdi/eval_sentences.tsv). "
        "Default is the 7-sentence smoke set below.",
    )
    ap.add_argument("--no-dialect", action="store_true")
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    from huggingface_hub import hf_hub_download
    from kokoro import KModel

    from najdi.g2p import NajdiG2P

    # Nabra's af_msa voicepack is only the right input for Nabra's decoder.
    # Once Stage 2 passes joint_epoch, decoder and style_encoder co-adapt to our
    # style distribution (norms ~6.7 acoustic / ~35.8 prosodic against Nabra's
    # ~0.35), and af_msa becomes out-of-distribution — a quality measurement of
    # our voice that is silently mostly measuring the mismatch.
    if args.model and not args.voice and not args.allow_base_voice:
        raise SystemExit(
            "refusing to synthesize a custom checkpoint with Nabra's af_msa voicepack.\n"
            "  pass --voice <voicepack.pt> matched to this checkpoint, e.g.\n"
            "    python kikiri/scripts/extract_voicepack.py --model <ckpt> \\\n"
            "        --style-encoder-model runs/stage1/first_stage.pth \\\n"
            "        --audio-dir data/voicecore_sample --output <voicepack.pt>\n"
            "  or pass --allow-base-voice if the mismatch is deliberate."
        )

    tok = token()
    config = hf_hub_download(REPO_ID, "config.json", token=tok)
    model_path = args.model or hf_hub_download(REPO_ID, "kokoro_arabic.pth", token=tok)
    voice_path = args.voice or hf_hub_download(REPO_ID, "af_msa.pt", token=tok)

    # disable_complex=True selects the real-valued STFT path, which Nabra's card
    # recommends as the portable one.
    model = KModel(repo_id=REPO_ID, config=config, model=model_path,
                   disable_complex=True).eval().to(args.device)
    model.vocab.update({"ʕ": 7, "ħ": 8})

    voice = torch.load(voice_path, map_location=args.device, weights_only=True)
    g2p = NajdiG2P(dialect=not args.no_dialect)

    if args.sentences:
        sents, cats = [], []
        for line in Path(args.sentences).read_text(encoding="utf-8").splitlines():
            if line.startswith("#") or "\t" not in line:
                continue
            cat, text = line.split("\t", 1)
            cats.append(cat)
            sents.append(text)
    else:
        sents, cats = list(SENTENCES), ["smoke"] * len(SENTENCES)

    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"model  : {model_path}")
    print(f"voice  : {voice_path}  shape={tuple(voice.shape)}")
    print(f"front-end: {'Najdi' if not args.no_dialect else 'MSA'}")
    print(f"sentences: {len(sents)}\n")

    rtfs = []
    for i, text in enumerate(sents):
        ph = g2p.phonemize(text)
        # KModel.forward takes the phoneme STRING and tokenizes internally.
        # Handing it token ids silently yields an empty sequence and ~0.25 s of
        # audio for every input, which looks like a model failure but is not.
        n_tok = len([c for c in ph if model.vocab.get(c) is not None])
        if n_tok + 2 > 510:
            print(f"  [{i}] SKIP: {n_tok} tokens > 510")
            continue

        # Voicepack row is indexed by len(phonemes)-1, matching KPipeline.
        ref_s = voice[len(ph) - 1]

        t0 = time.time()
        with torch.no_grad():
            out = model(ph, ref_s, speed=args.speed, return_output=True)
        el = time.time() - t0

        audio = out.audio.detach().cpu().numpy().astype(np.float32)
        dur = len(audio) / 24000
        rtf = el / max(dur, 1e-6)
        rtfs.append(rtf)

        path = out_dir / f"{i:02d}.wav"
        sf.write(path, audio, 24000)
        print(f"  [{i}] {dur:5.2f}s  rtf={rtf:.3f}  {n_tok:3d} tok  {text}")
        print(f"       {ph}")

    # Write the sentences ACTUALLY synthesized, not the module-level default —
    # wer_eval.py pairs sentences.txt line i with {i:02d}.wav, so a stale
    # manifest silently scores every clip against the wrong reference.
    (out_dir / "sentences.txt").write_text("\n".join(sents), encoding="utf-8")
    (out_dir / "categories.txt").write_text("\n".join(cats), encoding="utf-8")
    print(f"\nwrote {len(rtfs)} wavs to {out_dir}")
    if rtfs:
        print(f"median RTF ({args.device}): {float(np.median(rtfs)):.3f}")


if __name__ == "__main__":
    main()
