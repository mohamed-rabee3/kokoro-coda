#!/usr/bin/env bash
# A/B probe: what destabilised Stage 2's adversarial phase?
#
# Adversarial training took the model from WER 0.028 to 4.47 inside one epoch.
# Two things in that run were mine and not in the reference recipe:
#
#   band-limit  lowpass both real and generated audio at 8 kHz before MPD/MSD,
#               so the discriminators never see the empty >8 kHz band
#   hb-distill  L1 against a frozen Nabra decoder above 8 kHz
#
# hb-distill is suspect on its face: its target is ref_decoder(en, F0, N, s) fed
# OUR style vectors, whose norms had grown to ~6.4 against Nabra's ~0.35. That is
# far out of distribution for the frozen decoder, so the distillation target may
# have been garbage we then trained toward.
#
# Each variant restarts from the last intact checkpoint, runs the same number of
# adversarial steps, and is scored identically (WER + spectral centroid).
# Baseline for comparison: WER 0.028, centroid 412 Hz.
set -uo pipefail

ROOT=/home/ai2/kokoro-coda
STEPS=${STEPS:-400}
CFG=$ROOT/configs/config_najdi_probe.yml
SP=/tmp/claude-1004/-home-ai2-kokoro-coda/ab7dc9c9-1b1b-4451-bd82-13592d4a31ca/scratchpad

run () {                       # name, band_limit_hz, hb_lambda
  local name=$1 band=$2 hb=$3
  echo "=== $name  (band=$band hb=$hb, $STEPS steps) ==="
  rm -f "$ROOT/runs/probe/$name.pth"
  ( cd "$ROOT/kikiri/StyleTTS2" && \
    PYTHONPATH=.:$ROOT \
    NAJDI_BAND_LIMIT_HZ=$band \
    NAJDI_HB_LAMBDA=$hb \
    NAJDI_PROBE_STEPS=$STEPS \
    NAJDI_PROBE_OUT=$name.pth \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    "$ROOT/.venv/bin/python" train_second.py --config_path "$CFG" \
      > "$ROOT/runs/probe/$name.log" 2>&1 )

  if [ ! -f "$ROOT/runs/probe/$name.pth" ]; then
    echo "  FAILED to produce snapshot; tail:"; tail -5 "$ROOT/runs/probe/$name.log"; return
  fi
  PYTHONPATH=$ROOT "$ROOT/.venv/bin/python" "$ROOT/najdi/export_kokoro.py" \
    --ckpt "$ROOT/runs/probe/$name.pth" --out "$SP/$name.pth" >/dev/null 2>&1
  PYTHONPATH=$ROOT "$ROOT/.venv/bin/python" "$ROOT/najdi/synth.py" \
    --model "$SP/$name.pth" --out-dir "eval/probe_$name" >/dev/null 2>&1
  echo -n "  losses at end: "
  grep -E "Step \[" "$ROOT/runs/probe/$name.log" | tail -1 | \
    sed -E 's/.*(Loss: [0-9.]+).*(Disc Loss: [0-9.]+).*(Gen Loss: [0-9.]+).*/\1  \2  \3/'
}

run stock      0    0      # neither patch — reference StyleTTS2 behaviour
run bandonly   8000 0      # band-limited discriminators only
run hbonly     0    0.5    # high-band distillation only
run both       8000 0.5    # what actually ran

echo
echo "=== scoring (baseline: WER 0.028, centroid 412 Hz) ==="
PYTHONPATH=$ROOT "$ROOT/.venv/bin/python" "$ROOT/najdi/wer_eval.py" \
  --dirs eval/probe_stock eval/probe_bandonly eval/probe_hbonly eval/probe_both 2>&1 \
  | grep -E "^eval/|^dir"
for n in stock bandonly hbonly both; do
  echo -n "probe_$n centroid: "
  PYTHONPATH=$ROOT "$ROOT/.venv/bin/python" "$ROOT/najdi/band_check.py" \
    --wav-dir "eval/probe_$n" --ref-dir eval/nabra 2>&1 | grep centroid_hz
done
