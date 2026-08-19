#!/usr/bin/env bash
# Resume Stage 2 from the last checkpoint.
#
# Both env vars are REQUIRED, not optional:
#   NAJDI_BAND_LIMIT_HZ=8000  band-limited-safe training. The corpus is 16 kHz
#                             upsampled to 24 kHz, so without this the mel loss
#                             and the MPD/MSD discriminators both read the empty
#                             8-12 kHz band as a target and the voice goes muffled.
#   NAJDI_HB_LAMBDA=0.5       high-band self-distillation against a frozen Nabra
#                             decoder. Without it the decoder drifts tonally in
#                             that same unsupervised band (measured: 9-10 kHz
#                             peak/mean 12.4x vs Nabra's 3.1x after Stage 1).
#
# Dropping either one silently degrades the audio rather than erroring, so keep
# them together with the launch.
#
# To resume from a different checkpoint, edit `pretrained_model` in
# configs/config_najdi_stage2_resume.yml.
set -euo pipefail

ROOT=/home/ai2/kokoro-coda
CFG=$ROOT/configs/config_najdi_stage2_resume.yml

CKPT=$(grep '^pretrained_model:' "$CFG" | awk '{print $2}')
echo "resuming from: $CKPT"
[ -f "$CKPT" ] || { echo "ERROR: checkpoint missing"; exit 1; }

cd "$ROOT/kikiri/StyleTTS2"
# setsid, not just nohup. nohup only ignores SIGHUP; it does NOT protect against
# the parent's whole process group/session being killed, which is what happens
# when the launching agent session is torn down. That killed a run mid-epoch
# twice (2026-08-13 14:09, ~4.7 h of epoch 7 lost). setsid puts training in its
# own session so it outlives whatever started it.
setsid env \
  PYTHONPATH=.:$ROOT \
  NAJDI_BAND_LIMIT_HZ=8000 \
  NAJDI_HB_LAMBDA=0.5 \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  "$ROOT/.venv/bin/python" train_second.py --config_path "$CFG" \
  > "$ROOT/runs/stage2_resume.log" 2>&1 < /dev/null &

echo "started pid $! (detached via setsid — survives session teardown)"
echo "  stdout : $ROOT/runs/stage2_resume.log"
echo "  losses : $ROOT/runs/stage2/train.log"
echo
echo "sanity-check within ~5 min:"
echo "  grep -iE 'predictor_encoder|high-band' $ROOT/runs/stage2_resume.log"
echo "    -> want 'predictor_encoder already trained - keeping it (resume)'"
echo "    -> want 'high-band distillation ON: lambda=0.5'"
echo "  tail -1 $ROOT/runs/stage2/train.log"
echo "    -> Disc Loss and Gen Loss must be NON-zero (adversarial active)"
