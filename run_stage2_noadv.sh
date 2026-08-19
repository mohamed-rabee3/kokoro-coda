#!/usr/bin/env bash
# Stage 2, prosody only — the restart after the adversarial phase was proven
# to destroy this model (see configs/config_najdi_stage2_noadv.yml for the
# four-way measurement).
#
# The band-limit and high-band-distillation env vars are deliberately NOT set
# here. Both only bind on the decoder path, which joint_epoch: 999 never
# trains, so setting them would imply a protection that is not doing anything.
# They come back in plan §8, on full-band audio, where they matter.
#
# NAJDI_WER_GATE is left at its default (on). Do not switch it off for a long
# run: the losses stayed healthy through two separate collapses, and this is
# the only signal that caught either one.
set -euo pipefail

ROOT=/home/ai2/kokoro-coda
CFG=$ROOT/configs/config_najdi_stage2_noadv.yml

CKPT=$(grep '^pretrained_model:' "$CFG" | awk '{print $2}')
echo "resuming from : $CKPT"
[ -f "$CKPT" ] || { echo "ERROR: checkpoint missing"; exit 1; }
echo "disk free     : $(df -h "$ROOT" | awk 'NR==2{print $4}')"

cd "$ROOT/kikiri/StyleTTS2"
# setsid, not nohup. nohup only ignores SIGHUP; it does not survive the
# launching session's process group being torn down, which killed a run
# mid-epoch twice (~4.7 h of epoch 7 lost on 2026-08-13).
setsid env \
  PYTHONPATH=.:$ROOT \
  NAJDI_PE_FT=1 \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  "$ROOT/.venv/bin/python" train_second.py --config_path "$CFG" \
  > "$ROOT/runs/stage2_noadv.log" 2>&1 < /dev/null &

echo "started pid $! (detached via setsid — survives session teardown)"
echo
echo "verify within ~10 min:"
echo "  grep -i predictor_encoder $ROOT/runs/stage2_noadv.log"
echo "    -> want 'already trained — keeping it (resume)'"
echo "  grep -E 'Step \\[' $ROOT/runs/stage2_noadv.log | tail -1"
echo "    -> Disc Loss and Gen Loss must both be 0.00000 (adversarial OFF)"
echo
echo "then at each epoch boundary:"
echo "  grep -E 'transcribe-back WER' $ROOT/runs/stage2_noadv/train.log"
echo "    -> healthy is ~0.03; the run self-aborts above 0.35"
echo "  ls -la $ROOT/runs/stage2_noadv/epoch_2nd_best.pth"
