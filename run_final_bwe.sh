#!/usr/bin/env bash
# Final phase (plan §8): full-band fine-tune on the bandwidth-extended core.
#
# All three env vars matter and are NOT interchangeable with the Stage 2 ones:
#   NAJDI_PE_FT=1          keeps predictor_encoder on ft_lr + weight decay. This
#                          is the fix for the norm explosion (1.66 -> 45 in 100
#                          steps) that destroyed every earlier adversarial run.
#                          Dropping it re-breaks everything.
#   NAJDI_BAND_LIMIT_HZ=0  band-limiting OFF. The training audio is genuinely
#                          full-band now, so protecting an empty band would
#                          actively prevent the decoder from learning it.
#   NAJDI_HB_LAMBDA=0      high-band distillation OFF, same reason — its whole
#                          purpose was supervising an unsupervised band.
#
# Adversarial is ON (joint_epoch: 0) because optimizer.step("decoder") is gated
# on it. That is required, not accidental — see the config header.
#
# Probe result at 400 steps (all four gates passed):
#   prosodic norm 1.316 (stable)      Gen Loss 9.16 -> 5.6 (descending)
#   pkmean_8_10k 22.2 -> 12.4         WER 0.061 (vs 0.063 before)
set -euo pipefail

ROOT=/home/ai2/kokoro-coda
CFG=$ROOT/configs/config_najdi_final_bwe.yml

CKPT=$(grep '^pretrained_model:' "$CFG" | awk '{print $2}')
echo "from      : $CKPT"
echo "data      : $(wc -l < $ROOT/data/bwe_train_list.txt) full-band clips"
echo "disk free : $(df -h "$ROOT" | awk 'NR==2{print $4}')"
[ -f "$CKPT" ] || { echo "ERROR: checkpoint missing"; exit 1; }

cd "$ROOT/kikiri/StyleTTS2"
setsid env \
  PYTHONPATH=.:$ROOT \
  NAJDI_PE_FT=1 \
  NAJDI_BAND_LIMIT_HZ=0 \
  NAJDI_HB_LAMBDA=0 \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  "$ROOT/.venv/bin/python" train_second.py --config_path "$CFG" \
  > "$ROOT/runs/final_bwe/train_stdout.log" 2>&1 < /dev/null &

echo "started pid $! (setsid — survives session teardown)"
echo
echo "verify within ~5 min:"
echo "  grep -iE 'fine-tune LR' $ROOT/runs/final_bwe/train.log     -> NAJDI_PE_FT active"
echo "  grep -E 'Step \\[' $ROOT/runs/final_bwe/train.log | tail -1 -> Disc/Gen NON-zero"
echo "  grep -E 'transcribe-back WER' $ROOT/runs/final_bwe/train.log -> gate, ~0.06 healthy"
