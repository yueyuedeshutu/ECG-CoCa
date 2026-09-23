#!/bin/bash
set -e

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export OMP_NUM_THREADS=8
export NCCL_DEBUG=WARN
export NCCL_IB_DISABLE=1
export PYTHONPATH=$(pwd):${PYTHONPATH}

MASTER_PORT=29688
NPROC=8

DATA_ROOT="xxx"
EXP_STAGE1="xx"

torchrun \
  --nproc_per_node=${NPROC} \
  --master_port=${MASTER_PORT} \
  -- \
  training/main.py \
  --model coca_ViT-B-32 \
  --name ${EXP_STAGE1} \
  --train-stage stage1 \
  --resume latest \
  --workers 4 \
  --batch-size 72 \
  --epochs 30 \
  --lr 1e-4 \
  --wd 0.05 \
  --warmup 100 \
  --grad-clip-norm 0.5 \
  --save-frequency 5 \
  --save-most-recent \
  --report-to none \
  --precision amp \
  --val-frequency 5 \
  --logs ./logs \
  --mimic-iv-ecg-path ${DATA_ROOT}/mimic-iv-ecg \
  --ptbxl-path ${DATA_ROOT}/ptb-xl \
  --champan-path ${DATA_ROOT}/champan-shaoxing \
  --sph-path ${DATA_ROOT}/sph \
  --cpsc2018-path ${DATA_ROOT}/CPSC2018