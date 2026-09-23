#!/bin/bash
set -e

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export OMP_NUM_THREADS=8
export NCCL_DEBUG=WARN
export NCCL_IB_DISABLE=1
export PYTHONPATH=$(pwd):${PYTHONPATH}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

MASTER_PORT=29688
NPROC=8

DATA_ROOT="xxx"

EXP_STAGE1="xxx"
EXP_STAGE2="xxx"

STAGE1_CKPT="./logs/${EXP_STAGE1}/checkpoints/epoch_20.pt"

CONCEPT_FREQ="./Concept-Sample/concept_frequency.json"
CONCEPT_VOCAB="./Concept-Sample/concept_vocab.json"


torchrun \
  --nproc_per_node=${NPROC} \
  --master_port=${MASTER_PORT} \
  -- \
  training/main.py \
  --model coca_ViT-B-32 \
  --name ${EXP_STAGE2} \
  --train-stage stage2 \
  --resume ${STAGE1_CKPT} \
  --workers 4 \
  --batch-size 24 \
  --epochs 30 \
  --lr 1e-4 \
  --wd 0.01 \
  --warmup 1000 \
  --grad-clip-norm 0.5 \
  --save-frequency 2 \
  --save-most-recent \
  --report-to none \
  --precision amp \
  --logs ./logs \
  --mimic-iv-ecg-path ${DATA_ROOT}/mimic-iv-ecg \
  --ptbxl-path ${DATA_ROOT}/ptb-xl \
  --champan-path ${DATA_ROOT}/champan-shaoxing \
  --sph-path ${DATA_ROOT}/sph \
  --use-ccar \
  --concept-frequency-path ${CONCEPT_FREQ} \
  --concept-vocab-path ${CONCEPT_VOCAB} \
  --ccar-alpha 0.75 \
  --ccar-eps 1e-3 \
  --ccar-wmin 1.0 \
  --ccar-wmax 2.0 \
  --ccar-default-weight 1.0 \
  --use-dcm \
  --dcm-dropout 0.1