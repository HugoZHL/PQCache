#!/bin/bash 
set -x
COMPRESSOR="pq_search" 
EXP_NAME="ultimate_recheck_wo_cache"
DEVICE=1
COMPRESS=0.2
TOPK=0.5
RECENT_RATIO=0.5
SINK_SIZE=32
SUBVEC=2
SUBBITS=6
TOPR=1
METRIC="euc" # euc ip
GQA="True" # True False
MEAN_V_TRICK="False"
MAX_ITER=0 # 0 for dynamic setting

CUDA_VISIBLE_DEVICES=${DEVICE} \
SUBVEC=${SUBVEC} \
SUBBITS=${SUBBITS} \
METRIC=${METRIC} \
python vq_pred.py \
    --model mistral-7b-Instruct-32k \
    --compress_ratio ${COMPRESS} \
    --important_ratio ${TOPK} \
    --recent_ratio ${RECENT_RATIO} \
    --enable_vq_cache \
    --fp16 \
    --pp-size 1 \
    --sink-size ${SINK_SIZE} \
    --exp_name ${EXP_NAME} \
    --compressor ${COMPRESSOR} \
    --n_subvec_per_head ${SUBVEC} \
    --n_subbits ${SUBBITS} \
    --topr ${TOPR} \
    --gqa ${GQA} \
    --sparq_mean_v_trick ${MEAN_V_TRICK} \
    --max_iter ${MAX_ITER}