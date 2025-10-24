#!/bin/bash 
set -x
# "original", "pq_search", "sparq_f", "infllm", "h2o"
SEED="4321"
COMPRESSOR="pq_search" 
EXP_NAME=default
MODE="off" # profile or off 
DEVICE=0 # GPU index.
CORE_OFFSET=0 # 64
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

MAX_CPU_IN_USE=48
DROP=0
RECENT_SIZE=32
PRESERVE_LAYER=0
THRESHOLD=100000
SCORE_FUNC="sum" # sum, max
KEYFORMER_MODE=0
USE_LINGUA=0

export CORE_OFFSET=${CORE_OFFSET}

MAX_CPU_IN_USE=${MAX_CPU_IN_USE} \
RANDOM_SEED=${SEED} \
PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:64" \
CUDA_VISIBLE_DEVICES=${DEVICE} \
TOKENIZERS_PARALLELISM=false \
SUBVEC=${SUBVEC} SUBBITS=${SUBBITS} \
METRIC=${METRIC} \
python vq_pred.py \
    --model  mistral-7b-Instruct-32k \
    --compress_ratio ${COMPRESS} \
    --important_ratio ${TOPK} \
    --recent_ratio ${RECENT_RATIO} \
    --recent_size ${RECENT_SIZE} \
    --drop_ratio ${DROP} \
    --enable_vq_cache \
    --fp16 \
    --pp-size 1 \
    --sink-size ${SINK_SIZE} \
    --exp_name ${EXP_NAME} \
    --score_func ${SCORE_FUNC} \
    --preserve_layer ${PRESERVE_LAYER} \
    --keyformer_mode ${KEYFORMER_MODE} \
    --compressor ${COMPRESSOR} \
    --threshold ${THRESHOLD} \
    --n_subvec_per_head ${SUBVEC} \
    --n_subbits ${SUBBITS} \
    --topr ${TOPR} \
    --gqa ${GQA} \
    --sparq_mean_v_trick ${MEAN_V_TRICK} \
    --max_iter ${MAX_ITER}
