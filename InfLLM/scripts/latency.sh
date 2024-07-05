set -x
config=config/llama-2-inf-llm.yaml
exp_name=${1}

datasets="narrativeqa,qasper,multifieldqa_en,\
hotpotqa,2wikimqa,musique,\
gov_report,qmsum,multi_news,\
trec,triviaqa,samsum,\
passage_count,passage_retrieval_en,\
lcc,repobench-p"

TOKENIZERS_PARALLELISM=false \
CUDA_VISIBLE_DEVICES=0 \
COMPRESS_RATIO=0.2 \
LOCAL_RATIO=0.5 \
nsys profile -c cudaProfilerApi \
python benchmark/test_latency.py \
--config_path ${config} \
--output_dir_path benchmark/${exp_name} \
--datasets ${datasets} 
# TOKENIZERS_PARALLELISM=false \
# CUDA_VISIBLE_DEVICES=0 \
# COMPRESS_RATIO=0.2 \
# LOCAL_RATIO=0.5 \
# python benchmark/test_latency.py \
# --config_path ${config} \
# --output_dir_path benchmark/${exp_name} \
# --datasets ${datasets} 