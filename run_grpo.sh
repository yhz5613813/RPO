unset LD_LIBRARY_PATH
# export HF_ENDPOINT=https://hf-mirror.com
# export TRANSFORMERS_OFFLINE=1
# export HF_DATASETS_OFFLINE=1
export WANDB_MODE="offline"
export VLLM_WORKER_MULTIPROC_METHOD="spawn"
export VLLM_LOGGING_LEVEL=DEBUG
# export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd src
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 ACCELERATE_LOG_LEVEL=info accelerate launch --config_file recipes/accelerate_configs/zero2.yaml \
    --num_processes=7 open_r1/grpo.py \
    --config poer/recipes/grpo_7b.yaml