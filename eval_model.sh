export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# export HF_ENDPOINT=https://hf-mirror.com
export HF_DATASETS_CACHE=/group/40143///huggingface/datasets
# PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
NUM_GPUS=8
MODEL=Qwen/Qwen2.5-7B-Instruct
MODEL_ARGS="pretrained=$MODEL,dtype=bfloat16,data_parallel_size=$NUM_GPUS,max_model_length=32768,gpu_memory_utilization=0.8,generation_parameters={max_new_tokens:32768,temperature:0.6,top_p:0.95}"
TASK="aime24"
OUTPUT_DIR=data/evals/$MODEL

lighteval vllm \
  "model_name=deepseek-ai/DeepSeek-R1-Distill-Qwen-7B,dtype=float16,tensor_parallel_size=4,max_model_length=2048" \
  "lighteval|$TASK|0|0" \
  --output-dir "$OUTPUT_DIR"