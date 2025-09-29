# Policy Optimization with Experience Replay: Guiding Reasoning Models to Complete the Reasoning Path

To our knowledge, all existing reinforcement fine-tuning algorithms in the field of large language models require generating a complete reasoning process starting from the question.
Challenging this convention, we propose the assumption that "During reinforcement fine-tuning, the model only needs to generate part of the reasoning process." We analyze the impact of different segments of the reasoning path on the correctness of the final result, and based on these insights, we introduce a reinforcement fine-tuning algorithm named Policy Optimization with Experience Replay (POER). Unlike traditional reinforcement fine-tuning algorithms that generate full reasoning paths, POER trains the model by generating suffixes of the reasoning path using experience caching, significantly reducing training time meanwhile improving training stability. Experiments on mathematical reasoning benchmarks show that POER improves accuracy by 2.2\% for the 1.5B model and 3.4\% for the 7B model compared to full-path reinforcement fine-tuning algorithms, while reducing training time by 62.4\% for the 1.5B model and 65.9\% for the 7B model.This work demonstrates how experience replay techniques can significantly reduce the reasoning overhead of large language models, offering a plug-and-play, resource-efficient alternative to traditional approaches.

## Resources

### Models
- [POER-7b](https://huggingface.co/anonymity0516/POER-7b)
- [POER-1.5b](https://huggingface.co/anonymity0516/POER-1.5b)



## Installation

### Prerequisites
Install `uv` for managing virtual environments:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
```

Set up a virtual environment with Python 3.11:
```bash
uv venv openr1 --python 3.11
source openr1/bin/activate
uv pip install --upgrade pip
export UV_LINK_MODE=copy
```

### Dependencies
Install `vLLM` and `FlashAttention`:
```bash
uv pip install vllm==0.7.2
uv pip install setuptools
uv pip install flash-attn --no-build-isolation
uv pip install trl==0.14.0
```

> **Note**: This installs PyTorch `v2.5.1`, which is required for `vLLM` compatibility. Using a different version may cause issues.

Install additional dependencies based on your use case:
```bash
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e ".[dev]"
```

### Authentication
Log in to Hugging Face and Weights & Biases:
```bash
huggingface-cli login
wandb login
```

### Git LFS
Ensure Git LFS is installed for model/dataset management:
```bash
git-lfs --version
```
If not installed:
```bash
sudo apt-get install git-lfs
```

## Training

Train models using a YAML config with 8 GPUs (set `num_processes=7`):
```bash
sh run_grpo_cache.sh
```

## Evaluation

Evaluate models using `lighteval` with custom tasks in `src/open_r1/evaluate.py`. For single-GPU setups:
```bash
MODEL=model
MODEL_ARGS="pretrained=$MODEL,dtype=bfloat16,max_model_length=32768,gpu_memory_utilization=0.8,generation_parameters={max_new_tokens:32768,temperature:0.6,top_p:0.95}"
OUTPUT_DIR=data/evals/$MODEL

# Example: AIME 2024
TASK=aime24
lighteval vllm "$MODEL_ARGS" "custom|$TASK|0|0" \
  --custom-tasks src/open_r1/evaluate.py \
  --use-chat-template \
  --output-dir "$OUTPUT_DIR"
```

> **Important**: Set `max_model_length=32768` to match `max_new_tokens`, or `lighteval` will fail.

For multi-GPU evaluation with data parallelism:
```bash
NUM_GPUS=4
MODEL=model
MODEL_ARGS="pretrained=$MODEL,dtype=bfloat16,data_parallel_size=$NUM_GPUS,max_model_length=32768,gpu_memory_utilization=0.8,generation_parameters={max_new_tokens:32768,temperature:0.6,top_p:0.95}"
TASK=aime24
OUTPUT_DIR=data/evals/$MODEL

lighteval vllm "$MODEL_ARGS" "custom|$TASK|0|0" \
  --custom-tasks src/open_r1/evaluate.py \
  --use-chat-template \
  --output-dir "$OUTPUT_DIR"
```

Alternatively, use the evaluation script:
```bash
sh eval_model.sh
```
Modify tasks in `eval_model.sh` (line 8) as needed.


