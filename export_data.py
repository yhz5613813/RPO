import multiprocessing

import datasets
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer
import torch
from vllm import LLM, SamplingParams
from trl.data_utils import apply_chat_template, is_conversational, maybe_apply_chat_template
# from trl.trainer.utils import pad
from torch.nn.utils.rnn import pad_sequence
import os
from tqdm import tqdm
import time
import hashlib
import os
# os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
# os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3,4,5,6,7"
# 加载原始数据集
data = datasets.load_dataset("knoveleng/open-rs")["train"]
model_name = "/group/40143///model/DeepSeek-R1-Distill-Qwen-1.5B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
MAX_TOKENS = 2048
batch_size = 256
device = "cuda:0"

llm = LLM(
    model=model_name,
    tokenizer=model_name,
    tensor_parallel_size=4,
    gpu_memory_utilization=0.8,
    trust_remote_code=True,
)

sampling_params = SamplingParams(
    temperature=0.9,
    top_p=0.9,
    max_tokens=MAX_TOKENS,
)

def make_conversation(example):
    prompt = []
    prompt.append({"role": "system", "content": "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer, and put your final answer within \\boxed{{}} . The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>. Note that respond by English, NOT use other languages."})
    prompt.append({"role": "user", "content": example["problem"]})
    unique_str = f"{time.time_ns()}{os.urandom(4)}{os.getpid()}".encode()
    unique_str = hashlib.blake2b(unique_str, digest_size=16).hexdigest()
    return {"prompt": prompt, "uuid": unique_str}

def pad_tensor_list(tensor_list, padding_value=0):
    tensor_list = [t.to(tensor_list[0].device) for t in tensor_list]
    
    # 使用pad_sequence自动padding
    padded = pad_sequence(
        tensor_list,
        batch_first=True,
        padding_value=padding_value
    )
    return padded

data = data.map(make_conversation)
result = {}
prompts = []
for index, item in enumerate(tqdm(data)):
    prompts.append(item)
    if len(prompts) >= batch_size:
        prompts_text = [maybe_apply_chat_template(example, tokenizer)["prompt"] for example in prompts]
        outputs = llm.generate(prompts_text, sampling_params)
        completion_ids = [out.token_ids for completions in outputs for out in completions.outputs]
        completion_ids = [torch.tensor(ids, device=device) for ids in completion_ids]
        print(len(completion_ids))
        completion_ids = pad_tensor_list(completion_ids, padding_value=tokenizer.pad_token_id)
        completions = tokenizer.batch_decode(completion_ids, skip_special_tokens=True)
        for index, item in enumerate(prompts):
            uuid = item["uuid"]
            cache = completions[index]
            result[uuid] = cache
        prompts = []

# 处理最后不足batch_size的剩余数据
print(f"final length: {len(prompts)}")
if prompts:
    prompts_text = [maybe_apply_chat_template(example, tokenizer)["prompt"] for example in prompts]
    outputs = llm.generate(prompts_text, sampling_params)
    
    completion_ids = [out.token_ids for completions in outputs for out in completions.outputs]
    completion_ids = [torch.tensor(ids, device=device) for ids in completion_ids]
    completion_ids = pad_tensor_list(completion_ids, padding_value=tokenizer.pad_token_id)
    completions = tokenizer.batch_decode(completion_ids, skip_special_tokens=True)

    for index, item in enumerate(prompts):
        result[item["uuid"]] = completions[index]
    print(f"Final batch processed. Total results: {len(result)}")
print(len(result))
def process_final_data(example):
    uuid = example['uuid']
    cache = result[uuid]
    return {"cache": cache}

data = data.map(process_final_data)
data.save_to_disk("aime24_cahce")

