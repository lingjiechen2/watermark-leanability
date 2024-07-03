import argparse
import os
os.environ['http_proxy'] = "http://10.176.52.116:7890"
os.environ['https_proxy'] = "http://10.176.52.116:7890"
os.environ['all_proxy'] = "socks5://10.176.52.116:7891"
import sys
sys.path.append(('../'))
sys.path.append(('../../'))
# sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))
import json
from typing import Dict
import torch
from datasets import load_dataset
from tqdm import tqdm
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
from task_vector import TaskVector

from watermarks.kgw.watermark_processor import WatermarkDetector
from watermarks.aar.aar_watermark import AarWatermarkDetector
from watermarks.watermark_types import WatermarkType
import logging

# Step 1: Configure logging to write to a file
logging.basicConfig(
    filename='/remote-home/miintern1/watermark-learnability/logs/watermark_strength_log.txt',  # Specify the log file name
    level=logging.INFO,          # Set the logging level to INFO
    format='%(asctime)s - %(levelname)s - %(message)s'  # Set the log message format
)

logging.info(f"{torch.cuda.device_count()=}")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
# dataset = load_dataset("allenai/c4", "realnewslike", "validation")
dataset = load_dataset("allenai/c4", "realnewslike", split="validation", streaming="store_true")

max_length = 250
min_length = 250
num_samples = 1000
batch_size = 16
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
set_seed(42)
logging.info("Using device: {}".format(device))
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
def filter_length(example):
        return len(tokenizer(example['text'], truncation=True, max_length=max_length)["input_ids"]) >= min_length

def encode(examples):
    trunc_tokens = tokenizer(
        examples['text'],
        truncation=True,
        padding=True,
        max_length=max_length,
        return_tensors="pt"
    ).to(device)
    # Examples are truncated to max_length, which comprises the possible generation prompt and the text to be generated
    examples["text"] = tokenizer.batch_decode(trunc_tokens["input_ids"], skip_special_tokens=True)
    prompt = tokenizer(
        examples["text"], truncation=True, padding=True, max_length=50, return_tensors="pt",
    ).to(device)
    examples["prompt_text"] = tokenizer.batch_decode(prompt["input_ids"], skip_special_tokens=True)
    examples["input_ids"] = prompt["input_ids"]
    examples["attention_mask"] = prompt["attention_mask"]
    examples["text_completion"] = tokenizer.batch_decode(
        trunc_tokens["input_ids"][:, 50:], skip_special_tokens=True
    )
    return examples

dataset = dataset.filter(filter_length)
# Set how many samples will be skipped
dataset = dataset.map(encode, batched=True)

dataloader = torch.utils.data.DataLoader(dataset, batch_size)

prompts = []
human_text = []
prompt_text = []
full_human_text = []
for batch in dataloader:
    if len(human_text) >= num_samples:
        break
    if (type(batch["input_ids"]) == list):
        batch["input_ids"] = torch.stack(batch["input_ids"], dim=1).to(device)
    if (type(batch["attention_mask"]) == list):
        batch["attention_mask"] = torch.stack(batch["attention_mask"], dim=1).to(device)
    prompts.append(batch)
    human_text.extend(batch["text_completion"])
    prompt_text.extend(batch["prompt_text"])
    full_human_text.extend(batch["text"])
human_text = human_text[:num_samples]
prompt_text = prompt_text[:num_samples]
full_human_text = full_human_text[:num_samples]
raw_input = {
    "prompts": prompts,
    "human_text": human_text,
    "prompt_text": prompt_text,
    "full_human_text": full_human_text,
}
logging.info("Data loaded and processed successfully")

watermark_config = {
     "cygu/llama-2-7b-logit-watermark-distill-kgw-k0-gamma0.25-delta1":{"type": "kgw", "k": 0, "gamma": 0.25, "delta": 1.0, "seeding_scheme": "simple_0", "kgw_device": "cpu"},
     "cygu/llama-2-7b-logit-watermark-distill-kgw-k0-gamma0.25-delta2":{"type": "kgw", "k": 0, "gamma": 0.25, "delta": 2.0, "seeding_scheme": "simple_0", "kgw_device": "cpu"},
     "cygu/llama-2-7b-logit-watermark-distill-kgw-k1-gamma0.25-delta1":{"type": "kgw", "k": 1, "gamma": 0.25, "delta": 1.0, "seeding_scheme": "simple_1", "kgw_device": "cpu"},
     "cygu/llama-2-7b-logit-watermark-distill-kgw-k1-gamma0.25-delta2":{"type": "kgw", "k": 1, "gamma": 0.25, "delta": 2.0, "seeding_scheme": "simple_1", "kgw_device": "cpu"},
     "cygu/llama-2-7b-logit-watermark-distill-kgw-k2-gamma0.25-delta2":{"type": "kgw", "k": 2, "gamma": 0.25, "delta": 2.0, "seeding_scheme": "simple_2", "kgw_device": "cpu"},
}
vanilla_model_name = "meta-llama/Llama-2-7b-hf"

def compute_p_value(samples, detector):
    score_list = []
    for s in tqdm(samples):
        score = detector.detect(s)
        score_list.append(score['p_value'])
    return score_list

def move_to_device(batch, device):
    """Move batch to the specified device."""
    new_batch = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            new_batch[key] = value.to(device)
        elif isinstance(value, list):
            # Assuming lists are lists of tensors, move each tensor to the device
            new_batch[key] = [v.to(device) if isinstance(v, torch.Tensor) else v for v in value]
        else:
            new_batch[key] = value
    return new_batch


all_model_dict = {}
for watermark_name, watermark_config in watermark_config.items():
    logging.info(f"Processing watermark {watermark_name}")
    watermarked_model = AutoModelForCausalLM.from_pretrained(watermark_name)
                                                        #, device_map = "auto")
    watermarked_model = watermarked_model.half()
    
    vanilla_model= AutoModelForCausalLM.from_pretrained(vanilla_model_name)
                                                        #, device_map = "auto")
    vanilla_model = vanilla_model.half()
    task_vector = TaskVector(vanilla_model, watermarked_model)
    # task_vector.to("cuda:1")
    # watermarked_model.to("cuda:0")
    # vanilla_model.to("cuda:1")
    all_model_dict[watermark_name] = dict()

    DO_SAMPLE = True
    temperature=1.0
    top_p=0.9
    top_k=0
    vanilla_output_results = []
    watermarked_output_results = []
    
    coefficient_list = np.arange(0, 1.05, 0.05)

    for coefficient in coefficient_list:
        all_model_dict[watermark_name][coefficient] = dict()
        coefficient_watermarked_model = task_vector.apply_to(vanilla_model, scaling_coef = coefficient)
        coefficient_watermarked_model.to("cuda:0")
        print(next(coefficient_watermarked_model.parameters()).device)
        vanilla_model.to("cuda:1")

        for batch in tqdm(prompts):
            if len(vanilla_output_results) >= num_samples:
                break
            with torch.no_grad():
                # print(batch)
                batch_vanilla = move_to_device(batch, "cuda:1")
                batch_watermarked = move_to_device(batch, "cuda:0")
             
                logging.info(f"Vanilla model input device: {batch_vanilla['input_ids'].device}")
                logging.info(f"Vanilla model device: {next(vanilla_model.parameters()).device}")
                logging.info(f"Watermarked model input device: {batch_watermarked['input_ids'].device}")
                logging.info(f"Watermarked model device: {next(coefficient_watermarked_model.parameters()).device}")

                vanilla_output = vanilla_model.generate(
                            input_ids=batch_vanilla["input_ids"],
                            attention_mask=batch_vanilla["attention_mask"],
                            do_sample=DO_SAMPLE,
                            min_new_tokens=200,
                            max_new_tokens=200,
                            temperature=temperature,
                            top_p=top_p,
                            top_k=top_k,
                            pad_token_id=tokenizer.eos_token_id,
                        )
                watermarked_output = coefficient_watermarked_model.generate(
                            input_ids=batch_watermarked["input_ids"],
                            attention_mask=batch_watermarked["attention_mask"],
                            do_sample=DO_SAMPLE,
                            min_new_tokens=200,
                            max_new_tokens=200,
                            temperature=temperature,
                            top_p=top_p,
                            top_k=top_k,
                            pad_token_id=tokenizer.eos_token_id,
                        )
            # break
        
        # torch.cuda.empty_cache()
        n_input_tokens = batch["input_ids"].shape[1]
        vanilla_output_cpu = vanilla_output[:, n_input_tokens:].cpu()
        watermarked_output_cpu = watermarked_output[:, n_input_tokens:].cpu()

        vanilla_output_results.extend(tokenizer.batch_decode(vanilla_output_cpu, skip_special_tokens=True))
        watermarked_output_results.extend(tokenizer.batch_decode(watermarked_output_cpu, skip_special_tokens=True))

        vanilla_output_results = vanilla_output_results[:num_samples]
        watermarked_output_results = watermarked_output_results[:num_samples]

        detector = WatermarkDetector(
                        device=watermark_config.get("kgw_device", 'cpu'),
                        tokenizer=tokenizer,
                        vocab=tokenizer.get_vocab().values(),
                        gamma=watermark_config["gamma"],
                        seeding_scheme=watermark_config["seeding_scheme"],
                        normalizers=[],
                    )
        
        vanilla_scores = compute_p_value(vanilla_output_results, detector)
        watermarked_scores = compute_p_value(watermarked_output_results, detector)
        all_model_dict[watermark_name][coefficient]["vanilla_scores"] = vanilla_scores
        all_model_dict[watermark_name][coefficient]["watermarked_scores"] = watermarked_scores
        logging.info(f"Finished processing coefficient {coefficient}")
        # break
    del vanilla_model
    del watermarked_model
    logging.info(f"Finished processing watermark {watermark_name}")


save_path = '/remote-home/miintern1/watermark-learnability/data/c4/watermark_strength.json'
with open(save_path, 'w') as json_file:
    json.dump(all_model_dict, json_file, indent=4)

logging.info(f"Dictionary has been saved to {save_path}")

    
    
