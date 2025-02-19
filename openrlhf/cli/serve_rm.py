import argparse
import re
import os
import json
import torch
import uvicorn
import hashlib
import datasets
import subprocess
import regex as re
import numpy as np
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from tqdm import tqdm
from pathlib import Path
from openrlhf.models import get_llm_for_sequence_regression
from openrlhf.utils import get_tokenizer
from openrlhf.utils.logging_utils import init_logger
from transformers import AutoTokenizer
from typing import List
import asyncio
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

app = FastAPI()
# Create a thread pool for CPU-bound operations
thread_pool = ThreadPoolExecutor(max_workers=4)
# thread_pool = ProcessPoolExecutor(max_workers=2)

logger = init_logger(__name__)


def strip_sequence(text, pad_token, eos_token):
    pad_token_escaped = re.escape(pad_token)
    eos_token_escaped = re.escape(eos_token)

    pattern = f"^({eos_token_escaped}|{pad_token_escaped})+"
    text = re.sub(pattern, "", text)

    pattern = f"({eos_token_escaped}|{pad_token_escaped})+$"
    text = re.sub(pattern, "", text)
    return text

def hash_string(s):
    return hashlib.sha256(s.encode()).hexdigest()

def parse_conversation_from_llama3_prompt(prompt):
    # system\nuser\n...assistant\n...
    start_of_header_token = "<|start_header_id|>"
    end_of_header_token = "<|end_header_id|>\n\n"
    end_conv_token = "<|eot_id|>"
    start_idx = 0
    conversation = []
    while start_of_header_token in prompt:
        start_idx = prompt.find(start_of_header_token)
        role_end_idx = prompt.find(end_of_header_token, start_idx)
        role = prompt[start_idx + len(start_of_header_token):role_end_idx]
        role_end_idx += len(end_of_header_token)
        end_idx = prompt.find(end_conv_token, role_end_idx)
        content = prompt[role_end_idx:end_idx]
        conversation.append(
            {
                "role": role,
                "content": content
            }
        )
        prompt = prompt[end_idx + len(end_conv_token):]
    return conversation    

def parse_conversation_from_qwen2vl_prompt(prompt):
    # system\nuser\n...assistant\n...
    start_conv_token = "<|im_start|>"
    end_conv_token = "<|im_end|>"
    start_idx = 0
    conversation = []
    while start_conv_token in prompt:
        start_idx = prompt.find(start_conv_token)
        role_end_idx = prompt.find("\n", start_idx)
        role = prompt[start_idx + len(start_conv_token):role_end_idx]
        end_idx = prompt.find(end_conv_token, role_end_idx)
        content = prompt[role_end_idx + 1:end_idx]
        conversation.append(
            {
                "role": role,
                "content": content
            }
        )
        prompt = prompt[end_idx + len(end_conv_token):]
    return conversation   

class RewardModelProxy:
    def __init__(self, args):
        self.reward_model = get_llm_for_sequence_regression(
            args.reward_pretrain,
            "reward",
            normalize_reward=args.normalize_reward,
            use_flash_attention_2=args.flash_attn,
            bf16=args.bf16,
            load_in_4bit=args.load_in_4bit,
            value_head_prefix=args.value_head_prefix,
            device_map="auto",
        )
        self.reward_model.eval()

        self.tokenizer = get_tokenizer(
            args.reward_pretrain, self.reward_model, "left", None, use_fast=not args.disable_fast_tokenizer
        )
        self.policy_tokenizer = AutoTokenizer.from_pretrained(args.policy_pretrain, trust_remote_code=True, use_fast=not args.disable_fast_tokenizer)
        self.max_length = args.max_len
        self.batch_size = args.batch_size
        self.parse_conv_fn = parse_conversation_from_llama3_prompt if "llama3" in args.policy_pretrain else parse_conversation_from_qwen2vl_prompt

    def get_reward(self, queries):
        if self.batch_size is None:
            batch_size = len(queries)
        else:
            batch_size = self.batch_size
            
        # now, the current queries applied to the chat template of the policy model, we need to recover the original queries into a list of conversations
        conversations = []
        for query in queries:
            conversation = self.parse_conv_fn(query)
            conversations.append(conversation)
        queries = []
        for conversation in conversations:
            query = self.tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=False)
            queries.append(query)

        # remove pad_token
        for i in range(len(queries)):
            queries[i] = (
                strip_sequence(queries[i], self.tokenizer.pad_token, self.tokenizer.eos_token)
                + self.tokenizer.eos_token
            )
        logger.info(f"queries[0]: {queries[0]}")

        scores = []
        # batch
        with torch.no_grad():
            for i in range(0, len(queries), batch_size):
                inputs = self.tokenize_fn(
                    queries[i : min(len(queries), i + batch_size)], device=self.reward_model.device
                )
                r = self.reward_model(inputs["input_ids"], inputs["attention_mask"])
                r = r.tolist()
                scores.extend(r)
        return scores

    def tokenize_fn(self, texts, device):
        batch = self.tokenizer(
            texts,
            return_tensors="pt",
            add_special_tokens=False,
            max_length=self.max_length,
            padding=True,
            truncation=True,
        )
        return {k: v.to(device) for k, v in batch.items()}

class RuleBasedRewardModelProxy:
    def __init__(self, args):
        self.policy_tokenizer = AutoTokenizer.from_pretrained(args.policy_pretrain, trust_remote_code=True, use_fast=not args.disable_fast_tokenizer)
        self.max_length = args.max_len
        self.batch_size = args.batch_size
        self.parse_conv_fn = parse_conversation_from_llama3_prompt if "llama3" in args.policy_pretrain else parse_conversation_from_qwen2vl_prompt
        self.rule = args.rule
        self.dataset_name = args.dataset
        self.dataset = datasets.load_dataset(args.dataset, split="train")
        self.input_key = args.input_key
        self.gt_key = args.gt_key
        self.binary = args.binary
        self.n_workers = args.n_workers
        assert self.rule in ["test_case", "exact_match", "code_format_reward"]
        if self.rule == "test_case":    
            from acecoder import evaluate_test_cases

        dataset_questions = []
        for conversation in self.dataset['context_messages']:
            idx = 0
            while conversation[idx]['role'] != "user":
                idx += 1
            dataset_questions.append(conversation[idx]['content'])
        dataset_questions = [self.policy_tokenizer.decode(self.policy_tokenizer.encode(x)) for x in tqdm(dataset_questions, desc="Tokenizing dataset questions")]
        self.hash_map = {
            hash_string(q): item for q, item in zip(dataset_questions, self.dataset)
        }
        
    
    async def get_reward_async(self, queries: List[str]) -> List[float]:
        # If get_reward is CPU-bound, run it in a thread pool
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            thread_pool, 
            self.get_reward,
            queries
        )
            
    def get_reward(self, queries):
        if self.batch_size is None:
            batch_size = len(queries)
        else:
            batch_size = self.batch_size
            
        # now, the current queries applied to the chat template of the policy model, we need to recover the original queries into a list of conversations
        conversations = []
        for query in queries:
            conversation = self.parse_conv_fn(query)
            conversations.append(conversation)
        
        scores = []
        if self.rule == "test_case":
            from acecoder import evaluate_test_cases
            questions = []
            responses = []
            for conversation in conversations:
                idx = 0
                while conversation[idx]['role'] != "user":
                    idx += 1
                questions.append(conversation[idx]['content'])
                responses.append(conversation[idx + 1]['content'])
            question_hashes = [hash_string(question) for question in questions]
            if not all([x in self.hash_map for x in question_hashes]):
                torch.save(question_hashes, "question_hashes.pt")
                torch.save(conversations, "conversations.pt")
                torch.save(questions, "questions.pt")
                torch.save(responses, "responses.pt")
                torch.save(self.hash_map, "hash_map.pt")
                raise Exception("Not all questions are in the dataset")
            assert all([x in self.hash_map for x in question_hashes]), "Not all questions are in the dataset"
            test_cases = [self.hash_map[x][self.gt_key] for x in question_hashes]
            samples = [
                {
                    'task_id': question_hash,
                    'prompt': question,
                    'output': response,
                    'tests': test_case,
                    '_identifier': f"{question_hash}_{i}"
                }
                for i, (question_hash, question, response, test_case) in enumerate(zip(question_hashes, questions, responses, test_cases))
            ]
            ## save samples to a file
            # temp_dir = "./temp/"
            # temp_file = temp_dir + f"{hash_string(''.join(queries))}.jsonl"
            # os.makedirs(temp_dir, exist_ok=True)
            # with open(temp_file, "w") as f:
            #     for sample in samples:
            #         f.write(json.dumps(sample) + "\n")
            # # python -m openrlhf.cli.eval_test_cases --samples temp_file --n_workers 8 --test_details --output_file output_file
            # # python -m openrlhf.cli.eval_test_cases --samples /root/dongfu/OpenRLHF/temp/a9f6894d094547fb67e2aad4026c4b7c8a8b5889ae79b25ef59c7ef7372cdde3.jsonl --n_workers 8 --test_details --output_file /root/dongfu/OpenRLHF/temp/a9f6894d094547fb67e2aad4026c4b7c8a8b5889ae79b25ef59c7ef7372cdde3.eval_results.jsonl
            # if not self.binary:
            #     output_file = Path(temp_file).with_suffix(".eval_results.jsonl").absolute()
            #     command = f"python -m openrlhf.cli.eval_test_cases --samples {temp_file} --n_workers {self.n_workers} --test_details --output_file {output_file}"
            # else:
            #     output_file = Path(temp_file).with_suffix(".eval_results_binary.jsonl")
            #     command = f"python -m openrlhf.cli.eval_test_cases --samples {temp_file} --n_workers {self.n_workers} --output_file {output_file}"
            # print(command)
            # subprocess.run(command, shell=True)
            # with open(output_file, "r") as f:
            #     all_samples_results = [json.loads(x) for x in f]
            # pass_rates = [x['pass_rate'] for x in all_samples_results]
            # save samples to a file
            all_samples_results, pass_rates = evaluate_test_cases(samples, n_workers=self.n_workers, test_details=not self.binary, min_time_limit=1, gt_time_limit_factor=1)
            scores = pass_rates
            if self.binary:
                scores = [1 if x == 1 else 0 for x in scores] # if binary
        elif self.rule == "code_format_reward":
            questions = []
            responses = []
            for conversation in conversations:
                idx = 0
                while conversation[idx]['role'] != "user":
                    idx += 1
                questions.append(conversation[idx]['content'])
                responses.append(conversation[idx + 1]['content'])
            question_hashes = [hash_string(question) for question in questions]
            
            
            
            # Format Reward 1: force <think> ... </think> <answer> ... </answer> format
            scores_1 = [int(re.match(r"<think>(.|\n)*?</think>\s*<answer>(.|\n)*?</answer>", response.strip(' \n')) is not None) for response in responses]
            
            # Format Reward 2: encourage the include of coding in the thinking process
            extracted_thinks = [re.search(r"<think>(.|\n)*?</think>", response) for response in responses]
            extracted_thinks = [x.group() if x is not None else None for x in extracted_thinks]
            scores_2 = [re.findall(r"```.*\n(.|\n)*?\n```", think) if think is not None else [] for think in extracted_thinks]
            scores_2 = [len(x) for x in scores_2]
            scores_2 = [np.tanh(x) for x in scores_2]
            
            # Format Reward 3: There should be a code block in the answer
            extracted_answers = [re.search(r"<answer>(.|\n)*?</answer>", response) for response in responses]
            extracted_answers = [x.group() if x is not None else None for x in extracted_answers]
            scores_3 = [int(re.match(r"```.*\n(.|\n)*?\n```", answer) is not None) if answer is not None else 0 for answer in extracted_answers]
            
            # Combine the scores    
            scores = [0.5 * x + 0.25 * y + 0.25 * z for x, y, z in zip(scores_1, scores_2, scores_3)]
            
        elif self.rule == "exact_match":
            raise NotImplementedError

        return scores

    def tokenize_fn(self, texts, device):
        batch = self.tokenizer(
            texts,
            return_tensors="pt",
            add_special_tokens=False,
            max_length=self.max_length,
            padding=True,
            truncation=True,
        )
        return {k: v.to(device) for k, v in batch.items()}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Reward Model
    parser.add_argument("--reward_pretrain", type=str, default=None, help="HF model name or path")
    parser.add_argument("--policy_pretrain", type=str, default=None, help="HF model name or path")
    parser.add_argument("--normalize_reward", action="store_true", default=False, help="Enable Reward Normazation")
    parser.add_argument("--value_head_prefix", type=str, default="score")
    parser.add_argument("--max_len", type=int, default="2048")
    parser.add_argument("--rule", type=str, default="reward_model", help="Rule-based reward model")
    parser.add_argument("--dataset", type=str, default="test", help="Dataset for the rule-based reward model")
    parser.add_argument("--input_key", type=str, default="context_messages", help="Key for the input")
    parser.add_argument("--gt_key", type=str, default="tests", help="Key used for rule-based reward model that compares with the output to get the reward")
    parser.add_argument("--binary", action="store_true", default=False, help="Binary reward")
    parser.add_argument("--n_workers", type=int, default=8, help="Number of workers for the rule-based reward model")

    parser.add_argument("--port", type=int, default=5000, help="Port number for the server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="IP for the server")

    # Performance
    parser.add_argument("--load_in_4bit", action="store_true", default=False)
    parser.add_argument("--bf16", action="store_true", default=False, help="Enable bfloat16")
    parser.add_argument("--flash_attn", action="store_true", default=False, help="Enable FlashAttention2")
    parser.add_argument("--disable_fast_tokenizer", action="store_true", default=False)
    parser.add_argument("--batch_size", type=int, default=None)

    args = parser.parse_args()

    # server
    if args.rule == "reward_model":
        reward_model = RewardModelProxy(args)
    else:
        reward_model = RuleBasedRewardModelProxy(args)
    app = FastAPI()

    @app.post("/get_reward")
    async def get_reward(request: Request):
        data = await request.json()
        queries = data.get("query")
        if args.rule == "reward_model":
            rewards = reward_model.get_reward(queries)
        else:
            print(f"Computing rewards for {len(queries)} queries")
            rewards = await reward_model.get_reward_async(queries)
        result = {"rewards": rewards}
        logger.info(f"Sent JSON: {result}")
        return JSONResponse(result)

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
