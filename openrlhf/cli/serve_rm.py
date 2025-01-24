import argparse
import re

import torch
import uvicorn
import hashlib
import datasets
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from openrlhf.models import get_llm_for_sequence_regression
from openrlhf.utils import get_tokenizer
from openrlhf.utils.logging_utils import init_logger
from transformers import AutoTokenizer

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
            query = self.tokenizer.apply_template(conversation, self.policy_tokenizer, tokenize=False, add_generation_prompt=False)
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
        self.hash_map = {
            hash_string(item['context_messages'][0]['content']): item for item in self.dataset
        }
        assert self.rule in ["test_case", "exact_match"]
        
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
            questions = [conv[0]['content'] for conv in conversations]
            responses = [conv[1]['content'] for conv in conversations]
            extracted_codes = ...
            question_hashes = [hash_string(question) for question in questions]
            assert all([x in self.hash_map for x in question_hashes]), "Not all questions are in the dataset"
            test_cases = [self.hash_map[x][self.gt_key] for x in question_hashes]

        

        # remove pad_token
        for i in range(len(queries)):
            queries[i] = (
                strip_sequence(queries[i], self.tokenizer.pad_token, self.tokenizer.eos_token)
                + self.tokenizer.eos_token
            )
        logger.info(f"queries[0]: {queries[0]}")

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
        rewards = reward_model.get_reward(queries)
        result = {"rewards": rewards}
        logger.info(f"Sent JSON: {result}")
        return JSONResponse(result)

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
