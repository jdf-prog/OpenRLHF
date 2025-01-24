import json
import multiprocessing
import os
import pickle
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from warnings import warn

import numpy as np
from termcolor import cprint
from tqdm import tqdm

from evalplus.codegen import run_codegen
from evalplus.config import *
from evalplus.data import (
    get_human_eval_plus,
    get_human_eval_plus_hash,
    get_mbpp_plus,
    get_mbpp_plus_hash,
    load_solutions,
)
from evalplus.data.mbpp import mbpp_serialize_inputs
from evalplus.data.utils import CACHE_DIR
from evalplus.eval import (
    PASS,
    compatible_eval_result,
    estimate_pass_at_k,
    untrusted_check,
)
from evalplus.eval._special_oracle import MBPP_OUTPUT_NOT_NONE_TASKS
from evalplus.gen.util import trusted_exec

# 1st item: the status
# 2nd item (optional): the detailed pass/fail boolean for each input
Result = Tuple[str, List[bool]]

def check_correctness(
    dataset: str,
    task_id: int,
    completion_id: int,
    test_inputs: List[Any],
    entry_point: str,
    solution: str,
    expected_output: Dict[str, List],
    atol: int=1e-6,
    base_only=False,
    fast_check=False,
    identifier=None,
    min_time_limit: float = DEFAULT_MIN_TIME_LIMIT,
    gt_time_limit_factor: float = DEFAULT_GT_TIME_LIMIT_FACTOR,
) -> Dict[str, Result]:  # {...}, "base" | "plus" -> (status, details)
    ret = {
        "completion_id": completion_id,
        "task_id": task_id,
        "_identifier": identifier,
        "solution": solution,
    }
    ret["base"] = untrusted_check(
        dataset,
        solution,
        test_inputs,
        entry_point,
        expected=expected_output["base"],
        atol=atol,
        ref_time=expected_output["base_time"],
        fast_check=fast_check,
        min_time_limit=min_time_limit,
        gt_time_limit_factor=gt_time_limit_factor,
    )

    return ret

def get_entry_point_from_test_case(test_case: str) -> str:
    """
    Get the entry point from the first test case.
    Args:
        test_case: a test case string, like "assert f(1) == 2"
    Returns:
        the entry point, like "f"
    """
    start_idx = test_case.find("assert ") + len("assert ")
    end_idx = test_case.find("(")
    return test_case[start_idx:end_idx]

def get_test_inputs_outputs_from_test_case(test_cases: List[str]) -> Tuple[List[str], List[str]]:
    """
    Get the inputs and outputs from the test cases.
    Args:
        test_cases: a list of test case strings
    Returns:
        a tuple of inputs and outputs
    """
    inputs = []
    outputs = []
    for test_case in test_cases:
        start_idx = test_case.find("(") + 1
        end_idx = test_case.find(")")
        test_input = test_case[start_idx:end_idx]
        test_input = test_input.split(",") if test_input else []
        test_input = [eval(x) for x in test_input]
        inputs.append(test_input)
        
        output_start_idx = test_case.find("==") + 2
        assert output_start_idx != -1, f"Cannot find '==' in {test_case}"
        output = eval(test_case[output_start_idx:].strip())
        outputs.append(output)
    return inputs, outputs
    

def evaluate(
    dataset: str,
    samples: str,
    base_only: bool = False,
    parallel: Optional[int] = None,
    i_just_wanna_run: bool = False,
    test_details: bool = False,
    min_time_limit: float = DEFAULT_MIN_TIME_LIMIT,
    gt_time_limit_factor: float = DEFAULT_GT_TIME_LIMIT_FACTOR,
    output_file: Optional[str] = None,
):

    n_workers = parallel or max(1, multiprocessing.cpu_count() // 2)

    if os.path.isdir(samples):
        result_path = os.path.join(samples, "eval_results.json")
    else:
        assert samples.endswith(".jsonl")
        # legacy compatibility
        if os.path.exists(samples.replace(".jsonl", "_eval_results.json")):
            result_path = samples.replace(".jsonl", "_eval_results.json")
        else:
            result_path = samples.replace(".jsonl", ".eval_results.json")

    if output_file is not None:
        result_path = output_file

    if os.path.isfile(result_path) and not i_just_wanna_run:
        print(f"Load from previous results from {result_path}")
        with open(result_path, "r") as f:
            results = json.load(f)

        results = compatible_eval_result(results)
    else:
        all_samples = load_solutions(samples)
        dataset_hash = None

        results = {
            "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "hash": dataset_hash,
            "eval": {},
        }

        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = []
            completion_id = Counter()
            n_samples = 0
            eval_results = defaultdict(list)  # task_id ->
            remainings = set()

            print("Reading samples...")
            for sample in tqdm(all_samples):
                task_id = sample["task_id"]
                test_inputs, expected_output = get_test_inputs_outputs_from_test_case(sample["tests"])
                entry_point = get_entry_point_from_test_case(sample['tests'][0])
                solution = sample["solution"]
                remainings.add(sample["_identifier"])
                args = (
                    dataset,
                    task_id,
                    completion_id[task_id],
                    test_inputs,
                    entry_point,
                    solution,
                    expected_output,
                    base_only,
                    not test_details,  # fast_check
                    sample["_identifier"],
                    min_time_limit,
                    gt_time_limit_factor,
                )
                futures.append(executor.submit(check_correctness, *args))
                completion_id[task_id] += 1
                n_samples += 1

            assert n_samples == len(remainings), "Missing problems in unfinished"

            def stucking_checker():
                while remainings:
                    last_size = len(remainings)
                    time.sleep(20)
                    if last_size != len(remainings) or len(remainings) == 0:
                        continue
                    # Potential stucking
                    warn("No samples had finished testing in the last 20s")
                    warn(f"{len(remainings)} samples to be tested: {remainings}")

            threading.Thread(target=stucking_checker).start()

            all_samples_results = []
            for future in tqdm(as_completed(futures), total=n_samples):
                result = future.result()
                remainings.remove(result["_identifier"])
                result['pass_rate'] = sum(result['base'][1]) / len(result['base'][1])
                all_samples_results.append(result)
                eval_results[result["task_id"]].append(result)
    pass_rates = [x['pass_rate'] for x in all_samples_results]
    return all_samples_results, pass_rates

def main():
    from fire import Fire

    Fire(evaluate)


if __name__ == "__main__":
    main()
