# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# from . import gsm8k, math, prime_math, prime_code

from verl.utils.import_utils import deprecated


def default_compute_score(
    data_source,
    solution_str,
    ground_truth,
    extra_info=None,
    sandbox_fusion_url=None,
    concurrent_semaphore=None,
    memory_limit_mb=None,
):
    """Compute the score for a given solution based on the data source.

    Args:
        data_source (str): The source dataset identifier which determines the scoring method.
        solution_str (str): The solution string to be evaluated.
        ground_truth (str): The ground truth answer for comparison.
        extra_info (dict, optional): Additional information that might be needed for scoring. Defaults to None.

    Returns:
        float: The computed score as a floating point number. If the result is a dictionary,
               it returns the dictionary instead.

    Raises:
        NotImplementedError: If the reward function is not implemented for the given data source.
    """
    if data_source == "openai/gsm8k":
        from . import gsm8k

        res = gsm8k.compute_score(solution_str, ground_truth)
    elif data_source in ["lighteval/MATH", "DigitalLearningGmbH/MATH-lighteval", "HuggingFaceH4/MATH-500"]:
        from . import math_reward

        res = math_reward.compute_score(solution_str, ground_truth)
        # [Optional] Math-Verify Integration
        # For enhanced accuracy, consider utilizing Math-Verify (https://github.com/huggingface/Math-Verify).
        # Note: Math-Verify needs to be manually installed via pip: `pip install math-verify`.
        # To use it, override the `compute_score` function with the following implementation:

        # from . import math_verify
        # res = math_verify.compute_score(solution_str, ground_truth)
    elif data_source in ["math_dapo", "math", "math_dapo_reasoning"] or data_source.startswith("aime"):
        from . import math_dapo

        res = math_dapo.compute_score(solution_str, ground_truth)
    elif data_source in [
        "numina_aops_forum",
        "numina_synthetic_math",
        "numina_amc_aime",
        "numina_synthetic_amc",
        "numina_cn_k12",
        "numina_olympiads",
    ]:
        from . import prime_math

        res = prime_math.compute_score(solution_str, ground_truth)
    elif data_source in ["codecontests", "apps", "codeforces", "taco"]:
        # Use the passed sandbox_fusion_url if available
        if sandbox_fusion_url:
            from . import sandbox_fusion

            # Pass the URL directly, ground_truth likely contains test cases here
            res = sandbox_fusion.compute_score(
                sandbox_fusion_url, concurrent_semaphore, memory_limit_mb, solution_str, ground_truth, continuous=True
            )
        else:
            # If no sandbox URL is provided, fall back to prime_code or raise error
            from . import prime_code

            # Assuming prime_code doesn't need the URL
            res = prime_code.compute_score(solution_str, ground_truth, continuous=True)
    elif data_source in ["hiyouga/geometry3k"]:
        from . import geo3k

        res = geo3k.compute_score(solution_str, ground_truth)
    elif data_source in [
        "searchR1_nq",
        "searchR1_triviaqa",
        "searchR1_popqa",
        "searchR1_hotpotqa",
        "searchR1_2wikimultihopqa",
        "searchR1_musique",
        "searchR1_bamboogle",
    ]:
        from . import search_r1_like_qa_em

        res = search_r1_like_qa_em.compute_score(solution_str, ground_truth)

    else:
        raise NotImplementedError(f"Reward function is not implemented for {data_source=}")

    if isinstance(res, dict):
        return res
    elif isinstance(res, int | float | bool):
        return float(res)
    else:
        return float(res[0])

def dapo_wi_rllm_compute_score(data_source, solution_str, ground_truth, extra_info=None):
    if data_source.startswith("MATH##") or data_source.startswith("aime"):
        from . import math_verify
        res = math_verify.compute_score(solution_str, ground_truth)
    elif data_source == 'math_dapo':
        from . import math_dapo
        res = math_dapo.compute_score(solution_str, ground_truth)
    elif data_source == 'math_dapo_math_verify':
        from . import math_dapo
        res = math_dapo.compute_score(solution_str, ground_truth, is_use_math_verify = True)
    elif data_source == 'math_longcot':
        from . import math_dapo
        res = math_dapo.compute_score(solution_str, ground_truth, is_longcot=True, is_use_math_verify=False)
    elif 'math_longcot_math_verify' in data_source: # AIME2024(math_longcot_math_verify_aime2024) AIME2025(math_longcot_math_verify_aime2025) TrainingData(math_longcot_math_verify)
        from . import math_dapo
        # print("solution str: ", solution_str)
        res = math_dapo.compute_score(solution_str, ground_truth, is_longcot=True, is_use_math_verify=True)
    elif data_source == "coder1":
        from . import code_r1_compute
        res = code_r1_compute.compute_score(solution_str, ground_truth, is_longcot = False)
    elif "coder1_longcot" in data_source: # LiveCodeBench V5(coder1_longcot_lcbv5) LiveCodeBench V6(coder1_longcot_lcbv6) TrainingData(coder1_longcot)
        from . import code_r1_compute
        res = code_r1_compute.compute_score(solution_str, ground_truth, is_longcot = True)
    elif data_source == 'math_stem':
        from . import math_stem
        res = math_stem.compute_score(solution_str, ground_truth)
    elif data_source == 'math_stem_longcot':
        from . import math_stem
        res = math_stem.compute_score(solution_str, ground_truth, is_longcot=True)
    elif data_source == "longbench_pro":
        resp_rm_think = solution_str.split("</think>")[-1]
        if isinstance(ground_truth, list):
            ground_truth = ground_truth[-1]
        
        from . import long_bench_pro
        golden_label = ground_truth["doc_ids"]
        sub_task = extra_info["reward_mode"]
        language = ground_truth["summary"]
        res = long_bench_pro.compute_score(solution_str, golden_label, sub_task, language)
    elif data_source in [
            "ifeval", "apps", "taco", "code_contests", "codeforces", "livecodebench", "kodcode", "leetcode", "primeintellect", "humanevalplus",
            "ifeval_longcot", "apps_longcot", "taco_longcot", "code_contests_longcot", "codeforces_longcot", "livecodebench_longcot", "kodcode_longcot", "leetcode_longcot", "primeintellect_longcot", "humanevalplus_longcot",
    ]:
        from .rllm_code_reward import rllm_reward_fn_code
        
        if "_longcot" not in data_source:
            res = rllm_reward_fn_code(data_source, solution_str, ground_truth, is_longcot=False)
        else:
            data_source = data_source.replace("_longcot","")
            res = rllm_reward_fn_code(data_source, solution_str, ground_truth, is_longcot=True)
    else:
        raise NotImplementedError(f"Reward function is not implemented for {data_source=}")

    if isinstance(res, dict):
        return res
    elif isinstance(res, (int, float, bool)):
        return float(res)
    else:
        return float(res[0])


@deprecated("verl.utils.reward_score.default_compute_score")
def _default_compute_score(
    data_source,
    solution_str,
    ground_truth,
    extra_info=None,
    sandbox_fusion_url=None,
    concurrent_semaphore=None,
    memory_limit_mb=None,
):
    """
    Legacy function API to be deprecated. Please use `default_compute_score` instead.
    """
    return default_compute_score(
        data_source, solution_str, ground_truth, extra_info, sandbox_fusion_url, concurrent_semaphore, memory_limit_mb
    )


__all__ = ["default_compute_score"]
