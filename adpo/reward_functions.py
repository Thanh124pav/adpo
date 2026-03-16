"""
Reward functions for mathematical reasoning evaluation.

Each function follows verl's reward interface:
    compute_score(data_source, solution_str, ground_truth, extra_info) -> float
"""

import re
import math
from typing import Optional


def extract_boxed_answer(text: str) -> Optional[str]:
    """Extract answer from \\boxed{...} in LaTeX-formatted responses."""
    idx = text.rfind(r"\boxed{")
    if idx == -1:
        return None
    i = idx + len(r"\boxed{")
    depth = 1
    result = []
    while i < len(text) and depth > 0:
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                break
        result.append(text[i])
        i += 1
    return "".join(result).strip()


def extract_last_number(text: str) -> Optional[str]:
    """Extract the last numeric value from text."""
    pattern = r"-?[\d,]+\.?\d*"
    matches = re.findall(pattern, text)
    if matches:
        return matches[-1].replace(",", "")
    return None


def extract_answer_after_marker(text: str, marker: str = "####") -> Optional[str]:
    """Extract answer appearing after a specific marker (e.g., GSM8K ####)."""
    if marker in text:
        return text.split(marker)[-1].strip()
    return None


def normalize_numeric(value: str) -> Optional[float]:
    """Normalize a string to a float for comparison."""
    if value is None:
        return None
    value = value.strip().replace(",", "").replace(" ", "")
    if "/" in value:
        parts = value.split("/")
        if len(parts) == 2:
            try:
                return float(parts[0]) / float(parts[1])
            except (ValueError, ZeroDivisionError):
                return None
    if value.endswith("%"):
        try:
            return float(value[:-1]) / 100.0
        except ValueError:
            return None
    try:
        return float(value)
    except ValueError:
        return None


def normalize_answer(answer: str) -> str:
    """Normalize a LaTeX/text answer for comparison."""
    if answer is None:
        return ""
    answer = answer.strip()
    answer = answer.replace(r"\$", "").replace("$", "")
    answer = answer.replace(r"\%", "%")
    answer = answer.replace(r"\text{", "").rstrip("}")
    answer = answer.replace(r"\mathrm{", "").rstrip("}")
    answer = answer.replace(r"\,", "").replace(r"\!", "")
    answer = answer.replace(r"\left", "").replace(r"\right", "")
    answer = answer.replace(r"\cdot", "*")
    return answer.strip()


def is_equiv(pred: str, target: str, tol: float = 1e-5) -> bool:
    """Check if two answers are mathematically equivalent."""
    if pred is None or target is None:
        return False
    pred_norm = normalize_answer(pred)
    target_norm = normalize_answer(target)
    if pred_norm == target_norm:
        return True
    pred_num = normalize_numeric(pred_norm)
    target_num = normalize_numeric(target_norm)
    if pred_num is not None and target_num is not None:
        if abs(target_num) < tol:
            return abs(pred_num - target_num) < tol
        return abs(pred_num - target_num) / max(abs(target_num), 1e-10) < tol
    return False


def compute_score_gsm8k(data_source, solution_str, ground_truth, extra_info=None):
    """GSM8K reward: extract answer after ####, compare numerically."""
    pred = extract_boxed_answer(solution_str)
    if pred is None:
        pred = extract_answer_after_marker(solution_str, "####")
    if pred is None:
        pred = extract_last_number(solution_str)
    target = extract_answer_after_marker(ground_truth, "####")
    if target is None:
        target = extract_last_number(ground_truth)
    if is_equiv(pred, target):
        return 1.0
    if pred is not None and extract_boxed_answer(solution_str) is not None:
        return 0.1
    return 0.0


def compute_score_math(data_source, solution_str, ground_truth, extra_info=None):
    """MATH dataset reward: extract \boxed{} answer, compare."""
    pred = extract_boxed_answer(solution_str)
    if pred is None:
        pred = extract_last_number(solution_str)
    target = extract_boxed_answer(ground_truth)
    if target is None:
        target = ground_truth.strip()
    return 1.0 if is_equiv(pred, target) else 0.0


def compute_score_competition(data_source, solution_str, ground_truth, extra_info=None):
    """Competition math reward (AMC, AIME, HMMT, etc.): numeric answer."""
    pred = extract_boxed_answer(solution_str)
    if pred is None:
        pred = extract_last_number(solution_str)
    target = ground_truth.strip()
    return 1.0 if is_equiv(pred, target) else 0.0


def compute_score_olympiad(data_source, solution_str, ground_truth, extra_info=None):
    """OlympiadBench / proof-style: check boxed answer or numeric."""
    pred = extract_boxed_answer(solution_str)
    if pred is None:
        pred = extract_last_number(solution_str)
    target = extract_boxed_answer(ground_truth)
    if target is None:
        target = ground_truth.strip()
    return 1.0 if is_equiv(pred, target) else 0.0


REWARD_REGISTRY = {
    "gsm8k": compute_score_gsm8k,
    "math": compute_score_math,
    "math500": compute_score_math,
    "numina_math": compute_score_math,
    "open_math_reasoning": compute_score_math,
    "aops_instruct": compute_score_math,
    "big_math_rl": compute_score_math,
    "fine_math": compute_score_math,
    "amc_2023": compute_score_competition,
    "aime_2024": compute_score_competition,
    "aime_2025": compute_score_competition,
    "olympiad_bench": compute_score_olympiad,
    "minerva_math": compute_score_math,
    "omni_math": compute_score_math,
    "hmmt": compute_score_competition,
    "brumo": compute_score_competition,
    "cmimc": compute_score_competition,
}


def compute_score(data_source, solution_str, ground_truth, extra_info=None):
    """Unified reward function -- dispatches based on data_source.

    Entry point for verl's RewardManager:
        custom_reward_function.path = "adpo/reward_functions.py"
        custom_reward_function.name = "compute_score"
    """
    key = data_source.lower().replace("-", "_").replace(" ", "_")
    for registered_key, fn in REWARD_REGISTRY.items():
        if registered_key in key:
            return fn(data_source, solution_str, ground_truth, extra_info)
    return compute_score_math(data_source, solution_str, ground_truth, extra_info)
