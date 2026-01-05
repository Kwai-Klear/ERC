from math_verify import parse
from math_verify import verify as m_verify
import sys
from dataclasses import dataclass, field
import json

@dataclass
class ScoringConfig:
    correct_score: float = 1.0
    incorrect_score: float = -1.0
    format_error_score: float = -1.0
    wo_bos_think: float = -2.0
    wo_eos_think: float = -0.5

scoring_config = ScoringConfig()

def math_verify_from_sky(solution_str: str, ground_truth: str):
    ground_truth = [ground_truth] if isinstance(ground_truth, str) else ground_truth
    
    # 0 in case parsing cannot be completed
    try:
        math_verify_parsed = parse(solution_str, parsing_timeout=5)
    except Exception:
        return scoring_config.incorrect_score
    
    # 0 if parsing is problematic
    if len(math_verify_parsed) < 2:
        return scoring_config.incorrect_score
    
    # We perform a quick string match first
    if math_verify_parsed[1] in ground_truth:
        return scoring_config.correct_score
    
    # We now fallback to semantic verification
    for gt in ground_truth:
        try:
            if m_verify(
                parse(f"\\boxed{{{gt}}}", parsing_timeout=5),
                math_verify_parsed,
                timeout_seconds=5,
            ):
                return scoring_config.correct_score
        except Exception:
            continue
    
    # Very unlikely to be correct after the above matches
    return scoring_config.incorrect_score


data = json.load(sys.stdin)
solution_str, ground_truth, = data[0], data[1]
run = math_verify_from_sky(solution_str,ground_truth)
print(run)
