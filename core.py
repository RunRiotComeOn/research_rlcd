"""Prompt, strict output contract, RLCD reward, and calibration metrics."""
from dataclasses import dataclass
import math
import re

CONFIDENCES = tuple(0.05 + 0.1 * i for i in range(10))
FORMAT_PENALTY = -1.0
PATTERN = re.compile(r"\AAction: ([A-Z])\nConfidence: ([0-9])\Z")


@dataclass(frozen=True)
class DecisionOutput:
    action_index: int | None
    confidence_bin: int | None
    confidence: float | None
    valid_format: bool


def prompt_text(row):
    choices = "\n".join(f"{chr(65+i)}. {choice}" for i, choice in enumerate(row["choices"]))
    return (
        "You are a video decision model. Use the video to answer the question.\n"
        "Choose one option and estimate the probability that your choice is correct.\n"
        "Do not explain your reasoning.\n\n"
        f"Question:\n{row['question']}\n\nAvailable actions:\n{choices}\n\n"
        "Respond exactly in this format:\nAction: <LETTER>\nConfidence: <0-9>"
    )


def parse_output(raw, num_choices):
    match = PATTERN.fullmatch(raw.strip())
    if not match:
        return DecisionOutput(None, None, None, False)
    index = ord(match.group(1)) - 65
    if index < 0 or index >= num_choices:
        return DecisionOutput(None, None, None, False)
    confidence_bin = int(match.group(2))
    return DecisionOutput(index, confidence_bin, CONFIDENCES[confidence_bin], True)


def reward(decision, gold, lambda_cal=0.2, format_penalty=FORMAT_PENALTY):
    if not decision.valid_format:
        return format_penalty
    if lambda_cal < 0:
        raise ValueError("lambda_cal must be nonnegative")
    correct = int(decision.action_index == gold)
    return correct - lambda_cal * (decision.confidence - correct) ** 2


def assert_correctness_margin(lambda_cal):
    correct_min = min(1 - lambda_cal * (q - 1) ** 2 for q in CONFIDENCES)
    wrong_max = max(-lambda_cal * q**2 for q in CONFIDENCES)
    if correct_min <= wrong_max:
        raise ValueError("Calibration weight violates correctness-first reward margin")
    return correct_min - wrong_max


def calibration_metrics(rows, key="confidence", bins=10):
    valid = [r for r in rows if r.get(key) is not None]
    n = len(valid)
    if not n:
        return {"count": 0, "brier": None, "ece": None, "nll": None, "table": [], "risk_coverage": []}
    brier = sum((r[key] - r["correct"]) ** 2 for r in valid) / n
    nll = -sum(
        r["correct"] * math.log(max(1e-6, min(1-1e-6, r[key])))
        + (1-r["correct"]) * math.log(max(1e-6, min(1-1e-6, 1-r[key])))
        for r in valid
    ) / n
    table = []
    ece = 0.0
    for i in range(bins):
        lo, hi = i / bins, (i+1) / bins
        group = [r for r in valid if lo <= r[key] < hi or (i == bins-1 and r[key] == 1)]
        count = len(group)
        mean_q = sum(r[key] for r in group) / count if count else None
        accuracy = sum(r["correct"] for r in group) / count if count else None
        if count:
            ece += (count/n) * abs(mean_q-accuracy)
        table.append({"lower": lo, "upper": hi, "count": count,
                      "mean_confidence": mean_q, "accuracy": accuracy})
    curve = []
    for threshold in CONFIDENCES:
        accepted = [r for r in valid if r[key] >= threshold-1e-9]
        curve.append({
            "threshold": threshold,
            "coverage_all": len(accepted) / len(rows),
            "selective_accuracy": sum(r["correct"] for r in accepted) / len(accepted) if accepted else None,
            "accepted": len(accepted),
        })
    return {"count": n, "brier": brier, "ece": ece, "nll": nll, "table": table, "risk_coverage": curve}


def summarize(rows):
    n = len(rows)
    if not n:
        raise ValueError("No predictions")
    result = {
        "count": n,
        "accuracy": sum(r["correct"] for r in rows) / n,
        "invalid_format_rate": sum(not r["valid_format"] for r in rows) / n,
        "explicit": calibration_metrics(rows),
    }
    native_rows = [
        {"correct": r["native_correct"], "confidence": r["native_confidence"]}
        for r in rows if r.get("native_confidence") is not None
    ]
    if native_rows:
        result["native_accuracy"] = sum(r["correct"] for r in native_rows) / len(native_rows)
        result["native"] = calibration_metrics(native_rows)
    return result
