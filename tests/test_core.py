import math
import pytest

from core import parse_output, reward, assert_correctness_margin, calibration_metrics


def test_strict_parser_and_choice_range():
    result = parse_output("Action: B\nConfidence: 8", 4)
    assert result.valid_format and result.action_index == 1
    assert result.confidence_bin == 8 and result.confidence == pytest.approx(0.85)
    for raw in ("Action:B\nConfidence:8", "Action: Z\nConfidence: 8", "B\n8", "Action: B\nConfidence: 10"):
        assert not parse_output(raw, 4).valid_format


def test_reward_margin_and_invalid_penalty():
    assert assert_correctness_margin(0.2) > 0
    correct = [reward(parse_output(f"Action: A\nConfidence: {i}", 2), 0) for i in range(10)]
    wrong = [reward(parse_output(f"Action: B\nConfidence: {i}", 2), 0) for i in range(10)]
    assert min(correct) > max(wrong)
    assert reward(parse_output("junk", 2), 0) == -1.0
    with pytest.raises(ValueError):
        assert_correctness_margin(2.0)


def test_calibration_metrics_known_case():
    rows = [{"confidence": 0.1, "correct": int(i == 0)} for i in range(10)]
    rows += [{"confidence": 0.9, "correct": int(i != 0)} for i in range(10)]
    metrics = calibration_metrics(rows)
    assert metrics["count"] == 20
    assert metrics["ece"] == pytest.approx(0, abs=1e-12)
    assert metrics["brier"] == pytest.approx(0.09)
    expected = -(0.1*math.log(0.1) + 0.9*math.log(0.9))
    assert metrics["nll"] == pytest.approx(expected)
