"""Inspect step-300 argmax confidence errors without exporting individual rows."""
import json
from collections import defaultdict
from pathlib import Path

from core import CONFIDENCES

ROOT = Path(__file__).resolve().parent


def main():
    rows = [json.loads(line) for line in (ROOT / 'expected_brier_step300_predictions.jsonl').read_text().splitlines()]
    if len(rows) != 481:
        raise ValueError('Wrong row count')
    groups = defaultdict(list)
    for row in rows:
        groups[row['confidence_bin']].append(row)
    result = {}
    for digit in range(10):
        group = groups[digit]
        if not group:
            continue
        result[str(digit)] = {
            'rows': len(group),
            'correct': sum(r['correct'] for r in group),
            'accuracy': sum(r['correct'] for r in group) / len(group),
            'argmax_q': CONFIDENCES[digit],
            'mean_softmax_q': sum(r['expected_q'] for r in group) / len(group),
            'mean_confidence_variance': sum(r['confidence_variance'] for r in group) / len(group),
            'direct_brier': sum((r['confidence']-r['correct'])**2 for r in group) / len(group),
        }
    (ROOT / 'jev10_expected_brier_bins.json').write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
