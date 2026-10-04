"""Summarize confidence-only training without exposing question rows."""
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def avg(values):
    return sum(values) / len(values)


def window(rows):
    bins = Counter(r['argmax_bin'] for r in rows)
    return {'steps': [rows[0]['step'], rows[-1]['step']],
            'rows': len(rows), 'greedy_action_correct_rate': avg([r['correct'] for r in rows]),
            'mean_train_brier': avg([r['brier'] for r in rows]),
            'mean_q': avg([r['mean_q'] for r in rows]),
            'mean_digit_entropy': avg([r['digit_entropy'] for r in rows]),
            'mean_action_entropy': avg([r['action_entropy'] for r in rows]),
            'mean_grad_norm': avg([r['grad_norm'] for r in rows]),
            'argmax_bins': {str(i): bins[i] for i in range(10)}}


def main():
    rows = [json.loads(line) for line in (ROOT / 'frozen_action_meanq_metrics.jsonl').read_text().splitlines()]
    if len(rows) != 300 or [r['step'] for r in rows] != list(range(1, 301)):
        raise ValueError('Incomplete training log')
    result = {'windows_100': [window(rows[i:i + 100]) for i in range(0, 300, 100)],
              'whole_run': window(rows)}
    (ROOT / 'jev10_frozen_action_meanq_training.json').write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print(json.dumps(result['windows_100'], indent=2))


if __name__ == '__main__':
    main()
