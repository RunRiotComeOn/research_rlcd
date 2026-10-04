"""Audit exact-Brier training logs and summarize 100-step windows."""
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def mean(values):
    return sum(values) / len(values)


def window(rows):
    bins = Counter(r['confidence_argmax_bin'] for r in rows)
    all_same = [r for r in rows if not r['action_advantage_nonzero']]
    return {
        'steps': [rows[0]['step'], rows[-1]['step']],
        'sampled_action_correct_rate': mean([y for r in rows for y in r['sampled_correct']]),
        'greedy_action_correct_rate': mean([r['greedy_correct'] for r in rows]),
        'action_updates': sum(r['action_advantage_nonzero'] for r in rows),
        'confidence_updates_when_action_zero': sum(r['grad_norm'] > 0 for r in all_same),
        'action_zero_groups': len(all_same),
        'mean_training_distribution_brier': mean([r['confidence_expected_brier'] for r in rows]),
        'mean_confidence_q': mean([r['confidence_mean_q'] for r in rows]),
        'mean_confidence_entropy': mean([r['confidence_entropy'] for r in rows]),
        'mean_action_entropy': mean([r['action_entropy'] for r in rows]),
        'argmax_confidence_bins': {str(i): bins[i] for i in range(10)},
    }


def main():
    rows = [json.loads(line) for line in (ROOT / 'expected_brier_metrics.jsonl').read_text().splitlines()]
    if len(rows) != 300 or [r['step'] for r in rows] != list(range(1, 301)):
        raise ValueError('Incomplete training log')
    if any('sampled_digits' in r or 'rewards' in r for r in rows):
        raise ValueError('Unexpected sampled confidence reward')
    result = {'rows': len(rows),
              'windows_100': [window(rows[i:i + 100]) for i in range(0, 300, 100)],
              'whole_run': window(rows)}
    (ROOT / 'jev10_expected_brier_training.json').write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print(json.dumps(result['windows_100'], indent=2))


if __name__ == '__main__':
    main()
