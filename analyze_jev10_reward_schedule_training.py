"""Audit scheduled reward and compare training dynamics with fixed beta=.2."""
import json
from pathlib import Path

from analyze_jev10_reward_dynamics import Q, stage

ROOT = Path(__file__).resolve().parent


def lam(step):
    return .2 if step <= 100 else .2 - .15 * (step - 100) / 200


def read(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]


def main():
    fixed = read(ROOT / 'reward_new_metrics.jsonl')
    scheduled = read(ROOT / 'reward_schedule_metrics.jsonl')
    for name, rows in (('fixed', fixed), ('scheduled', scheduled)):
        if len(rows) != 300 or [row['step'] for row in rows] != list(range(1, 301)):
            raise ValueError(f'{name}: incomplete or unordered training log')
    for row in scheduled:
        weight = lam(row['step'])
        if abs(row['lambda_confidence'] - weight) > 1e-12:
            raise ValueError(f'Wrong lambda at step {row["step"]}')
        for y, digit, reward in zip(row['correct'], row['sampled_digits'], row['rewards']):
            expected = 1.2 * y - weight * (Q[digit] - y) ** 2
            if abs(reward - expected) > 1e-8:
                raise ValueError(f'Wrong reward at step {row["step"]}')
    matched = 0
    for a, b in zip(fixed[:100], scheduled[:100]):
        if (a['id'], a['sampled_actions'], a['sampled_digits']) != (
                b['id'], b['sampled_actions'], b['sampled_digits']):
            break
        matched += 1
    result = {
        'matched_initial_steps': matched,
        'scheduled_lambda_1_100_200_300': [lam(step) for step in (1, 100, 200, 300)],
        'fixed': {'whole_run': stage(fixed),
                  'windows_25': [stage(fixed[i:i + 25]) for i in range(0, 300, 25)],
                  'windows_100': [stage(fixed[i:i + 100]) for i in range(0, 300, 100)]},
        'scheduled': {'whole_run': stage(scheduled),
                      'windows_25': [stage(scheduled[i:i + 25]) for i in range(0, 300, 25)],
                      'windows_100': [stage(scheduled[i:i + 100]) for i in range(0, 300, 100)]},
    }
    (ROOT / 'jev10_reward_schedule_training.json').write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print(json.dumps({'matched_initial_steps': matched,
                      'lambda': result['scheduled_lambda_1_100_200_300'],
                      'windows': {name: [
                          {'steps': w['steps'], 'sampled_correct_rate': w['sampled_correct_rate'],
                           'mean_q': w['mean_q'], 'sampled_brier': w['sampled_binary_brier'],
                           'q9': w['q_bins']['9'], 'updates': w['updates'],
                           'digit_entropy': w['mean_digit_entropy']}
                          for w in result[name]['windows_100']]
                          for name in ('fixed', 'scheduled')}}, indent=2))


if __name__ == '__main__':
    main()
