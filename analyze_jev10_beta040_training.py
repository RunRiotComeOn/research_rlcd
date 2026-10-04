"""Compare the matched beta=.2 and beta=.4 training trajectories."""
import json
from pathlib import Path

from analyze_jev10_reward_dynamics import Q, stage

ROOT = Path(__file__).resolve().parent


def main():
    result = {}
    for name, beta, file in (
        ('beta020', .2, 'reward_new_metrics.jsonl'),
        ('beta040', .4, 'reward_beta040_metrics.jsonl'),
    ):
        rows = [json.loads(line) for line in (ROOT / file).read_text(encoding='utf-8').splitlines()]
        if len(rows) != 300 or [row['step'] for row in rows] != list(range(1, 301)):
            raise ValueError(f'{name}: incomplete training log')
        for row in rows:
            for y, digit, reward in zip(row['correct'], row['sampled_digits'], row['rewards']):
                q = Q[digit]
                if abs(reward - (y + beta * (2 * y * q - q * q))) > 1e-8:
                    raise ValueError(f'{name}: wrong reward at step {row["step"]}')
        result[name] = {
            'beta': beta,
            'whole_run': stage(rows),
            'windows_25': [stage(rows[i:i + 25]) for i in range(0, 300, 25)],
            'windows_100': [stage(rows[i:i + 100]) for i in range(0, 300, 100)],
        }
    out = ROOT / 'jev10_reward_beta040_training.json'
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    for name, item in result.items():
        print(name)
        for row in item['windows_100']:
            print(json.dumps({
                'steps': row['steps'],
                'sampled_correct_rate': row['sampled_correct_rate'],
                'mean_q': row['mean_q'],
                'sampled_brier': row['sampled_binary_brier'],
                'q9': row['q_bins']['9'],
                'updates': row['updates'],
                'same_y_updates': row['group_types'].get('same_y_confidence_only_updates', 0),
                'action_entropy': row['mean_action_entropy'],
                'digit_entropy': row['mean_digit_entropy'],
            }))


if __name__ == '__main__':
    main()
