"""Audit same-action calibration gains for an independent confidence LoRA."""
import hashlib
import json
from pathlib import Path

from analyze_jev10_reward_compare import paired, records

ROOT = Path('/pfs/hyx/videojev-rlcd')
EVAL = ROOT / 'evaluations/jev10_frozen_action_meanq_v1'
BASELINE = ROOT / 'evaluations/jev10_reward_compare_v2/baseline/predictions.jsonl'
RUN = ROOT / 'runs/jev10_frozen_action_meanq_v1_300'
OUT = ROOT / 'comparisons/jev10_frozen_action_meanq_validation_v1'


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def score(group):
    return {'correct': sum(r['correct'] for r in group.values()),
            'accuracy': sum(r['correct'] for r in group.values()) / len(group),
            'brier': sum((r['confidence'] - r['correct']) ** 2 for r in group.values()) / len(group),
            'mean_q': sum(r['confidence'] for r in group.values()) / len(group)}


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    if (RUN / 'STATUS.txt').read_text().strip() != 'COMPLETED 300':
        raise ValueError('Training incomplete')
    config = json.loads((RUN / 'run_config.json').read_text())
    if not config['action_policy_frozen'] or not config['separate_confidence_model']:
        raise ValueError('Wrong training configuration')
    base = {r['id']: {**r, 'confidence': r['expected_q']} for r in records(BASELINE)}
    if len(base) != 481:
        raise ValueError('Incomplete baseline')
    result = {'protocol': str(ROOT / 'JEV10_FROZEN_ACTION_MEANQ_PROTOCOL.md'),
              'training_config_sha256': sha256(RUN / 'run_config.json'),
              'baseline_predictions_sha256': sha256(BASELINE),
              'rows': 481, 'baseline_mean_q': score(base), 'steps': {}}
    for step in (100, 200, 300):
        folder = EVAL / f'step_{step:04d}'
        if (folder / 'STATUS.txt').read_text().strip() != 'COMPLETED 481':
            raise ValueError(f'Incomplete evaluation: {folder}')
        new = {r['id']: r for r in records(folder / 'predictions.jsonl')}
        if len(new) != 481 or set(new) != set(base):
            raise ValueError(f'IDs mismatch at {step}')
        for key, row in new.items():
            old = base[key]
            if (old['video_path'], old['gold_action'], old['predicted_action'], old['correct']) != (
                    row['video_path'], row['gold_action'], row['predicted_action'], row['correct']):
                raise ValueError(f'Action or metadata mismatch at {step}: {key}')
        result['steps'][str(step)] = {
            'mean_q': score(new),
            'rounded_percent_brier': sum((r['confidence_rounded'] - r['correct']) ** 2
                                         for r in new.values()) / 481,
            'minus_baseline_mean_q_brier': paired(base, new, 'brier', 20261014 + step),
            'predictions_sha256': sha256(folder / 'predictions.jsonl'),
        }
    OUT.mkdir(parents=True)
    (OUT / 'validation.json').write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n')
    (OUT / 'STATUS.txt').write_text('COMPLETED VALIDATION\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
