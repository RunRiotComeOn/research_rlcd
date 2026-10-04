"""Matched fixed-versus-scheduled confidence-reward validation comparison."""
import hashlib
import json
from pathlib import Path

from analyze_jev10_reward_compare import paired, records

ROOT = Path('/pfs/hyx/videojev-rlcd')
EVAL = ROOT / 'evaluations/jev10_reward_compare_v2'
RUN = ROOT / 'runs/jev10_reward_proposed_schedule_v1_300'
FIXED = ROOT / 'runs/jev10_reward_proposed_v2_300'
OUT = ROOT / 'comparisons/jev10_reward_schedule_v1'


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    if (RUN / 'STATUS.txt').read_text().strip() != 'COMPLETED 300':
        raise ValueError('Training incomplete')
    scheduled = json.loads((RUN / 'run_config.json').read_text())
    fixed = json.loads((FIXED / 'run_config.json').read_text())
    if scheduled['mode'] != 'proposed_schedule' or fixed['mode'] != 'proposed':
        raise ValueError('Wrong run modes')
    if scheduled['beta'] != fixed['beta'] or scheduled['beta'] != .2:
        raise ValueError('Wrong beta')
    for key in ('seed', 'source_adapter_weights_sha256', 'data_sha256', 'limit',
                'num_rollouts', 'action_sampling_temperature',
                'digit_sampling_temperature', 'learning_rate', 'objective'):
        if scheduled[key] != fixed[key]:
            raise ValueError(f'Mismatched setting: {key}')
    result = {'protocol': str(ROOT / 'JEV10_REWARD_SCHEDULE_PROTOCOL.md'),
              'fixed_config_sha256': sha256(FIXED / 'run_config.json'),
              'scheduled_config_sha256': sha256(RUN / 'run_config.json'),
              'rows': 481, 'steps': {}}
    for step in (100, 200, 300):
        a = EVAL / f'proposed_step_{step:04d}'
        b = EVAL / f'schedule_step_{step:04d}'
        for folder in (a, b):
            if (folder / 'STATUS.txt').read_text().strip() != 'COMPLETED 481':
                raise ValueError(f'Evaluation incomplete: {folder}')
        old = {r['id']: r for r in records(a / 'predictions.jsonl')}
        new = {r['id']: r for r in records(b / 'predictions.jsonl')}
        if len(old) != 481 or set(old) != set(new):
            raise ValueError(f'IDs mismatch at {step}')
        for key in old:
            if (old[key]['video_path'], old[key]['gold_action']) != (new[key]['video_path'], new[key]['gold_action']):
                raise ValueError(f'Metadata mismatch at {step}')
        old_summary = json.loads((a / 'summary.json').read_text())
        new_summary = json.loads((b / 'summary.json').read_text())
        result['steps'][str(step)] = {
            'fixed': {'correct': old_summary['correct'], 'accuracy': old_summary['accuracy'],
                      'direct_brier': old_summary['direct']['brier'],
                      'mean_q': old_summary['mean_reported_confidence'],
                      'q9_count': old_summary['confidence_bins'].get('9', 0)},
            'scheduled': {'correct': new_summary['correct'], 'accuracy': new_summary['accuracy'],
                          'direct_brier': new_summary['direct']['brier'],
                          'mean_q': new_summary['mean_reported_confidence'],
                          'q9_count': new_summary['confidence_bins'].get('9', 0)},
            'scheduled_minus_fixed_accuracy': paired(old, new, 'accuracy', 20261006 + step),
            'scheduled_minus_fixed_brier': paired(old, new, 'brier', 20261007 + step),
            'gained_questions': sum(old[k]['correct'] == 0 and new[k]['correct'] == 1 for k in old),
            'lost_questions': sum(old[k]['correct'] == 1 and new[k]['correct'] == 0 for k in old),
            'fixed_predictions_sha256': sha256(a / 'predictions.jsonl'),
            'scheduled_predictions_sha256': sha256(b / 'predictions.jsonl'),
        }
    OUT.mkdir(parents=True)
    (OUT / 'validation.json').write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n')
    (OUT / 'STATUS.txt').write_text('COMPLETED VALIDATION\n')
    print(json.dumps(result['steps'], indent=2))


if __name__ == '__main__':
    main()
