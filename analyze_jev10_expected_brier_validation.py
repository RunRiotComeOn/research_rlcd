"""Compare exact-Brier training against fixed and scheduled reward pilots."""
import hashlib
import json
from pathlib import Path

from analyze_jev10_reward_compare import paired, records
from core import CONFIDENCES

ROOT = Path('/pfs/hyx/videojev-rlcd')
EVAL = ROOT / 'evaluations/jev10_reward_compare_v2'
RUN = ROOT / 'runs/jev10_expected_brier_v1_300'
OUT = ROOT / 'comparisons/jev10_expected_brier_validation_v1'


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def dist_brier(row):
    return sum(p * (q - row['correct']) ** 2
               for p, q in zip(row['digit_probabilities'], CONFIDENCES))


def summary(rows):
    return {'correct': sum(r['correct'] for r in rows.values()),
            'accuracy': sum(r['correct'] for r in rows.values()) / len(rows),
            'direct_brier': sum((r['confidence'] - r['correct']) ** 2 for r in rows.values()) / len(rows),
            'mean_q_brier': sum((r['expected_q'] - r['correct']) ** 2 for r in rows.values()) / len(rows),
            'distribution_expected_brier': sum(dist_brier(r) for r in rows.values()) / len(rows),
            'mean_q': sum(r['confidence'] for r in rows.values()) / len(rows),
            'q9_count': sum(r['confidence_bin'] == 9 for r in rows.values())}


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    if (RUN / 'STATUS.txt').read_text().strip() != 'COMPLETED 300':
        raise ValueError('Training incomplete')
    cfg = json.loads((RUN / 'run_config.json').read_text())
    if cfg['beta'] != .2 or cfg['limit'] != 300 or not cfg['greedy_action_for_confidence']:
        raise ValueError('Wrong training configuration')
    result = {'protocol': str(ROOT / 'JEV10_EXPECTED_BRIER_PROTOCOL.md'),
              'run_config_sha256': sha256(RUN / 'run_config.json'),
              'rows': 481, 'steps': {}}
    for step in (100, 200, 300):
        groups = {}
        for key, name in (('fixed', f'proposed_step_{step:04d}'),
                          ('schedule', f'schedule_step_{step:04d}'),
                          ('expected_brier', f'expected_brier_step_{step:04d}')):
            folder = EVAL / name
            if (folder / 'STATUS.txt').read_text().strip() != 'COMPLETED 481':
                raise ValueError(f'Incomplete: {folder}')
            groups[key] = {r['id']: r for r in records(folder / 'predictions.jsonl')}
        ids = set(groups['fixed'])
        if len(ids) != 481 or any(set(g) != ids for g in groups.values()):
            raise ValueError(f'ID mismatch at {step}')
        for key in ids:
            a = groups['fixed'][key]
            for g in groups.values():
                b = g[key]
                if (a['video_path'], a['gold_action'], a['data_source']) != (
                        b['video_path'], b['gold_action'], b['data_source']):
                    raise ValueError(f'Metadata mismatch at {step}: {key}')
        new = groups['expected_brier']
        result['steps'][str(step)] = {
            name: summary(group) for name, group in groups.items()}
        for control in ('fixed', 'schedule'):
            old = groups[control]
            result['steps'][str(step)][f'expected_minus_{control}_accuracy'] = paired(
                old, new, 'accuracy', 20261010 + step)
            result['steps'][str(step)][f'expected_minus_{control}_direct_brier'] = paired(
                old, new, 'brier', 20261011 + step)
    OUT.mkdir(parents=True)
    (OUT / 'validation.json').write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n')
    (OUT / 'STATUS.txt').write_text('COMPLETED VALIDATION\n')
    print(json.dumps(result['steps'], indent=2))


if __name__ == '__main__':
    main()
