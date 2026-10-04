"""Audit the frozen JeV holdout and compare fixed vs scheduled reward."""
import hashlib
import json
from pathlib import Path

from analyze_jev10_reward_compare import paired, records

ROOT = Path('/pfs/hyx/videojev-rlcd')
DATA = ROOT / 'data/jev10_schedule_blind_v1'
EVAL = ROOT / 'evaluations/jev10_schedule_blind_v1'
OUT = ROOT / 'comparisons/jev10_reward_schedule_blind_v1'


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    manifest = json.loads((DATA / 'manifest.json').read_text())
    if manifest['rows'] != 1000 or manifest['output_sha256'] != sha256(DATA / 'blind.jsonl'):
        raise ValueError('Holdout changed')
    a, b = (EVAL / name for name in ('fixed_step_0300', 'schedule_step_0300'))
    for folder in (a, b):
        if (folder / 'STATUS.txt').read_text().strip() != 'COMPLETED 1000':
            raise ValueError(f'Evaluation incomplete: {folder}')
    old = {r['id']: r for r in records(a / 'predictions.jsonl')}
    new = {r['id']: r for r in records(b / 'predictions.jsonl')}
    if len(old) != 1000 or set(old) != set(new):
        raise ValueError('Holdout IDs mismatch')
    for key in old:
        if (old[key]['video_path'], old[key]['gold_action'], old[key]['data_source']) != (
                new[key]['video_path'], new[key]['gold_action'], new[key]['data_source']):
            raise ValueError(f'Metadata mismatch: {key}')
    ca = json.loads((a / 'run_config.json').read_text())
    cb = json.loads((b / 'run_config.json').read_text())
    for config, expected_run in ((ca, 'jev10_reward_proposed_v2_300'),
                                 (cb, 'jev10_reward_proposed_schedule_v1_300')):
        if config['data_sha256'] != manifest['output_sha256']:
            raise ValueError('Data hash mismatch')
        if expected_run not in config['adapter'] or config['step'] != 300:
            raise ValueError('Wrong adapter or step')
        if sha256(Path(config['adapter']) / 'adapter_model.safetensors') != config['adapter_weights_sha256']:
            raise ValueError('Adapter changed')
    sa = json.loads((a / 'summary.json').read_text())
    sb = json.loads((b / 'summary.json').read_text())
    result = {
        'protocol': str(ROOT / 'JEV10_REWARD_SCHEDULE_BLIND_PROTOCOL.md'),
        'dataset_sha256': manifest['output_sha256'],
        'video_paths': manifest['video_paths'],
        'rows': 1000,
        'fixed': {'correct': sa['correct'], 'accuracy': sa['accuracy'],
                  'direct_brier': sa['direct']['brier'],
                  'mean_q': sa['mean_reported_confidence'],
                  'q9_count': sa['confidence_bins'].get('9', 0)},
        'scheduled': {'correct': sb['correct'], 'accuracy': sb['accuracy'],
                      'direct_brier': sb['direct']['brier'],
                      'mean_q': sb['mean_reported_confidence'],
                      'q9_count': sb['confidence_bins'].get('9', 0)},
        'scheduled_minus_fixed_accuracy': paired(old, new, 'accuracy', 20261006),
        'scheduled_minus_fixed_brier': paired(old, new, 'brier', 20261007),
        'gained_questions': sum(old[k]['correct'] == 0 and new[k]['correct'] == 1 for k in old),
        'lost_questions': sum(old[k]['correct'] == 1 and new[k]['correct'] == 0 for k in old),
        'fixed_predictions_sha256': sha256(a / 'predictions.jsonl'),
        'scheduled_predictions_sha256': sha256(b / 'predictions.jsonl'),
    }
    OUT.mkdir(parents=True)
    (OUT / 'summary.json').write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n')
    (OUT / 'STATUS.txt').write_text('COMPLETED BLIND 1000\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
