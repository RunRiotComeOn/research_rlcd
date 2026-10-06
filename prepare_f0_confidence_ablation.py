"""Split F0-unseen JeV fold1 by video for confidence ablations."""
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path('/pfs/hyx/videojev-rlcd')
SOURCE = ROOT / 'data/jev_full_holmes_v1/folds/fold1.jsonl'
ACTIONS = ROOT / 'calibration/jevfull_oof_v1/fold1/predictions.jsonl'
OUT = ROOT / 'data/f0_confidence_ablation_v1'
SEED = 20261007


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def write_jsonl(path, rows):
    with path.open('w', encoding='utf-8') as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + '\n')


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    rows = [json.loads(s) for s in SOURCE.read_text(encoding='utf-8').splitlines()]
    actions = [json.loads(s) for s in ACTIONS.read_text(encoding='utf-8').splitlines()]
    assert len(rows) == len(actions) == 24063
    assert [r['id'] for r in rows] == [r['id'] for r in actions]
    assert all(a['source_answer_fold'] == 'fold0' for a in actions)
    grouped = defaultdict(list)
    for row in rows:
        grouped[Path(row['video_path']).name].append(row)
    names = list(grouped)
    random.Random(SEED).shuffle(names)
    selected = {'dev': set(), 'test': set()}
    remaining = {'dev': 2000, 'test': 2000}
    for split in ('dev', 'test'):
        for name in names:
            if name in selected['dev'] or name in selected['test']:
                continue
            count = len(grouped[name])
            if count <= remaining[split]:
                selected[split].add(name)
                remaining[split] -= count
            if remaining[split] == 0:
                break
        if remaining[split] != 0:
            raise RuntimeError(f'Could not fill {split} to 2,000 rows')
    selected['train'] = set(names) - selected['dev'] - selected['test']
    split_rows = {name: [] for name in ('train', 'dev', 'test')}
    split_actions = {name: [] for name in ('train', 'dev', 'test')}
    for row, action in zip(rows, actions):
        name = Path(row['video_path']).name
        split = next(s for s in split_rows if name in selected[s])
        split_rows[split].append(row)
        split_actions[split].append(action)
    assert [len(split_rows[s]) for s in ('train', 'dev', 'test')] == [20063, 2000, 2000]
    assert all(not selected[a] & selected[b] for a,b in
               [('train','dev'), ('train','test'), ('dev','test')])
    OUT.mkdir(parents=True)
    manifest = {'seed': SEED, 'source_data': str(SOURCE), 'source_data_sha256': sha256(SOURCE),
                'source_actions': str(ACTIONS), 'source_actions_sha256': sha256(ACTIONS),
                'answer_model': 'runs/jevfull_action_fold0_v1/checkpoints/examples_24063',
                'video_filename_overlap': 0, 'splits': {}}
    for split in ('train', 'dev', 'test'):
        data_path = OUT / f'{split}.jsonl'
        action_path = OUT / f'{split}_actions.jsonl'
        write_jsonl(data_path, split_rows[split])
        write_jsonl(action_path, split_actions[split])
        manifest['splits'][split] = {
            'rows': len(split_rows[split]), 'videos': len(selected[split]),
            'accuracy': sum(r['correct'] for r in split_actions[split])/len(split_actions[split]),
            'data_sources': dict(Counter(r['data_source'] for r in split_rows[split])),
            'data_sha256': sha256(data_path), 'actions_sha256': sha256(action_path)}
    (OUT / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
