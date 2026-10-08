"""Reserve a fresh video-disjoint JeV test from the former F0-confidence train pool."""
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path('/pfs/hyx/videojev-rlcd')
SOURCE = ROOT / 'data/f0_confidence_ablation_v1/train.jsonl'
ACTIONS = ROOT / 'data/f0_confidence_ablation_v1/train_actions.jsonl'
OUT = ROOT / 'data/f0_joint_confidence_v1'
SEED = 20261008


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
    assert len(rows) == len(actions) == 20063
    assert [r['id'] for r in rows] == [a['id'] for a in actions]
    assert all(a['source_answer_fold'] == 'fold0' for a in actions)
    groups = defaultdict(list)
    for row in rows:
        groups[Path(row['video_path']).name].append(row)
    keys = list(groups)
    random.Random(SEED).shuffle(keys)
    selected = {'dev': set(), 'test': set()}
    for split in ('dev', 'test'):
        remaining = 2000
        for key in keys:
            if key in selected['dev'] or key in selected['test']:
                continue
            count = len(groups[key])
            if count <= remaining:
                selected[split].add(key)
                remaining -= count
            if remaining == 0:
                break
        if remaining:
            raise RuntimeError(f'Could not fill {split} exactly')
    selected['train'] = set(keys)-selected['dev']-selected['test']
    partitions = {s: [] for s in ('train','dev','test')}
    predictions = {s: [] for s in partitions}
    for row, action in zip(rows, actions):
        key = Path(row['video_path']).name
        split = next(s for s in partitions if key in selected[s])
        partitions[split].append(row)
        predictions[split].append(action)
    assert [len(partitions[s]) for s in ('train','dev','test')] == [16063,2000,2000]
    assert all(not selected[a]&selected[b] for a,b in
               (('train','dev'),('train','test'),('dev','test')))
    OUT.mkdir(parents=True)
    manifest = {'seed': SEED, 'source': str(SOURCE), 'source_sha256': sha256(SOURCE),
                'source_actions': str(ACTIONS), 'source_actions_sha256': sha256(ACTIONS),
                'answer_adapter': 'runs/jevfull_action_fold0_v1/checkpoints/examples_24063',
                'pairwise_video_filename_overlap': 0, 'splits': {}}
    for split in partitions:
        data_path = OUT / f'{split}.jsonl'
        action_path = OUT / f'{split}_actions.jsonl'
        write_jsonl(data_path, partitions[split])
        write_jsonl(action_path, predictions[split])
        manifest['splits'][split] = {
            'rows': len(partitions[split]), 'videos': len(selected[split]),
            'f0_accuracy': sum(a['correct'] for a in predictions[split])/len(predictions[split]),
            'data_sources': dict(Counter(r['data_source'] for r in partitions[split])),
            'data_sha256': sha256(data_path), 'actions_sha256': sha256(action_path)}
    (OUT / 'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
