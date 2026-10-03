"""Video-disjoint 10% JeV training plus same-source validation/test splits."""
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from prepare_data import TRAIN_SOURCE, convert, write_jsonl

ROOT = Path('/pfs/hyx/videojev-rlcd')
OUT = ROOT / 'data/jev10_v2'
SEED = 20261002


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for part in iter(lambda: f.read(1 << 20), b''):
            h.update(part)
    return h.hexdigest()


def choose(groups, count):
    picked = []
    total = 0
    leftovers = []
    for group in groups:
        if total + len(group) <= count:
            picked.append(group)
            total += len(group)
        else:
            leftovers.append(group)
    if total != count:
        raise ValueError(f'Could select only {total} of {count} rows')
    return picked, leftovers


def main():
    if OUT.exists():
        raise FileExistsError(f'Refusing to overwrite {OUT}')
    source = json.loads(TRAIN_SOURCE.read_text(encoding='utf-8'))
    by_video = defaultdict(list)
    for row in source:
        path = (row.get('videos') or [row.get('path')])[0]
        by_video[path].append(row)
    rng = random.Random(SEED)
    available = list(by_video.values())
    rng.shuffle(available)
    train_target = round(len(source) * 0.10)
    holdout_target = round(len(source) * 0.01)
    train, remaining = choose(available, train_target)
    validation, remaining = choose(remaining, holdout_target)
    test, remaining = choose(remaining, holdout_target)
    splits = {'train': train, 'validation': validation, 'test': test}
    ids = {}
    paths = {}
    outputs = {}
    OUT.mkdir(parents=True)
    for split, groups in splits.items():
        raw = [r for group in groups for r in group]
        rng.shuffle(raw)
        converted = [convert(r, split) for r in raw]
        file = OUT / f'{split}.jsonl'
        write_jsonl(file, converted)
        ids[split] = {r['problem_id'] for r in raw}
        paths[split] = {r['video_path'] for r in converted}
        outputs[split] = {
            'rows': len(converted), 'video_paths': len(paths[split]),
            'data_source': dict(sorted(Counter(r['data_source'] for r in converted).items())),
            'sha256': sha256(file), 'file': str(file),
        }
    for a, b in (('train', 'validation'), ('train', 'test'), ('validation', 'test')):
        if ids[a] & ids[b] or paths[a] & paths[b]:
            raise ValueError(f'Split overlap: {a}/{b}')
    manifest = {
        'source': str(TRAIN_SOURCE), 'source_sha256': sha256(TRAIN_SOURCE),
        'source_rows': len(source), 'seed': SEED,
        'policy': 'Random whole-video groups; exact 10% train and 1% each validation/test; video and row IDs disjoint; fresh Qwen base for the experiment.',
        'splits': outputs,
    }
    (OUT / 'manifest.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps({k: v['rows'] for k, v in outputs.items()}))


if __name__ == '__main__':
    main()
