"""Make deterministic, video-disjoint two-fold cross-fitting data."""
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path

from prepare_data import ROOT, write_jsonl

SOURCE = ROOT / 'data/jev_full_holmes_v1/jev_all.jsonl'
OUT = ROOT / 'data/jev_full_holmes_v1/folds'
SEED = 20261005


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    rows = [json.loads(s) for s in SOURCE.read_text(encoding='utf-8').splitlines()]
    if len(rows) != 48126:
        raise ValueError(f'Expected 48126 rows, got {len(rows)}')
    groups = defaultdict(list)
    for row in rows:
        groups[Path(row['video_path']).name].append(row)
    names = sorted(groups)
    random.Random(SEED).shuffle(names)
    target = len(rows) // 2
    fold0, fold1 = [], []
    for name in names:
        group = groups[name]
        if len(fold0) + len(group) <= target:
            fold0.extend(group)
        else:
            fold1.extend(group)
    if len(fold0) != target or len(fold1) != target:
        raise ValueError(f'Could not balance folds: {len(fold0)}, {len(fold1)}')
    rng = random.Random(SEED + 1)
    rng.shuffle(fold0)
    rng.shuffle(fold1)
    if {r['id'] for r in fold0} & {r['id'] for r in fold1}:
        raise ValueError('Question ID overlap')
    if {Path(r['video_path']).name for r in fold0} & {Path(r['video_path']).name for r in fold1}:
        raise ValueError('Video overlap')
    OUT.mkdir(parents=True)
    manifest = {'seed': SEED, 'source': str(SOURCE), 'source_sha256': sha256(SOURCE),
                'cross_fold_question_id_overlap': 0,
                'cross_fold_video_filename_overlap': 0, 'folds': {}}
    for name, group in [('fold0', fold0), ('fold1', fold1)]:
        path = OUT / f'{name}.jsonl'
        write_jsonl(path, group)
        manifest['folds'][name] = {'rows': len(group),
                                   'video_filenames': len({Path(r['video_path']).name for r in group}),
                                   'output': str(path), 'output_sha256': sha256(path)}
    (OUT / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({k:v['rows'] for k,v in manifest['folds'].items()}))


if __name__ == '__main__':
    main()
