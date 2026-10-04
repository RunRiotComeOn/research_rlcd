"""Freeze video-disjoint fit, development and final confidence sets."""
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from prepare_data import TRAIN_SOURCE, convert, write_jsonl

ROOT = Path('/pfs/hyx/videojev-rlcd')
OUT = ROOT / 'data/jev10_unseen_calibration_v1'
SEED = 20261005
SIZES = {'final': 1000, 'development': 500, 'fit': 4000}
PRIOR = [ROOT / f'data/jev10_v2/{name}.jsonl' for name in ('train', 'validation', 'test')]
PRIOR += [ROOT / f'data/{name}/blind.jsonl' for name in
          ('jev10_blind_v1', 'jev10_reward_blind_v1', 'jev10_schedule_blind_v1')]


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def raw_video(row):
    return (row.get('videos') or [row.get('path')])[0]


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    source = json.loads(TRAIN_SOURCE.read_text(encoding='utf-8'))
    old = [json.loads(line) for path in PRIOR for line in path.read_text(encoding='utf-8').splitlines()]
    used_ids = {r['id'].split('-', 1)[1] for r in old}
    used_paths = {r['video_path'] for r in old}
    used_names = {Path(p).name for p in used_paths}
    groups = defaultdict(list)
    for row in source:
        path = raw_video(row)
        if str(row['problem_id']) not in used_ids and path not in used_paths and Path(path).name not in used_names:
            groups[Path(path).name].append(row)
    available = sum(map(len, groups.values()))
    if available < sum(SIZES.values()):
        raise ValueError(f'Only {available} eligible questions')
    rng = random.Random(SEED)
    names = sorted(groups)
    rng.shuffle(names)
    assigned = {}
    remaining = []
    for split, target in SIZES.items():
        chosen, count = [], 0
        for name in names:
            group = groups[name]
            if count + len(group) <= target:
                chosen.append(name)
                count += len(group)
                if count == target:
                    break
        if count != target:
            raise ValueError(f'Could select only {count}/{target} for {split}')
        selected = set(chosen)
        names = [name for name in names if name not in selected]
        raw = [row for name in chosen for row in groups[name]]
        rng.shuffle(raw)
        assigned[split] = [convert(row, f'unseen-{split}') for row in raw]
        remaining.append({'split': split, 'rows': count, 'video_names': len(chosen)})
    all_rows = [r for rows in assigned.values() for r in rows]
    all_ids = [r['id'][len(f'unseen-{split}-'):] for split, rows in assigned.items() for r in rows]
    all_names = [Path(r['video_path']).name for r in all_rows]
    if len(all_ids) != len(set(all_ids)) or set(all_ids) & used_ids:
        raise ValueError('Problem ID overlap')
    if set(all_names) & used_names:
        raise ValueError('Prior video overlap')
    if any(set(Path(r['video_path']).name for r in assigned[a]) &
           set(Path(r['video_path']).name for r in assigned[b])
           for a in assigned for b in assigned if a < b):
        raise ValueError('Cross-split video overlap')
    OUT.mkdir(parents=True)
    manifest = {'seed': SEED, 'source': str(TRAIN_SOURCE), 'source_sha256': sha256(TRAIN_SOURCE),
                'prior_sha256': {str(p): sha256(p) for p in PRIOR},
                'excluded_prior_rows': len(old), 'eligible_rows': available,
                'splits': {}}
    for split, rows in assigned.items():
        path = OUT / f'{split}.jsonl'
        write_jsonl(path, rows)
        manifest['splits'][split] = {'rows': len(rows),
                                     'video_filenames': len({Path(r['video_path']).name for r in rows}),
                                     'source_counts': dict(Counter(str(r['data_source']) for r in rows)),
                                     'path': str(path), 'sha256': sha256(path)}
    (OUT / 'manifest.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps({'eligible_rows': available, 'selected': remaining}, indent=2))


if __name__ == '__main__':
    main()
