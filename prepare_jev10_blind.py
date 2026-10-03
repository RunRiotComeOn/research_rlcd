"""Freeze a fresh, source-stratified, video-disjoint JeV blind set."""
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from prepare_data import TRAIN_SOURCE, convert, write_jsonl

ROOT = Path('/pfs/hyx/videojev-rlcd')
OLD = ROOT / 'data/jev10_v2'
OUT = ROOT / 'data/jev10_blind_v1'
SEED = 20261003
TARGET = 2000


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def video(row):
    return (row.get('videos') or [row.get('path')])[0]


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    source = json.loads(TRAIN_SOURCE.read_text(encoding='utf-8'))
    prior = [json.loads(s) for split in ('train', 'validation', 'test')
             for s in (OLD / f'{split}.jsonl').read_text(encoding='utf-8').splitlines()]
    used_paths = {r['video_path'] for r in prior}
    used_names = {Path(p).name for p in used_paths}
    used_ids = {r['id'].split('-', 1)[1] for r in prior}
    groups = defaultdict(list)
    for row in source:
        path = video(row)
        if path not in used_paths and Path(path).name not in used_names and str(row['problem_id']) not in used_ids:
            groups[path].append(row)
    mixed = {p for p, group in groups.items()
             if len({r.get('data_source') for r in group}) != 1}
    if mixed:
        raise ValueError(f'Mixed-source video groups: {len(mixed)}')
    by_source = defaultdict(list)
    for group in groups.values():
        by_source[group[0].get('data_source')].append(group)
    available_counts = Counter(r.get('data_source') for g in groups.values() for r in g)
    available_total = sum(available_counts.values())
    exact = {s: TARGET*n/available_total for s, n in available_counts.items()}
    quota = {s: int(v) for s, v in exact.items()}
    for s in sorted(exact, key=lambda k: (-(exact[k]-quota[k]), str(k)))[:TARGET-sum(quota.values())]:
        quota[s] += 1
    rng = random.Random(SEED)
    selected = []
    remainder = []
    for s in sorted(by_source, key=str):
        candidates = by_source[s]
        rng.shuffle(candidates)
        count = 0
        for group in candidates:
            if count + len(group) <= quota[s]:
                selected.append(group)
                count += len(group)
            else:
                remainder.append(group)
    rng.shuffle(remainder)
    count = sum(map(len, selected))
    for group in remainder:
        if count + len(group) <= TARGET:
            selected.append(group)
            count += len(group)
        if count == TARGET:
            break
    if count != TARGET:
        raise ValueError(f'Could select only {count} rows')
    raw = [r for group in selected for r in group]
    rng.shuffle(raw)
    converted = [convert(r, 'blind') for r in raw]
    if len({r['id'] for r in converted}) != TARGET:
        raise ValueError('Duplicate blind IDs')
    blind_paths = {r['video_path'] for r in converted}
    if blind_paths & used_paths or {Path(p).name for p in blind_paths} & used_names:
        raise ValueError('Prior video overlap')
    OUT.mkdir(parents=True)
    output = OUT / 'blind.jsonl'
    write_jsonl(output, converted)
    manifest = {
        'source': str(TRAIN_SOURCE), 'source_sha256': sha256(TRAIN_SOURCE),
        'prior_split_manifest_sha256': sha256(OLD / 'manifest.json'),
        'seed': SEED, 'target_rows': TARGET,
        'sampling': 'Video groups by data_source; Hamilton source quotas based on unused pool; seeded shuffle and greedy quota selection; seeded remainder fill.',
        'excluded_prior_rows': len(prior), 'excluded_prior_video_paths': len(used_paths),
        'available_rows': available_total, 'rows': len(converted),
        'video_paths': len(blind_paths), 'problem_id_overlap': 0,
        'video_path_overlap': 0, 'video_filename_overlap': 0,
        'source_quota': dict(sorted(quota.items())),
        'source_counts': dict(sorted(Counter(r['data_source'] for r in converted).items())),
        'output': str(output), 'output_sha256': sha256(output),
    }
    (OUT / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({k: manifest[k] for k in ('rows', 'video_paths', 'available_rows', 'output_sha256')}, indent=2))


if __name__ == '__main__':
    main()
