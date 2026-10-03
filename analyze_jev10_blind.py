"""Unblind only after both frozen JeV evaluations have completed."""
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path('/pfs/hyx/videojev-rlcd')
DATA = ROOT / 'data/jev10_blind_v1'
EVAL = ROOT / 'evaluations/jev10_blind_v1'
OUT = ROOT / 'comparisons/jev10_blind_v1'
N_BOOT = 10000
SEED = 20261003


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def read_rows(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]


def interval(values):
    values.sort()
    return [values[int(0.025*len(values))], values[int(0.975*len(values))]]


def brier(row):
    return (row['percent']/100 - row['correct'])**2


def summary(rows):
    n = len(rows)
    by_source = defaultdict(list)
    for row in rows:
        by_source[row['data_source']].append(row)
    return {
        'rows': n, 'correct': sum(r['correct'] for r in rows),
        'accuracy': sum(r['correct'] for r in rows)/n,
        'percent_brier': sum(map(brier, rows))/n,
        'mean_reported_percent': sum(r['percent'] for r in rows)/n,
        'mean_label_mass': sum(r['label_mass'] for r in rows)/n,
        'percent_buckets': dict(sorted(Counter(f"{min(r['percent']//10,9)*10:02d}-{min(r['percent']//10,9)*10+9:02d}" for r in rows).items())),
        'by_source': {s: {'rows': len(group),
                          'correct': sum(r['correct'] for r in group),
                          'accuracy': sum(r['correct'] for r in group)/len(group)}
                      for s, group in sorted(by_source.items())},
    }


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    manifest = json.loads((DATA/'manifest.json').read_text(encoding='utf-8'))
    blind_file = DATA/'blind.jsonl'
    if sha256(blind_file) != manifest['output_sha256'] or manifest['rows'] != 2000:
        raise ValueError('Blind data changed')
    data = {r['id']: r for r in read_rows(blind_file)}
    predictions = {}
    files = {}
    for name in ('existing', 'ce_brier'):
        folder = EVAL/name
        if (folder/'STATUS.txt').read_text().strip() != 'COMPLETED 2000':
            raise ValueError(f'{name} incomplete')
        run = json.loads((folder/'run_config.json').read_text())
        if run['blind_data_sha256'] != manifest['output_sha256']:
            raise ValueError(f'{name} used different data')
        if sha256(Path(run['calibration'])) != run['calibration_sha256']:
            raise ValueError(f'{name} calibration changed')
        if sha256(Path(run['adapter'])/'adapter_model.safetensors') != run['adapter_weights_sha256']:
            raise ValueError(f'{name} weights changed')
        file = folder/'predictions.jsonl'
        rows = read_rows(file)
        if len(rows) != 2000 or {r['id'] for r in rows} != set(data):
            raise ValueError(f'{name} IDs differ')
        by_id = {r['id']:r for r in rows}
        if len(by_id) != 2000:
            raise ValueError(f'{name} duplicate IDs')
        for key, row in by_id.items():
            original = data[key]
            if row['video_path'] != original['video_path'] or row['data_source'] != original['data_source']:
                raise ValueError(f'{name} metadata differs')
            if row['gold_action'] != chr(65+original['correct_choice']):
                raise ValueError(f'{name} gold differs')
            if row['correct'] != int(row['predicted_action'] == row['gold_action']):
                raise ValueError(f'{name} correctness differs')
            if row['percent'] != round(100*row['platt_q']):
                raise ValueError(f'{name} percent differs')
        predictions[name] = by_id
        files[name] = {'predictions': str(file), 'sha256': sha256(file)}
    ids = sorted(data)
    old = [predictions['existing'][key] for key in ids]
    new = [predictions['ce_brier'][key] for key in ids]
    cluster = defaultdict(list)
    for i, key in enumerate(ids):
        cluster[data[key]['video_path']].append(i)
    groups = list(cluster.values())
    group_stats = []
    for indices in groups:
        group_stats.append((len(indices),
                            sum(new[i]['correct']-old[i]['correct'] for i in indices),
                            sum(brier(new[i])-brier(old[i]) for i in indices)))
    rng = random.Random(SEED)
    acc_samples, brier_samples = [], []
    for _ in range(N_BOOT):
        picked = [group_stats[rng.randrange(len(groups))] for _ in groups]
        denom = sum(g[0] for g in picked)
        acc_samples.append(sum(g[1] for g in picked)/denom)
        brier_samples.append(sum(g[2] for g in picked)/denom)
    old_summary, new_summary = summary(old), summary(new)
    result = {
        'protocol': str(ROOT/'JEV10_NEW_BLIND_PROTOCOL.md'),
        'blind_manifest_sha256': sha256(DATA/'manifest.json'),
        'blind_data_sha256': manifest['output_sha256'],
        'prediction_files': files,
        'rows': len(ids), 'video_clusters': len(groups),
        'existing': old_summary, 'ce_brier': new_summary,
        'ce_brier_minus_existing_accuracy': new_summary['accuracy']-old_summary['accuracy'],
        'accuracy_cluster_bootstrap_95pct': interval(acc_samples),
        'gained_questions': sum(a['correct']==0 and b['correct']==1 for a,b in zip(old,new)),
        'lost_questions': sum(a['correct']==1 and b['correct']==0 for a,b in zip(old,new)),
        'ce_brier_minus_existing_percent_brier': new_summary['percent_brier']-old_summary['percent_brier'],
        'percent_brier_cluster_bootstrap_95pct': interval(brier_samples),
        'bootstrap_seed': SEED, 'bootstrap_repetitions': N_BOOT,
    }
    OUT.mkdir(parents=True)
    (OUT/'comparison.json').write_text(json.dumps(result, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')
    (OUT/'STATUS.txt').write_text('COMPLETED\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('existing','ce_brier')}, indent=2))
    print(json.dumps({'existing': {k:v for k,v in old_summary.items() if k!='by_source'},
                      'ce_brier': {k:v for k,v in new_summary.items() if k!='by_source'}}, indent=2))


if __name__ == '__main__':
    main()
