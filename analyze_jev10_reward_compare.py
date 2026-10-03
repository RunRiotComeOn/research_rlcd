"""Compare matched reward pilots on the reused JeV validation split."""
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path('/pfs/hyx/videojev-rlcd')
EVAL = ROOT/'evaluations/jev10_reward_compare_v2'
OUT = ROOT/'comparisons/jev10_reward_compare_v2'
MODES = ('old_brier','proposed')
STEPS = (100,200,300)


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def records(path):
    return [json.loads(s) for s in path.read_text(encoding='utf-8').splitlines()]


def interval(values):
    values.sort()
    return [values[int(.025*len(values))], values[int(.975*len(values))]]


def paired(old, new, key, seed):
    ids = sorted(old)
    cluster = defaultdict(list)
    for key_id in ids:
        cluster[old[key_id]['video_path']].append(key_id)
    group_stats = []
    for group in cluster.values():
        if key == 'accuracy':
            delta = sum(new[i]['correct']-old[i]['correct'] for i in group)
        elif key == 'brier':
            delta = sum((new[i]['confidence']-new[i]['correct'])**2 -
                        (old[i]['confidence']-old[i]['correct'])**2 for i in group)
        else:
            raise ValueError(key)
        group_stats.append((len(group),delta))
    observed = sum(g[1] for g in group_stats)/len(ids)
    rng = random.Random(seed)
    samples = []
    for _ in range(10000):
        picked = [group_stats[rng.randrange(len(group_stats))] for _ in group_stats]
        samples.append(sum(g[1] for g in picked)/sum(g[0] for g in picked))
    return {'difference': observed, 'cluster_bootstrap_95pct': interval(samples)}


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    names = ['baseline'] + [f'{mode}_step_{step:04d}' for mode in MODES for step in STEPS]
    summaries, groups, hashes = {}, {}, {}
    for name in names:
        folder = EVAL/name
        if (folder/'STATUS.txt').read_text().strip() != 'COMPLETED 481':
            raise ValueError(f'{name} incomplete')
        config = json.loads((folder/'run_config.json').read_text())
        if sha256(Path(config['adapter'])/'adapter_model.safetensors') != config['adapter_weights_sha256']:
            raise ValueError(f'{name} adapter changed')
        if sha256(Path(config['data'])) != config['data_sha256']:
            raise ValueError(f'{name} data changed')
        file = folder/'predictions.jsonl'
        rows = records(file)
        if len(rows) != 481 or len({r['id'] for r in rows}) != 481:
            raise ValueError(f'{name} row count or IDs')
        groups[name] = {r['id']:r for r in rows}
        summaries[name] = json.loads((folder/'summary.json').read_text())
        hashes[name] = sha256(file)
    ids = set(groups['baseline'])
    for name, rows in groups.items():
        if set(rows) != ids:
            raise ValueError(f'{name} ID mismatch')
        for key in ids:
            a, b = groups['baseline'][key], rows[key]
            if (a['video_path'],a['gold_action'],a['data_source']) != (b['video_path'],b['gold_action'],b['data_source']):
                raise ValueError(f'{name} metadata mismatch')
    old, new = groups['old_brier_step_0300'], groups['proposed_step_0300']
    training = {}
    for mode in MODES:
        run = ROOT/f'runs/jev10_reward_{mode}_v2_300'
        if (run/'STATUS.txt').read_text().strip() != 'COMPLETED 300':
            raise ValueError(f'{mode} training incomplete')
        logs = records(run/'metrics.jsonl')
        if len(logs) != 300:
            raise ValueError(f'{mode} training log count')
        training[mode] = {'updates': sum(not r['skipped'] for r in logs),
                          'skipped': sum(r['skipped'] for r in logs),
                          'sampled_correct_rate': sum(y for r in logs for y in r['correct'])/1200,
                          'sampled_confidence_bins': dict(sorted(Counter(d for r in logs for d in r['sampled_digits']).items())),
                          'last_100_confidence_bins': dict(sorted(Counter(d for r in logs[200:] for d in r['sampled_digits']).items())),
                          'run_config_sha256': sha256(run/'run_config.json')}
    result = {'protocol': str(ROOT/'JEV10_REWARD_20261003_PROTOCOL.md'),
              'rows': 481, 'video_clusters': len({r['video_path'] for r in old.values()}),
              'summaries': summaries, 'prediction_sha256': hashes,
              'training': training,
              'proposed_minus_old_accuracy_step_300': paired(old,new,'accuracy',20261003),
              'proposed_minus_old_direct_brier_step_300': paired(old,new,'brier',20261004),
              'proposed_gained_questions': sum(old[i]['correct']==0 and new[i]['correct']==1 for i in ids),
              'proposed_lost_questions': sum(old[i]['correct']==1 and new[i]['correct']==0 for i in ids)}
    OUT.mkdir(parents=True)
    (OUT/'comparison.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
    (OUT/'STATUS.txt').write_text('COMPLETED\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('summaries','prediction_sha256')},indent=2))
    for name in names:
        s=summaries[name]
        print(json.dumps({'name':name,'correct':s['correct'],'accuracy':s['accuracy'],
                          'direct_brier':s['direct']['brier'],
                          'expected_brier':s['expected']['brier'],
                          'bins':s['confidence_bins']}))


if __name__ == '__main__':
    main()
