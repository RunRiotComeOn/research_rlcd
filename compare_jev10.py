"""Paired final JeV test comparison after checkpoint selection."""
import json
import random
from collections import defaultdict
from pathlib import Path

ROOT = Path('/pfs/hyx/videojev-rlcd')
BASE = ROOT / 'calibration/jev10_action/base/test_predictions.jsonl'
NEW = ROOT / 'calibration/jev10_action/selected_step_4813/test_predictions.jsonl'
OUT = ROOT / 'comparisons/jev10_action_v1'


def load(path):
    return {r['id']: r for r in (json.loads(s) for s in path.read_text().splitlines())}


def interval(values):
    ordered = sorted(values)
    return [ordered[int(0.025*len(ordered))], ordered[int(0.975*len(ordered))]]


def main():
    base = load(BASE)
    new = load(NEW)
    if len(base) != 481 or set(base) != set(new):
        raise ValueError('Paired test rows mismatch')
    keys = sorted(base)
    accuracy_gain = [new[k]['correct'] - base[k]['correct'] for k in keys]
    brier_gain = [
        (base[k]['platt_discrete']-base[k]['correct'])**2 -
        (new[k]['platt_discrete']-new[k]['correct'])**2 for k in keys
    ]
    wins = sum(x == 1 for x in accuracy_gain)
    losses = sum(x == -1 for x in accuracy_gain)
    rng = random.Random(20261003)
    accuracy_boot = []
    brier_boot = []
    for _ in range(10000):
        positions = [rng.randrange(len(keys)) for _ in keys]
        accuracy_boot.append(sum(accuracy_gain[i] for i in positions)/len(keys))
        brier_boot.append(sum(brier_gain[i] for i in positions)/len(keys))
    by_source = defaultdict(list)
    for k in keys:
        by_source[base[k]['data_source']].append(k)
    result = {
        'rows': len(keys), 'accuracy_gain': sum(accuracy_gain)/len(keys),
        'accuracy_gain_paired_bootstrap_95pct': interval(accuracy_boot),
        'brier_gain': sum(brier_gain)/len(keys),
        'brier_gain_paired_bootstrap_95pct': interval(brier_boot),
        'new_correct_base_wrong': wins, 'new_wrong_base_correct': losses,
        'by_source': {
            source: {'rows': len(group),
                     'base_accuracy': sum(base[k]['correct'] for k in group)/len(group),
                     'new_accuracy': sum(new[k]['correct'] for k in group)/len(group)}
            for source, group in sorted(by_source.items()) if len(group) >= 15
        },
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'paired_summary.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
