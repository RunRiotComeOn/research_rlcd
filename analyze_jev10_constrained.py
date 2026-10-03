"""Analyze the matched constrained RL-only and RLCD validation pilots."""
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path('/pfs/hyx/videojev-rlcd')
EVAL = ROOT / 'evaluations/jev10_constrained'
OUT = ROOT / 'comparisons/jev10_constrained_300'


def read(name):
    return {r['id']: r for r in (json.loads(s) for s in (EVAL / name / 'predictions.jsonl').read_text().splitlines())}


def auc(rows, q_key):
    pos = [r[q_key] for r in rows if r['correct']]
    neg = [r[q_key] for r in rows if not r['correct']]
    return sum((a > b) + 0.5*(a == b) for a in pos for b in neg)/(len(pos)*len(neg))


def interval(values):
    values = sorted(values)
    return [values[int(.025*len(values))], values[int(.975*len(values))]]


def main():
    names = ['sft_baseline_validation',
             'rl_only_step_0100', 'rl_only_step_0200', 'rl_only_step_0300',
             'rlcd_step_0100', 'rlcd_step_0200', 'rlcd_step_0300']
    groups = {name: read(name) for name in names}
    ids = set(groups[names[0]])
    if len(ids) != 481 or any(set(rows) != ids for rows in groups.values()):
        raise ValueError('Evaluation ID mismatch')
    platt_rows = [json.loads(s) for s in (ROOT / 'calibration/jev10_action/selected_step_4813/validation_predictions.jsonl').read_text().splitlines()]
    platt = {r['id']: r for r in platt_rows}
    if set(platt) != ids:
        raise ValueError('Platt validation IDs mismatch')
    if any(groups[names[0]][k]['predicted_action'] != platt[k]['predicted_action'] for k in ids):
        raise ValueError('SFT constrained actions differ from Platt source')
    results = {}
    for name, by_id in groups.items():
        rows = list(by_id.values())
        bins = defaultdict(list)
        for r in rows:
            bins[r['confidence_bin']].append(r)
        results[name] = {
            'accuracy': sum(r['correct'] for r in rows)/len(rows),
            'brier': sum((r['confidence']-r['correct'])**2 for r in rows)/len(rows),
            'rank_auc': auc(rows, 'confidence'),
            'mean_confidence_when_correct': sum(r['confidence'] for r in rows if r['correct'])/sum(r['correct'] for r in rows),
            'mean_confidence_when_wrong': sum(r['confidence'] for r in rows if not r['correct'])/sum(not r['correct'] for r in rows),
            'bins': {str(k): {'count': len(v), 'accuracy': sum(r['correct'] for r in v)/len(v)}
                     for k,v in sorted(bins.items())},
        }
    results['sft_platt'] = {
        'accuracy': sum(r['correct'] for r in platt_rows)/len(platt_rows),
        'brier': sum((r['platt_discrete']-r['correct'])**2 for r in platt_rows)/len(platt_rows),
        'rank_auc': auc(platt_rows, 'platt_discrete'),
    }
    rng = random.Random(20261003)
    ordered = sorted(ids)
    diffs = [
        (groups['rlcd_step_0300'][k]['confidence']-groups['rlcd_step_0300'][k]['correct'])**2 -
        (platt[k]['platt_discrete']-platt[k]['correct'])**2 for k in ordered
    ]
    boot = [sum(diffs[rng.randrange(len(diffs))] for _ in diffs)/len(diffs) for _ in range(10000)]
    results['rlcd300_minus_platt_brier'] = sum(diffs)/len(diffs)
    results['rlcd300_minus_platt_brier_paired_bootstrap_95pct'] = interval(boot)
    for mode, run_name in [('rl_only','jev10_constrained_rl_only_300'),('rlcd','jev10_constrained_rlcd_300')]:
        entries = [json.loads(s) for s in (ROOT / 'runs' / run_name / 'metrics.jsonl').read_text().splitlines()]
        results[mode+'_training'] = {'steps': len(entries), 'updates': sum(not e['skipped'] for e in entries),
                                     'skipped': sum(e['skipped'] for e in entries),
                                     'sampled_digit_bins': dict(sorted(Counter(d for e in entries for d in e['sampled_digits']).items()))}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'summary.json').write_text(json.dumps(results, indent=2) + '\n')
    print(json.dumps({k:v for k,v in results.items() if k in ('rlcd_step_0300','sft_platt','rlcd300_minus_platt_brier','rlcd300_minus_platt_brier_paired_bootstrap_95pct','rl_only_training','rlcd_training')}, indent=2))


if __name__ == '__main__':
    main()
