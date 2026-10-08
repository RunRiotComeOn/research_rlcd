"""Summarize selected joint model versus unmodified F0 and its Platt teacher."""
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score


ROOT = Path('/pfs/hyx/videojev-rlcd')
EVAL = ROOT / 'evaluations/f0_joint_confidence_v1'
DATA = ROOT / 'data/f0_joint_confidence_v1'
PLATT = ROOT / 'calibration/f0_joint_confidence_v1/platt/parameters.json'
F0_HOLMES = ROOT / 'evaluations/f0_confidence_ablation_v1/holmes_actions/predictions.jsonl'


def read_jsonl(path):
    with path.open() as stream:
        return [json.loads(line) for line in stream]


def ece(q,y):
    bins = np.minimum((q*10).astype(int),9)
    return float(sum(mask.mean()*abs(float(q[mask].mean()-y[mask].mean()))
                     for i in range(10) if (mask := bins==i).any()))


def score(q,y):
    return {'brier':float(np.mean((q-y)**2)),'auc':float(roc_auc_score(y,q)),
            'ece_10_equal_width':ece(q,y),'mean_q':float(q.mean())}


def interval(rows, values, seed):
    groups = defaultdict(list)
    for row,value in zip(rows,values):
        groups[Path(row['video_path']).name].append(float(value))
    keys = sorted(groups)
    sums = np.array([sum(groups[k]) for k in keys])
    counts = np.array([len(groups[k]) for k in keys])
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(5000):
        picks = rng.integers(0,len(keys),len(keys))
        draws.append(float(sums[picks].sum()/counts[picks].sum()))
    return np.quantile(draws,[0.025,0.975]).tolist()


def main():
    selection = json.loads((EVAL/'selection.json').read_text())
    chosen = selection['global_selection']
    variant, step = chosen['variant'], chosen['step']
    platt = json.loads(PLATT.read_text())
    result = {'selection':chosen,'data_rows':{'train':16063,'dev':2000,'test':2000,'holmes':1837},
              'results':{}}
    for split,expected in (('test',2000),('holmes',1837)):
        base = EVAL / split / variant / f'examples_{step:05d}'
        if (base/'STATUS.txt').read_text().strip() != f'COMPLETED {expected}':
            raise ValueError(f'Incomplete {split} result')
        student = read_jsonl(base/'predictions.jsonl')
        actions = read_jsonl(F0_HOLMES if split=='holmes' else DATA/f'{split}_actions.jsonl')
        assert len(student)==len(actions)==expected
        assert [r['id'] for r in student]==[r['id'] for r in actions]
        y = np.asarray([r['correct'] for r in student],dtype=np.float64)
        f0_y = np.asarray([r['correct'] for r in actions],dtype=np.float64)
        p = np.asarray([r['action_probabilities'][r['action_index']] for r in actions])
        z = np.log(np.clip(p,1e-6,1-1e-6)/np.clip(1-p,1e-6,1-1e-6))
        platt_q = 1/(1+np.exp(-(platt['slope']*z+platt['intercept'])))
        q = np.asarray([r['mean_q'] for r in student],dtype=np.float64)
        summary = json.loads((base/'summary.json').read_text())
        assert abs(summary['direct_mean_q']['brier']-float(np.mean((q-y)**2))) < 1e-10
        brier_delta = (q-y)**2-(platt_q-f0_y)**2
        accuracy_delta = y-f0_y
        result['results'][split] = {
            'rows':expected,'videos':summary['videos'],
            'f0_accuracy':float(f0_y.mean()),'joint_accuracy':float(y.mean()),
            'joint_minus_f0_accuracy':float(accuracy_delta.mean()),
            'accuracy_delta_video_bootstrap_95pct_interval':interval(student,accuracy_delta,20261008),
            'action_agreement_f0':summary['action_agreement_f0'],
            'mean_action_kl_to_f0':summary['mean_action_kl_to_f0'],
            'joint_direct_mean_q':summary['direct_mean_q'],
            'joint_direct_argmax_digit_q':summary['direct_argmax_digit_q'],
            'joint_direct_selected_action_p':summary['direct_selected_action_p'],
            'f0_raw_action_p':score(p,f0_y),
            'f0_platt_reference':score(platt_q,f0_y),
            'joint_minus_f0_platt_brier':float(brier_delta.mean()),
            'brier_delta_video_bootstrap_95pct_interval':interval(student,brier_delta,20261008),
            'joint_constant_oracle_brier':float(y.mean()*(1-y.mean()))}
    path = EVAL/'final_summary.json'
    path.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    main()
