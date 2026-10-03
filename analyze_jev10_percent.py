"""Compare direct integer percentages with calibrated percentages and 10 bins."""
import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path('/pfs/hyx/videojev-rlcd')
PERCENT = ROOT/'evaluations/jev10_percent'
BIN = ROOT/'evaluations/jev10_constrained'
PLATT = ROOT/'calibration/jev10_action/selected_step_4813'
OUT = ROOT/'comparisons/jev10_percent_v1'


def read(path):
    return [json.loads(s) for s in path.read_text().splitlines()]


def score(rows, key):
    return sum((r[key]-r['correct'])**2 for r in rows)/len(rows)


def deciles(rows, key):
    groups=defaultdict(list)
    for r in rows:
        groups[min(9,int(10*r[key]))].append(r)
    return {str(k):{'count':len(v),'mean_confidence':sum(r[key] for r in v)/len(v),
                    'accuracy':sum(r['correct'] for r in v)/len(v)}
            for k,v in sorted(groups.items())}


def main():
    direct={}
    for name in ('sft_baseline_validation','rlcd_step_0100','rlcd_step_0200','rlcd_step_0300'):
        rows=read(PERCENT/name/'predictions.jsonl')
        direct[name]={'rows':len(rows),'accuracy':sum(r['correct'] for r in rows)/len(rows),
                      'brier':score(rows,'confidence'),
                      'distinct_percentages':len({r['percent'] for r in rows}),
                      'most_common':Counter(r['percent'] for r in rows).most_common(10),
                      'deciles':deciles(rows,'confidence')}
    bin_rows=read(BIN/'rlcd_step_0300'/'predictions.jsonl')
    direct['old_rlcd_10_bin']={'rows':len(bin_rows),'accuracy':sum(r['correct'] for r in bin_rows)/len(bin_rows),
                               'brier':score(bin_rows,'confidence')}
    calibrated={}
    for split in ('validation','test'):
        rows=read(PLATT/f'{split}_predictions.jsonl')
        for r in rows: r['percent_q']=round(100*r['platt_q'])/100
        calibrated[split]={'rows':len(rows),'accuracy':sum(r['correct'] for r in rows)/len(rows),
                           'percent_brier':score(rows,'percent_q'),
                           'old_bin_brier':score(rows,'platt_discrete'),
                           'distinct_percentages':len({r['percent_q'] for r in rows}),
                           'deciles':deciles(rows,'percent_q')}
    result={'direct':direct,'calibrated':calibrated,
            'note':'Direct RL variants evaluated only on validation; Platt fitted on validation and independently evaluated on test.'}
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__': main()
