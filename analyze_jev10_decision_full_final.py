"""Paired test comparison of selected full run and existing production baseline."""
import json
import random
from pathlib import Path

ROOT=Path('/pfs/hyx/videojev-rlcd')
OLD=ROOT/'calibration/jev10_action/selected_step_4813_context/test_predictions.jsonl'
NEW=ROOT/'calibration/jev10_action/full_ce_brier_step_2000/test_predictions.jsonl'
OUT=ROOT/'comparisons/jev10_decision_full_v1'


def read(path):
    return {r['id']:r for r in (json.loads(s) for s in path.read_text().splitlines())}


def interval(values):
    values.sort()
    return [values[int(0.025*len(values))],values[int(0.975*len(values))]]


def main():
    old,new=read(OLD),read(NEW)
    ids=sorted(old)
    if len(ids)!=481 or set(ids)!=set(new): raise ValueError('Test IDs mismatch')
    old_rows=[old[k] for k in ids]
    new_rows=[new[k] for k in ids]
    for r in old_rows+new_rows:
        r['percent_q']=round(100*r['platt_q'])/100
    n=len(ids)
    old_correct=sum(r['correct'] for r in old_rows)
    new_correct=sum(r['correct'] for r in new_rows)
    accuracy_diff=[b['correct']-a['correct'] for a,b in zip(old_rows,new_rows)]
    brier_diff=[(b['percent_q']-b['correct'])**2-(a['percent_q']-a['correct'])**2
                for a,b in zip(old_rows,new_rows)]
    rng=random.Random(20261003)
    acc_boot=[];brier_boot=[]
    for _ in range(10000):
        chosen=[rng.randrange(n) for _ in range(n)]
        acc_boot.append(sum(accuracy_diff[i] for i in chosen)/n)
        brier_boot.append(sum(brier_diff[i] for i in chosen)/n)
    result={'rows':n,'old_correct':old_correct,'new_correct':new_correct,
            'old_accuracy':old_correct/n,'new_accuracy':new_correct/n,
            'new_minus_old_accuracy':sum(accuracy_diff)/n,
            'accuracy_paired_bootstrap_95pct':interval(acc_boot),
            'gained_questions':sum(a['correct']==0 and b['correct']==1 for a,b in zip(old_rows,new_rows)),
            'lost_questions':sum(a['correct']==1 and b['correct']==0 for a,b in zip(old_rows,new_rows)),
            'old_percent_brier':sum((r['percent_q']-r['correct'])**2 for r in old_rows)/n,
            'new_percent_brier':sum((r['percent_q']-r['correct'])**2 for r in new_rows)/n,
            'new_minus_old_percent_brier':sum(brier_diff)/n,
            'brier_paired_bootstrap_95pct':interval(brier_boot)}
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/'test_comparison.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__': main()
