"""Compare correctly tokenized decision-score pilots on JeV validation."""
import json
import math
from pathlib import Path

ROOT=Path('/pfs/hyx/videojev-rlcd')
EVAL=ROOT/'evaluations/jev10_action'
OUT=ROOT/'comparisons/jev10_decision_context'


def read(name):
    rows=[json.loads(s) for s in (EVAL/name/'predictions.jsonl').read_text().splitlines()]
    return {r['id']:r for r in rows}


def metrics(rows):
    n=len(rows)
    return {'rows':n,'accuracy':sum(r['correct'] for r in rows)/n,
            'selected_brier':sum((r['selected_p']-r['correct'])**2 for r in rows)/n,
            'multiclass_brier':sum(sum((p-int(j==ord(r['gold_action'])-65))**2
                                        for j,p in enumerate(r['native_probabilities'])) for r in rows)/n,
            'nll':sum(-math.log(max(1e-8,r['gold_p'])) for r in rows)/n,
            'mean_label_mass':sum(r['label_mass'] for r in rows)/n,
            'label_mass_below_half':sum(r['label_mass']<0.5 for r in rows)}


def main():
    names=['selected_step_4813_context_validation'] + [
        f'{mode}_context_step_{step:04d}_validation'
        for mode in ('ce','ce_brier') for step in (100,200,300)]
    groups={name:read(name) for name in names}
    ids=set(groups[names[0]])
    if len(ids)!=481 or any(set(g)!=ids for g in groups.values()):
        raise ValueError('Evaluation IDs mismatch')
    baseline=groups[names[0]]
    result={}
    for name,g in groups.items():
        rows=[g[k] for k in sorted(ids)]
        m=metrics(rows)
        if name!=names[0]:
            m['gained_from_baseline']=sum(g[k]['correct'] and not baseline[k]['correct'] for k in ids)
            m['lost_from_baseline']=sum(baseline[k]['correct'] and not g[k]['correct'] for k in ids)
        result[name]=m
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__': main()
