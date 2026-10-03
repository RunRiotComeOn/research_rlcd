"""Select a full-run checkpoint by prespecified validation rule."""
import json
import math
from pathlib import Path

ROOT=Path('/pfs/hyx/videojev-rlcd')
EVAL=ROOT/'evaluations/jev10_action'
OUT=ROOT/'comparisons/jev10_decision_full_v1'
STEPS=(500,1000,2000,3000,4000,4813)


def read(name):
    rows=[json.loads(s) for s in (EVAL/name/'predictions.jsonl').read_text().splitlines()]
    return {r['id']:r for r in rows}


def metrics(rows):
    n=len(rows)
    return {'rows':n,'correct':sum(r['correct'] for r in rows),
            'accuracy':sum(r['correct'] for r in rows)/n,
            'selected_brier':sum((r['selected_p']-r['correct'])**2 for r in rows)/n,
            'multiclass_brier':sum(sum((p-int(j==ord(r['gold_action'])-65))**2
                                        for j,p in enumerate(r['native_probabilities'])) for r in rows)/n,
            'nll':sum(-math.log(max(1e-8,r['gold_p'])) for r in rows)/n,
            'mean_label_mass':sum(r['label_mass'] for r in rows)/n}


def main():
    names=['selected_step_4813_context_validation'] + [
        f'full_{mode}_step_{step:04d}_validation'
        for mode in ('ce','ce_brier') for step in STEPS]
    groups={name:read(name) for name in names}
    ids=set(groups[names[0]])
    if len(ids)!=481 or any(set(g)!=ids for g in groups.values()):
        raise ValueError('Validation IDs mismatch')
    results={name:metrics(list(g.values())) for name,g in groups.items()}
    selected=min(names,key=lambda name:(-results[name]['correct'],
                                        results[name]['multiclass_brier']))
    result={'selection_rule':'highest validation correct count; tie lower multiclass Brier',
            'selected':selected,'validation':results}
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/'validation_selection.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__': main()
