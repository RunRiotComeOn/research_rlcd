"""Score the already-frozen production Platt mapping on reward blind rows."""
import hashlib
import json
import math
from pathlib import Path

from analyze_jev10_reward_compare import paired, records

ROOT = Path('/pfs/hyx/videojev-rlcd')
EVAL = ROOT/'evaluations/jev10_reward_blind_v1'
OUT = ROOT/'comparisons/jev10_reward_blind_v1/platt_reference.json'
CAL = ROOT/'calibration/jev10_action/selected_step_4813_context/summary.json'


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    cal = json.loads(CAL.read_text())
    baseline = {r['id']:r for r in records(EVAL/'baseline/predictions.jsonl')}
    proposed = {r['id']:r for r in records(EVAL/'proposed_step_0300/predictions.jsonl')}
    if len(baseline) != 1000 or set(baseline) != set(proposed):
        raise ValueError('Prediction sets differ')
    calibrated = {}
    for key,row in baseline.items():
        p = max(row['action_probabilities'])
        p = max(1e-6,min(1-1e-6,p))
        logit = cal['slope']*math.log(p/(1-p))+cal['intercept']
        q = 1/(1+math.exp(-max(-30,min(30,logit))))
        calibrated[key] = {**row,'confidence':round(100*q)/100,'platt_q':q}
    brier = sum((r['confidence']-r['correct'])**2 for r in calibrated.values())/1000
    continuous = sum((r['platt_q']-r['correct'])**2 for r in calibrated.values())/1000
    result = {'calibration':str(CAL),'calibration_sha256':sha256(CAL),
              'baseline_predictions_sha256':sha256(EVAL/'baseline/predictions.jsonl'),
              'rows':1000,'accuracy':sum(r['correct'] for r in calibrated.values())/1000,
              'production_rounded_percent_brier':brier,
              'production_continuous_platt_brier':continuous,
              'proposed_minus_production_accuracy':paired(calibrated,proposed,'accuracy',20261007),
              'proposed_gained_questions':sum(calibrated[i]['correct']==0 and proposed[i]['correct']==1 for i in calibrated),
              'proposed_lost_questions':sum(calibrated[i]['correct']==1 and proposed[i]['correct']==0 for i in calibrated),
              'proposed_direct_minus_production_percent_brier':paired(calibrated,proposed,'brier',20261006)}
    OUT.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
