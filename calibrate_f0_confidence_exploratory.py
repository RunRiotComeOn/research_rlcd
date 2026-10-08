"""Fit matched development Platt calibrators on LoRA confidence and F0 probability."""
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.metrics import roc_auc_score


ROOT = Path('/pfs/hyx/videojev-rlcd')
BASE = ROOT / 'evaluations/f0_confidence_ablation_v1'
OUT = ROOT / 'calibration/f0_lora_platt_exploratory_v1'


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def logit(q):
    p = np.clip(q, 1e-6, 1-1e-6)
    return np.log(p/(1-p))


def fit(q,y):
    z = logit(q)
    def objective(theta):
        logits = theta[0]*z+theta[1]
        loss = np.mean(np.logaddexp(0,logits)-y*logits)+1e-4*theta[0]**2
        residual = expit(logits)-y
        gradient = np.array([np.mean(residual*z)+2e-4*theta[0],np.mean(residual)])
        return loss,gradient
    result = minimize(objective,[1.0,0.0],jac=True,method='L-BFGS-B',
                      options={'maxiter':1000,'ftol':1e-12,'gtol':1e-9})
    if not result.success:
        raise RuntimeError(result.message)
    return result.x


def metrics(q,y):
    bins = np.minimum((q*10).astype(int),9)
    ece = 0.0
    for index in range(10):
        mask = bins==index
        if mask.any():
            ece += mask.mean()*abs(float(q[mask].mean()-y[mask].mean()))
    return {'brier':float(np.mean((q-y)**2)),'auc':float(roc_auc_score(y,q)),
            'ece_10_equal_width':float(ece),'mean_q':float(q.mean())}


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    selection_path = BASE/'selection.json'
    selection = json.loads(selection_path.read_text())
    variant,choice = min(selection['variants'].items(),
                         key=lambda item:item[1]['selected_dev']['brier'])
    step = choice['selected_step']
    assert variant=='f0_high' and step==20000
    suffix = Path(variant)/f'examples_{step:05d}'
    paths = {s:BASE/s/suffix/'predictions.jsonl' for s in ('dev','test')}
    rows = {s:read_jsonl(p) for s,p in paths.items()}
    assert all(len(rows[s])==2000 for s in rows)
    assert not {r['id'] for r in rows['dev']} & {r['id'] for r in rows['test']}
    assert not {Path(r['video_path']).name for r in rows['dev']} & {Path(r['video_path']).name for r in rows['test']}
    y_dev = np.array([r['correct'] for r in rows['dev']],dtype=np.float64)
    y_test = np.array([r['correct'] for r in rows['test']],dtype=np.float64)
    methods,parameters,calibrated = {},{},{}
    for name,field in [('lora','confidence_lora_q'),('answer_probability','selected_action_p')]:
        q_dev = np.array([r[field] for r in rows['dev']],dtype=np.float64)
        q_test = np.array([r[field] for r in rows['test']],dtype=np.float64)
        slope,intercept = fit(q_dev,y_dev)
        transformed = expit(slope*logit(q_test)+intercept)
        calibrated[name] = transformed
        parameters[name] = {'slope':float(slope),'intercept':float(intercept),
                            'dev_fit_metrics':metrics(expit(slope*logit(q_dev)+intercept),y_dev)}
        methods[f'raw_{name}'] = metrics(q_test,y_test)
        methods[f'dev_platt_{name}'] = metrics(transformed,y_test)
    methods['previous_train_platt_answer_probability'] = metrics(
        np.array([r['platt_q'] for r in rows['test']],dtype=np.float64),y_test)
    delta = (calibrated['lora']-y_test)**2-(calibrated['answer_probability']-y_test)**2
    groups = defaultdict(list)
    for i,row in enumerate(rows['test']):
        groups[Path(row['video_path']).name].append(i)
    indices = list(groups.values())
    sums = np.array([delta[index].sum() for index in indices])
    counts = np.array([len(index) for index in indices])
    rng = np.random.default_rng(20261008)
    draws = []
    for _ in range(5000):
        picks = rng.integers(0,len(indices),len(indices))
        draws.append(float(sums[picks].sum()/counts[picks].sum()))
    summary = {'exploratory':True,'reason':'development labels already selected the adapter checkpoint; test results were previously inspected',
               'variant':variant,'step':step,'fit_rows':2000,'test_rows':2000,
               'test_videos':len(indices),'test_accuracy':float(y_test.mean()),
               'constant_oracle_brier':float(y_test.mean()*(1-y_test.mean())),
               'selection_sha256':sha256(selection_path),
               'input_sha256':{s:sha256(p) for s,p in paths.items()},
               'calibration_input':'logit(confidence or chosen-answer probability)',
               'regularization':'1e-4 * slope^2; binary cross entropy fit',
               'parameters':parameters,'test_methods':methods,
               'platt_lora_minus_platt_answer_brier':float(delta.mean()),
               'video_bootstrap_95pct_interval':np.quantile(draws,[0.025,0.975]).tolist()}
    OUT.mkdir(parents=True)
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    main()
