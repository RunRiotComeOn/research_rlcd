"""Fit Platt on all out-of-fold JeV answer predictions."""
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch

from prepare_data import ROOT

BASE = ROOT / 'calibration/jevfull_oof_v1'
OUT = BASE / 'platt'


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    rows = []
    hashes = {}
    for fold in ('fold0', 'fold1'):
        base = BASE / fold
        if (base / 'STATUS.txt').read_text().strip() != 'COMPLETED 24063':
            raise ValueError(f'Incomplete OOF fold {fold}')
        path = base / 'predictions.jsonl'
        part = [json.loads(s) for s in path.read_text().splitlines()]
        if len(part) != 24063 or any(r['source_answer_fold'] == fold for r in part):
            raise ValueError(f'Bad OOF fold {fold}')
        rows.extend(part)
        hashes[fold] = sha256(path)
    if len({r['id'] for r in rows}) != 48126:
        raise ValueError('OOF IDs not unique')
    p = np.asarray([r['action_probabilities'][r['action_index']] for r in rows], dtype=np.float64)
    y = np.asarray([r['correct'] for r in rows], dtype=np.float64)
    z = np.log(np.clip(p,1e-6,1-1e-6)/np.clip(1-p,1e-6,1-1e-6))
    z_t = torch.tensor(z, dtype=torch.float64)
    y_t = torch.tensor(y, dtype=torch.float64)
    par = torch.nn.Parameter(torch.tensor([1.0,0.0], dtype=torch.float64))
    opt = torch.optim.LBFGS([par], max_iter=100, line_search_fn='strong_wolfe')
    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(par[0]*z_t+par[1],y_t)
        loss = loss + 1e-4*par[0].square()
        loss.backward()
        return loss
    opt.step(closure)
    slope, intercept = par.detach().tolist()
    q = torch.sigmoid(par[0]*z_t+par[1]).detach().numpy()
    rate = float(y.mean())
    result = {'rows': len(rows), 'oof_sha256': hashes,
              'accuracy': rate, 'constant_oracle_brier': rate*(1-rate),
              'selected_p_brier': float(np.mean((p-y)**2)),
              'slope': slope, 'intercept': intercept,
              'platt_mean_q': float(q.mean()),
              'platt_fit_brier': float(np.mean((q-y)**2))}
    OUT.mkdir(parents=True)
    (OUT / 'parameters.json').write_text(json.dumps(result, indent=2) + '\n')
    (OUT / 'STATUS.txt').write_text('COMPLETED 48126\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
