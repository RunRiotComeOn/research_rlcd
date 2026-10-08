"""Fit the training-only Platt teacher on fresh F0-unseen JeV train actions."""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch


ROOT = Path('/pfs/hyx/videojev-rlcd')
SOURCE = ROOT / 'data/f0_joint_confidence_v1/train_actions.jsonl'
OUT = ROOT / 'calibration/f0_joint_confidence_v1/platt'


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    rows = [json.loads(s) for s in SOURCE.read_text().splitlines()]
    if len(rows) != 16063 or any(r['source_answer_fold'] != 'fold0' for r in rows):
        raise ValueError('Wrong teacher training actions')
    p = np.asarray([r['action_probabilities'][r['action_index']] for r in rows], dtype=np.float64)
    y = np.asarray([r['correct'] for r in rows], dtype=np.float64)
    z = np.log(np.clip(p,1e-6,1-1e-6)/np.clip(1-p,1e-6,1-1e-6))
    z_t = torch.tensor(z, dtype=torch.float64)
    y_t = torch.tensor(y, dtype=torch.float64)
    par = torch.nn.Parameter(torch.tensor([1.0,0.0], dtype=torch.float64))
    optimizer = torch.optim.LBFGS([par], max_iter=100, line_search_fn='strong_wolfe')
    def closure():
        optimizer.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(par[0]*z_t+par[1], y_t)
        loss = loss + 1e-4*par[0].square()
        loss.backward()
        return loss
    optimizer.step(closure)
    slope, intercept = par.detach().tolist()
    q = torch.sigmoid(par[0]*z_t+par[1]).detach().numpy()
    result = {'rows': len(rows), 'source': str(SOURCE), 'source_sha256': sha256(SOURCE),
              'slope': slope, 'intercept': intercept,
              'train_accuracy': float(y.mean()), 'train_brier': float(np.mean((q-y)**2)),
              'train_constant_oracle_brier': float(y.mean()*(1-y.mean()))}
    OUT.mkdir(parents=True)
    (OUT / 'parameters.json').write_text(json.dumps(result, indent=2)+'\n')
    (OUT / 'STATUS.txt').write_text(f'COMPLETED {len(rows)}\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
