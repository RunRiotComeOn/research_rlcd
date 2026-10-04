"""Fit Platt and a small frozen-feature head; select by new development set."""
import copy
import json
import math
from pathlib import Path

import numpy as np
import torch
from sklearn.decomposition import PCA
from torch import nn

ROOT = Path('/pfs/hyx/videojev-rlcd')
BASE = ROOT / 'calibration/jev10_unseen_v1'
OUT = BASE / 'models'
SEED = 20261005


def load(split):
    base = BASE / split
    expected = 4000 if split == 'fit' else 500
    if (base / 'STATUS.txt').read_text().strip() != f'COMPLETED {expected}':
        raise ValueError(f'Incomplete {split}')
    rows = [json.loads(s) for s in (base / 'predictions.jsonl').read_text().splitlines()]
    hidden = np.load(base / 'action_hidden.npy', mmap_mode='r')
    if len(rows) != expected or len(hidden) != expected:
        raise ValueError(f'Length mismatch {split}')
    return rows, hidden


def action_features(rows):
    result = []
    for row in rows:
        p = np.asarray(row['action_probabilities'], dtype=np.float64)
        chosen = float(p[row['action_index']])
        ranked = np.sort(p)[::-1]
        entropy = -np.sum(p * np.log(np.maximum(p, 1e-12))) / math.log(len(p))
        chosen = np.clip(chosen, 1e-6, 1-1e-6)
        result.append([math.log(chosen/(1-chosen)), chosen, ranked[0]-ranked[1],
                       entropy, len(p)/6])
    return np.asarray(result, dtype=np.float32)


def score(q, y):
    q = np.asarray(q)
    return {'brier': float(np.mean((q-y)**2)), 'mean_q': float(np.mean(q)),
            'accuracy': float(np.mean(y))}


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    fit, hidden_fit = load('fit')
    dev, hidden_dev = load('development')
    y_fit = np.asarray([r['correct'] for r in fit], dtype=np.float32)
    y_dev = np.asarray([r['correct'] for r in dev], dtype=np.float32)
    a_fit, a_dev = action_features(fit), action_features(dev)
    torch.set_num_threads(4)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    # Frozen visual-language representation is compressed using fit examples only.
    pca = PCA(n_components=32, svd_solver='randomized', random_state=SEED)
    z_fit = pca.fit_transform(np.asarray(hidden_fit, dtype=np.float32))
    z_dev = pca.transform(np.asarray(hidden_dev, dtype=np.float32))
    x_fit = np.concatenate([a_fit, z_fit], axis=1)
    x_dev = np.concatenate([a_dev, z_dev], axis=1)
    mu = x_fit.mean(0)
    scale = np.maximum(x_fit.std(0), 1e-5)
    train_x = torch.from_numpy(((x_fit-mu)/scale).astype(np.float32))
    dev_x = torch.from_numpy(((x_dev-mu)/scale).astype(np.float32))
    train_y = torch.from_numpy(y_fit)
    dev_y = torch.from_numpy(y_dev)
    # Production-compatible Platt: sigmoid(slope*logit(chosen_p)+intercept).
    platt = nn.Parameter(torch.tensor([1.0, 0.0]))
    optimizer = torch.optim.LBFGS([platt], max_iter=100, line_search_fn='strong_wolfe')
    z = torch.from_numpy(a_fit[:, 0])
    def closure():
        optimizer.zero_grad()
        loss = nn.functional.binary_cross_entropy_with_logits(platt[0]*z+platt[1], train_y)
        loss = loss + 1e-4*platt[0].square()
        loss.backward()
        return loss
    optimizer.step(closure)
    platt_params = platt.detach().numpy().copy()
    head = nn.Sequential(nn.Linear(x_fit.shape[1], 16), nn.Tanh(), nn.Linear(16, 1))
    opt = torch.optim.AdamW(head.parameters(), lr=0.002, weight_decay=0.01)
    best_score, best_epoch, best_state, patience = float('inf'), None, None, 0
    generator = torch.Generator().manual_seed(SEED)
    for epoch in range(1, 301):
        head.train()
        for idx in torch.randperm(len(train_x), generator=generator).split(128):
            opt.zero_grad()
            q_logits = head(train_x[idx]).squeeze(-1)
            loss = nn.functional.binary_cross_entropy_with_logits(q_logits, train_y[idx])
            loss.backward()
            opt.step()
        head.eval()
        with torch.no_grad():
            dev_q = torch.sigmoid(head(dev_x).squeeze(-1))
            dev_brier = float((dev_q-dev_y).square().mean())
        if dev_brier < best_score - 1e-6:
            best_score, best_epoch = dev_brier, epoch
            best_state = copy.deepcopy(head.state_dict())
            patience = 0
        else:
            patience += 1
            if patience >= 20:
                break
    head.load_state_dict(best_state)
    head.eval()
    OUT.mkdir(parents=True)
    np.savez(OUT / 'feature_transform.npz', pca_mean=pca.mean_,
             pca_components=pca.components_, mean=mu, scale=scale)
    torch.save(head.state_dict(), OUT / 'head_state.pt')
    parameters = {'platt_slope': float(platt_params[0]),
                  'platt_intercept': float(platt_params[1]),
                  'head_input_dim': x_fit.shape[1], 'head_hidden_dim': 16,
                  'head_best_epoch': best_epoch, 'head_best_development_brier': best_score,
                  'fit_accuracy': float(y_fit.mean())}
    (OUT / 'parameters.json').write_text(json.dumps(parameters, indent=2) + '\n')
    summary = {}
    for name, rows, actions, x, y in [('fit', fit, a_fit, train_x, y_fit),
                                      ('development', dev, a_dev, dev_x, y_dev)]:
        with torch.no_grad():
            p_q = torch.sigmoid(torch.from_numpy(actions[:, 0])*platt_params[0]+platt_params[1]).numpy()
            h_q = torch.sigmoid(head(x).squeeze(-1)).numpy()
        base_rate = float(y_fit.mean())
        oracle_rate = float(y.mean())
        summary[name] = {'rows': len(rows), 'accuracy': oracle_rate,
                         'constant_oracle_brier': oracle_rate*(1-oracle_rate),
                         'methods': {'constant_fit_rate': score(np.full(len(y), base_rate), y),
                                     'original_mean_q': score([r['original_q'] for r in rows], y),
                                     'chosen_action_p': score([r['action_probabilities'][r['action_index']] for r in rows], y),
                                     'platt': score(p_q, y), 'head': score(h_q, y)}}
    (OUT / 'fit_development_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
