"""Score all selected confidence methods on one sealed final set."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from fit_jev10_unseen_calibration import BASE, OUT as MODELS, action_features, score

ROOT = Path('/pfs/hyx/videojev-rlcd')
OUT = ROOT / 'evaluations/jev10_unseen_final_v1'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--examples', type=int, choices=(1000, 2000, 3000, 4000), required=True)
    args = parser.parse_args()
    if OUT.exists():
        raise FileExistsError(OUT)
    selected = ROOT / 'calibration/jev10_unseen_v1/selected_checkpoint.json'
    selection = json.loads(selected.read_text())
    if selection['examples'] != args.examples:
        raise ValueError('Requested checkpoint differs from development-selected checkpoint')
    base = BASE / 'final'
    if (base / 'STATUS.txt').read_text().strip() != 'COMPLETED 1000':
        raise ValueError('Final action predictions incomplete')
    rows = [json.loads(s) for s in (base / 'predictions.jsonl').read_text().splitlines()]
    hidden = np.load(base / 'action_hidden.npy', mmap_mode='r')
    conf = ROOT / f'evaluations/jev10_unseen_confidence_v1/examples_{args.examples:04d}/final'
    if (conf / 'STATUS.txt').read_text().strip() != 'COMPLETED 1000':
        raise ValueError('Final LoRA confidence incomplete')
    conf_rows = [json.loads(s) for s in (conf / 'predictions.jsonl').read_text().splitlines()]
    if len(rows) != 1000 or [r['id'] for r in rows] != [r['id'] for r in conf_rows]:
        raise ValueError('Final row mismatch')
    par = json.loads((MODELS / 'parameters.json').read_text())
    transform = np.load(MODELS / 'feature_transform.npz')
    actions = action_features(rows)
    z = (np.asarray(hidden, dtype=np.float32)-transform['pca_mean']) @ transform['pca_components'].T
    x = np.concatenate([actions, z], axis=1)
    x = torch.from_numpy(((x-transform['mean'])/transform['scale']).astype(np.float32))
    head = nn.Sequential(nn.Linear(par['head_input_dim'], par['head_hidden_dim']),
                         nn.Tanh(), nn.Linear(par['head_hidden_dim'], 1))
    head.load_state_dict(torch.load(MODELS / 'head_state.pt', weights_only=True, map_location='cpu'))
    head.eval()
    with torch.no_grad():
        head_q = torch.sigmoid(head(x).squeeze(-1)).numpy()
        platt_q = torch.sigmoid(torch.from_numpy(actions[:, 0])*par['platt_slope']+
                                par['platt_intercept']).numpy()
    y = np.asarray([r['correct'] for r in rows], dtype=np.float32)
    accuracy = float(y.mean())
    fit_rate = par['fit_accuracy']
    methods = {'constant_fit_rate': np.full(len(rows), fit_rate),
               'chosen_action_p': np.asarray([r['action_probabilities'][r['action_index']] for r in rows]),
               'original_mean_q': np.asarray([r['original_q'] for r in rows]),
               'platt': platt_q, 'head': head_q,
               'confidence_lora': np.asarray([r['q'] for r in conf_rows])}
    summary = {'rows': len(rows), 'accuracy': accuracy,
               'constant_oracle_brier': accuracy*(1-accuracy),
               'selected_lora_training_examples': args.examples,
               'methods': {name: score(q, y) for name, q in methods.items()}}
    # Resample videos rather than individual questions because some videos have multiple questions.
    groups = {}
    for i, row in enumerate(rows):
        groups.setdefault(Path(row['video_path']).name, []).append(i)
    group_indices = list(groups.values())
    rng = np.random.default_rng(20261005)
    summary['paired_delta_vs_platt'] = {}
    platt_loss = (methods['platt']-y)**2
    for name in ('head', 'confidence_lora'):
        delta = (methods[name]-y)**2-platt_loss
        draws = []
        for _ in range(5000):
            picks = rng.integers(0, len(group_indices), len(group_indices))
            index = np.concatenate([group_indices[j] for j in picks])
            draws.append(float(delta[index].mean()))
        summary['paired_delta_vs_platt'][name] = {
            'brier_delta': float(delta.mean()),
            'video_bootstrap_95pct_interval': np.quantile(draws, [0.025, 0.975]).tolist()}
    OUT.mkdir(parents=True)
    (OUT / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    with (OUT / 'predictions.jsonl').open('w') as stream:
        for i, row in enumerate(rows):
            item = {'id': row['id'], 'correct': int(y[i]),
                    'action_index': row['action_index'],
                    'confidence': {name: float(q[i]) for name, q in methods.items()}}
            stream.write(json.dumps(item) + '\n')
    (OUT / 'STATUS.txt').write_text('COMPLETED 1000\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
