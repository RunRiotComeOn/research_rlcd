"""Evaluate full-JeV answer + OOF-trained confidence on all Holmes questions."""
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
import yaml
from peft import PeftModel
from sklearn.metrics import roc_auc_score

from core import CONFIDENCES
from decision_policy_v2 import candidate_logits, confidence_tokens
from model import load_model, prepare_video, to_device

ROOT = Path('/pfs/hyx/videojev-rlcd')
DATA = ROOT / 'data/jev_full_holmes_v1/holmes_all.jsonl'
ACTIONS = ROOT / 'evaluations/jevfull_holmes_v1/actions'
CONF_RUN = ROOT / 'runs/jevfull_confidence_oof_v1'
ADAPTER = CONF_RUN / 'checkpoints/examples_48126'
PLATT = ROOT / 'calibration/jevfull_oof_v1/platt'
OUT = ROOT / 'evaluations/jevfull_holmes_v1/confidence'


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def metrics(q, y):
    q = np.asarray(q, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    return {'brier': float(np.mean((q-y)**2)),
            'mean_q': float(np.mean(q)),
            'auc': float(roc_auc_score(y, q))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', required=True)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if (CONF_RUN / 'STATUS.txt').read_text().strip() != 'COMPLETED 48126':
        raise ValueError('Confidence training incomplete')
    if (ACTIONS / 'STATUS.txt').read_text().strip() != 'COMPLETED 1837':
        raise ValueError('Holmes actions incomplete')
    if (PLATT / 'STATUS.txt').read_text().strip() != 'COMPLETED 48126':
        raise ValueError('OOF Platt incomplete')
    rows = [json.loads(s) for s in DATA.read_text().splitlines()]
    actions = [json.loads(s) for s in (ACTIONS / 'predictions.jsonl').read_text().splitlines()]
    if len(rows) != 1837 or len(actions) != 1837 or [r['id'] for r in rows] != [r['id'] for r in actions]:
        raise ValueError('Holmes row mismatch')
    platt = json.loads((PLATT / 'parameters.json').read_text())
    config = {'data': str(DATA), 'data_sha256': sha256(DATA),
              'actions_sha256': sha256(ACTIONS / 'predictions.jsonl'),
              'confidence_adapter': str(ADAPTER),
              'confidence_adapter_sha256': sha256(ADAPTER / 'adapter_model.safetensors'),
              'platt_sha256': sha256(PLATT / 'parameters.json'), 'device': args.device}
    if OUT.exists() and not args.resume:
        raise FileExistsError(OUT)
    if args.resume and not OUT.exists():
        raise FileNotFoundError(OUT)
    if not args.resume:
        OUT.mkdir(parents=True)
        (OUT / 'config.json').write_text(json.dumps(config, indent=2) + '\n')
        start = 0
    else:
        if json.loads((OUT / 'config.json').read_text()) != config:
            raise ValueError('Resume config mismatch')
        old = [json.loads(s) for s in (OUT / 'predictions.jsonl').read_text().splitlines()]
        start = len(old)
        if [r['id'] for r in old] != [r['id'] for r in rows[:start]]:
            raise ValueError('Resume row mismatch')
    cfg = yaml.safe_load((ROOT / 'config.yaml').read_text(encoding='utf-8'))
    processor, model = load_model(cfg['model']['path'], args.device)
    model.requires_grad_(False)
    model.model.language_model = PeftModel.from_pretrained(
        model.model.language_model, str(ADAPTER), is_trainable=False)
    model.to(args.device).eval()
    q_values = torch.tensor(CONFIDENCES, dtype=torch.float32, device=args.device)
    tokens = {a: confidence_tokens(processor, a) for a in {r['action_index'] for r in actions}}
    with (OUT / 'predictions.jsonl').open('a' if args.resume else 'w') as stream:
        for i in range(start, len(rows)):
            row, action = rows[i], actions[i]
            batch = to_device(prepare_video(processor, row, cfg['media']['max_frames'],
                                            cfg['media']['max_pixels']), args.device)
            prefix, ids = tokens[action['action_index']]
            with torch.inference_mode():
                probs = candidate_logits(model, batch, prefix, ids).softmax(-1)
                q_lora = float((probs*q_values).sum())
            chosen_p = action['action_probabilities'][action['action_index']]
            p = min(max(chosen_p, 1e-6), 1-1e-6)
            logit = math.log(p/(1-p))
            q_platt = 1/(1+math.exp(-(platt['slope']*logit+platt['intercept'])))
            item = {'id': row['id'], 'video_path': row['video_path'],
                    'correct': action['correct'], 'action_index': action['action_index'],
                    'chosen_action_p': chosen_p, 'platt_q': q_platt,
                    'confidence_lora_q': q_lora,
                    'confidence_digit_probabilities': probs.tolist()}
            stream.write(json.dumps(item) + '\n')
            stream.flush()
            if (i+1) % 100 == 0 or i+1 == len(rows):
                print(json.dumps({'processed': i+1, 'total': len(rows)}), flush=True)
    results = [json.loads(s) for s in (OUT / 'predictions.jsonl').read_text().splitlines()]
    if len(results) != 1837:
        raise ValueError('Incomplete final output')
    y = np.asarray([r['correct'] for r in results], dtype=np.float64)
    p = float(y.mean())
    qs = {'selected_action_p': [r['chosen_action_p'] for r in results],
          'platt': [r['platt_q'] for r in results],
          'confidence_lora': [r['confidence_lora_q'] for r in results]}
    summary = {'rows': len(results), 'accuracy': p,
               'constant_oracle_brier': p*(1-p),
               'methods': {name: metrics(q,y) for name,q in qs.items()}}
    groups = {}
    for i, row in enumerate(results):
        groups.setdefault(Path(row['video_path']).name, []).append(i)
    indices = list(groups.values())
    delta = (np.asarray(qs['confidence_lora'])-y)**2-(np.asarray(qs['platt'])-y)**2
    rng = np.random.default_rng(20261005)
    draws = []
    for _ in range(5000):
        picks = rng.integers(0, len(indices), len(indices))
        sample = np.concatenate([indices[j] for j in picks])
        draws.append(float(delta[sample].mean()))
    summary['lora_minus_platt'] = {'brier_delta': float(delta.mean()),
                                    'video_bootstrap_95pct_interval': np.quantile(draws, [0.025,0.975]).tolist()}
    (OUT / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    (OUT / 'STATUS.txt').write_text('COMPLETED 1837\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
