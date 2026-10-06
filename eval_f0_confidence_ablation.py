"""Evaluate a confidence checkpoint on frozen F0 actions for one split."""
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
DATA = ROOT / 'data/f0_confidence_ablation_v1'
EVAL = ROOT / 'evaluations/f0_confidence_ablation_v1'
PLATT = ROOT / 'calibration/f0_confidence_ablation_v1/platt/parameters.json'
VARIANTS = ('base_low', 'base_high', 'f0_low', 'f0_high')


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def reliability(q, y):
    counts = np.zeros(10, dtype=int)
    ece = 0.0
    for i in range(10):
        mask = np.minimum((q*10).astype(int), 9) == i
        counts[i] = int(mask.sum())
        if counts[i]:
            ece += counts[i]/len(y)*abs(float(q[mask].mean()-y[mask].mean()))
    return float(ece)


def score(q, y):
    q = np.asarray(q, dtype=np.float64)
    return {'brier': float(np.mean((q-y)**2)),
            'auc': float(roc_auc_score(y, q)),
            'ece_10_equal_width': reliability(q, y),
            'mean_q': float(q.mean())}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', choices=('dev', 'test', 'holmes'), required=True)
    parser.add_argument('--variant', choices=VARIANTS, required=True)
    parser.add_argument('--step', type=int, required=True)
    parser.add_argument('--device', required=True)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if args.split == 'holmes':
        data = ROOT / 'data/jev_full_holmes_v1/holmes_all.jsonl'
        action_file = EVAL / 'holmes_actions/predictions.jsonl'
        if (EVAL / 'holmes_actions/STATUS.txt').read_text().strip() != 'COMPLETED 1837':
            raise ValueError('Holmes F0 actions incomplete')
        expected = 1837
    else:
        data = DATA / f'{args.split}.jsonl'
        action_file = DATA / f'{args.split}_actions.jsonl'
        expected = 2000
    run = ROOT / 'runs' / f'f0conf_{args.variant}_v1'
    adapter = run / 'checkpoints' / f'examples_{args.step:05d}'
    if (adapter / 'STATUS.txt').read_text().strip() != f'COMPLETED {args.step}':
        raise ValueError('Incomplete adapter checkpoint')
    if (ROOT / 'calibration/f0_confidence_ablation_v1/platt/STATUS.txt').read_text().strip() != 'COMPLETED 20063':
        raise ValueError('Incomplete Platt baseline')
    rows = [json.loads(s) for s in data.read_text(encoding='utf-8').splitlines()]
    actions = [json.loads(s) for s in action_file.read_text().splitlines()]
    if len(rows) != expected or len(actions) != expected or [r['id'] for r in rows] != [a['id'] for a in actions]:
        raise ValueError('Data/action row mismatch')
    if any(a['source_answer_fold'] != 'fold0' for a in actions):
        raise ValueError('Answers were not produced by F0')
    platt = json.loads(PLATT.read_text())
    out = EVAL / args.split / args.variant / f'examples_{args.step:05d}'
    config = {'split': args.split, 'variant': args.variant, 'step': args.step,
              'data_sha256': sha256(data), 'actions_sha256': sha256(action_file),
              'adapter_sha256': sha256(adapter / 'adapter_model.safetensors'),
              'platt_sha256': sha256(PLATT), 'device': args.device}
    if out.exists() and not args.resume:
        raise FileExistsError(out)
    if args.resume and not out.exists():
        raise FileNotFoundError(out)
    if not args.resume:
        out.mkdir(parents=True)
        (out / 'config.json').write_text(json.dumps(config, indent=2) + '\n')
        start = 0
    else:
        if json.loads((out / 'config.json').read_text()) != config:
            raise ValueError('Resume config mismatch')
        old = [json.loads(s) for s in (out / 'predictions.jsonl').read_text().splitlines()]
        start = len(old)
        if [r['id'] for r in old] != [r['id'] for r in rows[:start]]:
            raise ValueError('Resume row mismatch')
    cfg = yaml.safe_load((ROOT / 'config.yaml').read_text(encoding='utf-8'))
    processor, model = load_model(cfg['model']['path'], args.device)
    model.requires_grad_(False)
    model.model.language_model = PeftModel.from_pretrained(
        model.model.language_model, str(adapter), is_trainable=False)
    model.to(args.device).eval()
    q_values = torch.tensor(CONFIDENCES, dtype=torch.float32, device=args.device)
    tokens = {a: confidence_tokens(processor, a) for a in {r['action_index'] for r in actions}}
    with (out / 'predictions.jsonl').open('a' if args.resume else 'w') as stream:
        for i in range(start, len(rows)):
            row, action = rows[i], actions[i]
            batch = to_device(prepare_video(processor, row, cfg['media']['max_frames'],
                                            cfg['media']['max_pixels']), args.device)
            prefix, ids = tokens[action['action_index']]
            with torch.inference_mode():
                probs = candidate_logits(model, batch, prefix, ids).softmax(-1)
                q_lora = float((probs*q_values).sum())
            p = float(action['action_probabilities'][action['action_index']])
            pc = min(max(p,1e-6),1-1e-6)
            z = math.log(pc/(1-pc))
            q_platt = 1/(1+math.exp(-(platt['slope']*z+platt['intercept'])))
            item = {'id': row['id'], 'video_path': row['video_path'],
                    'correct': action['correct'], 'action_index': action['action_index'],
                    'selected_action_p': p, 'platt_q': q_platt,
                    'confidence_lora_q': q_lora}
            stream.write(json.dumps(item) + '\n')
            stream.flush()
            if (i+1) % 100 == 0 or i+1 == len(rows):
                print(json.dumps({'split': args.split, 'variant': args.variant,
                                  'step': args.step, 'processed': i+1,
                                  'total': len(rows)}), flush=True)
    predictions = [json.loads(s) for s in (out / 'predictions.jsonl').read_text().splitlines()]
    if len(predictions) != expected:
        raise ValueError('Incomplete predictions')
    y = np.asarray([r['correct'] for r in predictions], dtype=np.float64)
    summary = {'split': args.split, 'variant': args.variant, 'step': args.step,
               'rows': len(predictions), 'videos': len({Path(r['video_path']).name for r in predictions}),
               'accuracy': float(y.mean()), 'constant_oracle_brier': float(y.mean()*(1-y.mean())),
               'methods': {name: score([r[field] for r in predictions], y)
                           for name, field in [('selected_action_p', 'selected_action_p'),
                                               ('platt', 'platt_q'),
                                               ('confidence_lora', 'confidence_lora_q')]}}
    (out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    (out / 'STATUS.txt').write_text(f'COMPLETED {expected}\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
