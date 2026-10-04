"""Evaluate a confidence adapter on cached frozen actions without answer changes."""
import argparse
import hashlib
import json
from pathlib import Path

import torch
import yaml
from peft import PeftModel

from core import CONFIDENCES
from decision_policy_v2 import candidate_logits, confidence_tokens
from model import load_model, prepare_video, to_device

ROOT = Path('/pfs/hyx/videojev-rlcd')
DATA = ROOT / 'data/jev10_unseen_calibration_v1'
LABELS = ROOT / 'calibration/jev10_unseen_v1'
RUN = ROOT / 'runs/jev10_unseen_confidence_v1'
OUT = ROOT / 'evaluations/jev10_unseen_confidence_v1'


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', choices=('development', 'final'), required=True)
    parser.add_argument('--examples', type=int, choices=(1000, 2000, 3000, 4000), required=True)
    parser.add_argument('--device', required=True)
    args = parser.parse_args()
    adapter = RUN / 'checkpoints' / f'examples_{args.examples:04d}'
    if not adapter.is_dir() or (adapter / 'STATUS.txt').read_text().strip() != f'COMPLETED {args.examples}':
        raise ValueError(f'Incomplete checkpoint: {adapter}')
    data = DATA / f'{args.split}.jsonl'
    labels = LABELS / args.split / 'predictions.jsonl'
    rows = [json.loads(s) for s in data.read_text(encoding='utf-8').splitlines()]
    decisions = [json.loads(s) for s in labels.read_text(encoding='utf-8').splitlines()]
    if len(rows) != len(decisions) or [r['id'] for r in rows] != [r['id'] for r in decisions]:
        raise ValueError('Data/labels mismatch')
    out = OUT / f'examples_{args.examples:04d}' / args.split
    if out.exists():
        raise FileExistsError(out)
    out.mkdir(parents=True)
    (out / 'config.json').write_text(json.dumps({
        'adapter': str(adapter), 'adapter_sha256': sha256(adapter / 'adapter_model.safetensors'),
        'data': str(data), 'data_sha256': sha256(data),
        'labels': str(labels), 'labels_sha256': sha256(labels), 'device': args.device}, indent=2) + '\n')
    cfg = yaml.safe_load((ROOT / 'config.yaml').read_text(encoding='utf-8'))
    processor, model = load_model(cfg['model']['path'], args.device)
    model.requires_grad_(False)
    model.model.language_model = PeftModel.from_pretrained(
        model.model.language_model, str(adapter), is_trainable=False)
    model.to(args.device).eval()
    q_values = torch.tensor(CONFIDENCES, dtype=torch.float32, device=args.device)
    output = []
    with (out / 'predictions.jsonl').open('w', encoding='utf-8') as stream:
        for i, (row, decision) in enumerate(zip(rows, decisions), 1):
            batch = to_device(prepare_video(processor, row, cfg['media']['max_frames'],
                                            cfg['media']['max_pixels']), args.device)
            prefix, ids = confidence_tokens(processor, decision['action_index'])
            with torch.inference_mode():
                probs = candidate_logits(model, batch, prefix, ids).softmax(-1)
                q = float((probs*q_values).sum())
            item = {'id': row['id'], 'correct': decision['correct'], 'action_index': decision['action_index'],
                    'q': q, 'digit_probabilities': probs.tolist()}
            stream.write(json.dumps(item) + '\n')
            output.append(item)
            if i % 100 == 0 or i == len(rows):
                print(json.dumps({'split': args.split, 'examples': args.examples, 'processed': i}), flush=True)
    ys = [r['correct'] for r in output]
    qs = [r['q'] for r in output]
    accuracy = sum(ys)/len(ys)
    summary = {'split': args.split, 'training_examples': args.examples, 'rows': len(rows),
               'accuracy': accuracy, 'mean_q': sum(qs)/len(qs),
               'brier': sum((q-y)**2 for q,y in zip(qs,ys))/len(qs),
               'constant_oracle_brier': accuracy*(1-accuracy)}
    (out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    (out / 'STATUS.txt').write_text(f'COMPLETED {len(rows)}\n')
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
