"""Evaluate direct two-token RLCD confidence on video-disjoint JeV validation."""
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import torch
import yaml
from peft import PeftModel

from constrained_policy import greedy_decision
from core import calibration_metrics
from model import load_model, prepare_video, to_device

ROOT = Path('/pfs/hyx/videojev-rlcd')
START = ROOT / 'runs/jev10_action_sft_v1/checkpoints/step_4813'


def metric(rows, key):
    source = [{'correct': r['correct'], 'confidence': r[key]} for r in rows]
    result = calibration_metrics(source)
    return {k: result[k] for k in ('brier', 'ece', 'nll')}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run-name', required=True)
    p.add_argument('--adapter', default=str(START))
    p.add_argument('--split', choices=('validation', 'test'), default='validation')
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--limit', type=int)
    args = p.parse_args()
    out = ROOT / 'evaluations/jev10_constrained' / args.run_name
    if out.exists():
        raise FileExistsError(f'Refusing to overwrite {out}')
    out.mkdir(parents=True)
    cfg = yaml.safe_load((ROOT / 'config.yaml').read_text())
    data = [json.loads(s) for s in (ROOT / f'data/jev10_v2/{args.split}.jsonl').read_text().splitlines()]
    if args.limit:
        data = data[:args.limit]
    processor, model = load_model(cfg['model']['path'], args.device)
    model.requires_grad_(False)
    model.model.language_model = PeftModel.from_pretrained(model.model.language_model,
                                                           args.adapter, is_trainable=False)
    model.to(args.device).eval()
    results = []
    with (out / 'predictions.jsonl').open('w') as f:
        for i, row in enumerate(data, start=1):
            batch = to_device(prepare_video(processor, row, cfg['media']['max_frames'],
                                            cfg['media']['max_pixels']), args.device)
            with torch.no_grad():
                decision = greedy_decision(model, processor, batch, len(row['choices']))
            item = {
                'id': row['id'], 'data_source': row['data_source'],
                'correct': int(decision['action_index'] == row['correct_choice']),
                'gold_action': chr(65+row['correct_choice']),
                'predicted_action': chr(65+decision['action_index']),
                **decision,
            }
            results.append(item)
            f.write(json.dumps(item) + '\n')
            f.flush()
            if i % 50 == 0 or i == len(data):
                print(json.dumps({'processed': i, 'total': len(data)}), flush=True)
    grouped = defaultdict(list)
    for r in results:
        grouped[r['data_source']].append(r)
    summary = {
        'count': len(results), 'split': args.split, 'adapter': args.adapter,
        'accuracy': sum(r['correct'] for r in results)/len(results),
        'invalid_format_rate': 0.0,
        'direct': metric(results, 'confidence'),
        'expected': metric(results, 'expected_q'),
        'confidence_bins': dict(sorted(Counter(r['confidence_bin'] for r in results).items())),
        'by_source': {k: {'count': len(v), 'accuracy': sum(r['correct'] for r in v)/len(v)}
                      for k,v in sorted(grouped.items())},
    }
    (out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    (out / 'STATUS.txt').write_text('COMPLETED\n')
    print(json.dumps({k:v for k,v in summary.items() if k != 'by_source'}), flush=True)


if __name__ == '__main__':
    main()
