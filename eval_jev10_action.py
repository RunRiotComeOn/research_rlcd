"""Evaluate restricted-letter action accuracy on the JeV 10% experiment."""
import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import torch
import yaml
from peft import PeftModel

from model import load_model, native_action_scores, prepare_video, to_device

ROOT = Path('/pfs/hyx/videojev-rlcd')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--split', choices=('validation', 'test'), required=True)
    p.add_argument('--name', required=True)
    p.add_argument('--adapter')
    p.add_argument('--device', default='cuda:1')
    args = p.parse_args()
    out = ROOT / 'evaluations/jev10_action' / args.name
    if out.exists():
        raise FileExistsError(f'Refusing to overwrite {out}')
    out.mkdir(parents=True)
    cfg = yaml.safe_load((ROOT / 'config.yaml').read_text())
    rows = [json.loads(s) for s in (ROOT / f'data/jev10_v2/{args.split}.jsonl').read_text().splitlines()]
    processor, model = load_model(cfg['model']['path'], args.device)
    if args.adapter:
        model.model.language_model = PeftModel.from_pretrained(model.model.language_model,
                                                               args.adapter, is_trainable=False)
        model.to(args.device)
    model.eval()
    results = []
    with (out / 'predictions.jsonl').open('w') as f:
        for i, row in enumerate(rows, start=1):
            batch = to_device(prepare_video(processor, row, cfg['media']['max_frames'],
                                            cfg['media']['max_pixels']), args.device)
            with torch.no_grad():
                scores = native_action_scores(model, processor, batch, len(row['choices']))
            probs = scores['probabilities']
            predicted = max(range(len(probs)), key=probs.__getitem__)
            result = {'id': row['id'], 'data_source': row['data_source'],
                      'gold_action': chr(65+row['correct_choice']),
                      'predicted_action': chr(65+predicted),
                      'correct': int(predicted == row['correct_choice']),
                      'native_probabilities': probs,
                      'label_mass': scores['label_mass'],
                      'selected_p': probs[predicted], 'gold_p': probs[row['correct_choice']]}
            results.append(result)
            f.write(json.dumps(result) + '\n')
            f.flush()
            if i % 50 == 0 or i == len(rows):
                print(json.dumps({'processed': i, 'total': len(rows)}), flush=True)
    by_source = defaultdict(list)
    for r in results:
        by_source[r['data_source']].append(r)
    summary = {
        'split': args.split, 'rows': len(results), 'adapter': args.adapter,
        'accuracy': sum(r['correct'] for r in results)/len(results),
        'nll': -sum(math.log(max(r['gold_p'], 1e-8)) for r in results)/len(results),
        'mean_selected_p': sum(r['selected_p'] for r in results)/len(results),
        'mean_label_mass': sum(r['label_mass'] for r in results)/len(results),
        'by_source': {k: {'rows': len(v), 'accuracy': sum(r['correct'] for r in v)/len(v)}
                      for k, v in sorted(by_source.items())},
    }
    (out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    (out / 'STATUS.txt').write_text('COMPLETED\n')
    print(json.dumps({k: v for k, v in summary.items() if k != 'by_source'}), flush=True)


if __name__ == '__main__':
    main()
