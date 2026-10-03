"""Evaluate context-correct direct confidence after matched reward runs."""
import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import torch
import yaml
from peft import PeftModel

from core import calibration_metrics, parse_output
from decision_policy_v2 import greedy_decision
from model import load_model, prepare_video, to_device

ROOT = Path('/pfs/hyx/videojev-rlcd')
START = ROOT / 'runs/jev10_action_sft_v1/checkpoints/step_4813'


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def metrics(rows, key):
    result = calibration_metrics([{'correct': r['correct'], 'confidence': r[key]} for r in rows])
    return {field: result[field] for field in ('brier','ece','nll')}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=('baseline','old_brier','proposed'), required=True)
    parser.add_argument('--step', type=int, choices=(100,200,300))
    parser.add_argument('--device', required=True)
    parser.add_argument('--dataset', choices=('validation','blind'), default='validation')
    args = parser.parse_args()
    if args.dataset == 'validation':
        data_file = ROOT/'data/jev10_v2/validation.jsonl'
        out_root = ROOT/'evaluations/jev10_reward_compare_v2'
        expected_rows = 481
    else:
        data_file = ROOT/'data/jev10_reward_blind_v1/blind.jsonl'
        out_root = ROOT/'evaluations/jev10_reward_blind_v1'
        expected_rows = 1000
    if args.mode == 'baseline':
        if args.step is not None:
            raise ValueError('Baseline has no step')
        adapter = START
        name = 'baseline'
    else:
        if args.step is None:
            raise ValueError('Step required')
        adapter = ROOT / f'runs/jev10_reward_{args.mode}_v2_300/checkpoints/step_{args.step:04d}'
        name = f'{args.mode}_step_{args.step:04d}'
        run = adapter.parent.parent
        if (run/'STATUS.txt').read_text().strip() != 'COMPLETED 300':
            raise ValueError(f'Training incomplete: {run}')
    out = out_root/name
    if out.exists():
        raise FileExistsError(out)
    cfg = yaml.safe_load((ROOT/'config.yaml').read_text(encoding='utf-8'))
    rows = [json.loads(s) for s in data_file.read_text(encoding='utf-8').splitlines()]
    if len(rows) != expected_rows:
        raise ValueError(f'{args.dataset} row count changed')
    if args.dataset == 'blind':
        manifest = json.loads((data_file.parent/'manifest.json').read_text(encoding='utf-8'))
        if manifest['output_sha256'] != sha256(data_file):
            raise ValueError('Blind dataset checksum changed')
    out.mkdir(parents=True)
    run_cfg = {'mode': args.mode, 'step': args.step, 'dataset': args.dataset, 'adapter': str(adapter),
               'adapter_weights_sha256': sha256(adapter/'adapter_model.safetensors'),
               'data': str(data_file), 'data_sha256': sha256(data_file),
               'device': args.device,
               'max_frames': cfg['media']['max_frames'],
               'max_pixels': cfg['media']['max_pixels']}
    (out/'run_config.json').write_text(json.dumps(run_cfg, indent=2)+'\n')
    processor, model = load_model(cfg['model']['path'], args.device)
    model.requires_grad_(False)
    model.model.language_model = PeftModel.from_pretrained(
        model.model.language_model, str(adapter), is_trainable=False)
    model.to(args.device).eval()
    results = []
    with (out/'predictions.jsonl').open('w', encoding='utf-8') as stream:
        for i,row in enumerate(rows,1):
            batch = to_device(prepare_video(processor, row, cfg['media']['max_frames'],
                                            cfg['media']['max_pixels']), args.device)
            with torch.no_grad():
                decision = greedy_decision(model, processor, batch, len(row['choices']))
            parsed = parse_output(decision['output'], len(row['choices']))
            if not parsed.valid_format or parsed.action_index != decision['action_index']:
                raise ValueError('Invalid constrained output')
            item = {'id': row['id'], 'video_path': row['video_path'],
                    'data_source': row['data_source'],
                    'correct': int(decision['action_index'] == row['correct_choice']),
                    'gold_action': chr(65+row['correct_choice']),
                    'predicted_action': chr(65+decision['action_index']),
                    **decision}
            results.append(item)
            stream.write(json.dumps(item, ensure_ascii=False)+'\n')
            stream.flush()
            if i % 100 == 0 or i == len(rows):
                print(json.dumps({'processed': i, 'total': len(rows)}), flush=True)
    by_source = defaultdict(list)
    for row in results:
        by_source[row['data_source']].append(row)
    summary = {'count': len(results), 'mode': args.mode, 'step': args.step,
               'dataset': args.dataset,
               'adapter': str(adapter),
               'correct': sum(r['correct'] for r in results),
               'accuracy': sum(r['correct'] for r in results)/len(results),
               'direct': metrics(results,'confidence'),
               'expected': metrics(results,'expected_q'),
               'mean_reported_confidence': sum(r['confidence'] for r in results)/len(results),
               'confidence_bins': dict(sorted(Counter(r['confidence_bin'] for r in results).items())),
               'by_source': {k:{'rows':len(v),'accuracy':sum(r['correct'] for r in v)/len(v)}
                             for k,v in sorted(by_source.items())}}
    (out/'summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False)+'\n')
    (out/'STATUS.txt').write_text(f'COMPLETED {expected_rows}\n')
    print(json.dumps({k:v for k,v in summary.items() if k!='by_source'}), flush=True)


if __name__ == '__main__':
    main()
