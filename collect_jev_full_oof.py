"""Predict each JeV fold using the answer LoRA trained on the opposite fold."""
import argparse
import hashlib
import json
from pathlib import Path

import torch
import yaml
from peft import PeftModel

from decision_policy_v2 import action_tokens, candidate_logits
from model import load_model, prepare_video, to_device

ROOT = Path('/pfs/hyx/videojev-rlcd')
DATA_DIR = ROOT / 'data/jev_full_holmes_v1/folds'
OUT_ROOT = ROOT / 'calibration/jevfull_oof_v1'
EXPECTED = 24063


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--target-fold', choices=('fold0', 'fold1'), required=True)
    parser.add_argument('--device', required=True)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--max-examples', type=int, default=0, help='Smoke test only')
    args = parser.parse_args()
    source_fold = 'fold1' if args.target_fold == 'fold0' else 'fold0'
    adapter_run = ROOT / f'runs/jevfull_action_{source_fold}_v1'
    if (adapter_run / 'STATUS.txt').read_text().strip() != f'COMPLETED {EXPECTED}':
        raise ValueError(f'Answer fold incomplete: {source_fold}')
    adapter = adapter_run / f'checkpoints/examples_{EXPECTED:05d}'
    data = DATA_DIR / f'{args.target_fold}.jsonl'
    rows = [json.loads(s) for s in data.read_text(encoding='utf-8').splitlines()]
    if len(rows) != EXPECTED:
        raise ValueError('Fold size mismatch')
    if args.max_examples:
        rows = rows[:args.max_examples]
    out = OUT_ROOT / (args.target_fold + (f'_smoke_{args.max_examples}' if args.max_examples else ''))
    if out.exists() and not args.resume:
        raise FileExistsError(out)
    if args.resume and not out.exists():
        raise FileNotFoundError(out)
    config = {'target_fold': args.target_fold, 'source_answer_fold': source_fold,
              'data': str(data), 'data_sha256': sha256(data),
              'adapter': str(adapter), 'adapter_sha256': sha256(adapter / 'adapter_model.safetensors'),
              'rows': len(rows), 'device': args.device}
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
            raise ValueError('Resume row order mismatch')
    cfg = yaml.safe_load((ROOT / 'config.yaml').read_text(encoding='utf-8'))
    processor, model = load_model(cfg['model']['path'], args.device)
    model.requires_grad_(False)
    model.model.language_model = PeftModel.from_pretrained(
        model.model.language_model, str(adapter), is_trainable=False)
    model.to(args.device).eval()
    tokens = {n: action_tokens(processor, n) for n in {len(r['choices']) for r in rows}}
    with (out / 'predictions.jsonl').open('a' if args.resume else 'w') as stream:
        for i in range(start, len(rows)):
            row = rows[i]
            batch = to_device(prepare_video(processor, row, cfg['media']['max_frames'],
                                            cfg['media']['max_pixels']), args.device)
            if batch['input_ids'].shape[1] > cfg['media']['max_prompt_tokens']:
                raise ValueError(f'Prompt too long: {row["id"]}')
            prefix, ids = tokens[len(row['choices'])]
            with torch.inference_mode():
                probs = candidate_logits(model, batch, prefix, ids).softmax(-1)
                action = int(probs.argmax())
            item = {'id': row['id'], 'video_path': row['video_path'],
                    'data_source': row['data_source'], 'source_answer_fold': source_fold,
                    'action_index': action, 'gold_action_index': row['correct_choice'],
                    'correct': int(action == row['correct_choice']),
                    'action_probabilities': probs.tolist()}
            stream.write(json.dumps(item, ensure_ascii=False) + '\n')
            stream.flush()
            if (i+1) % 200 == 0 or i+1 == len(rows):
                print(json.dumps({'target_fold': args.target_fold,
                                  'processed': i+1, 'total': len(rows)}), flush=True)
    (out / 'STATUS.txt').write_text(f'COMPLETED {len(rows)}\n')


if __name__ == '__main__':
    main()
