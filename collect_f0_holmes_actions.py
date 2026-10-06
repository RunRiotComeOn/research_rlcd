"""Collect frozen F0 greedy answers and probabilities on all Holmes questions."""
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
DATA = ROOT / 'data/jev_full_holmes_v1/holmes_all.jsonl'
RUN = ROOT / 'runs/jevfull_action_fold0_v1'
ADAPTER = RUN / 'checkpoints/examples_24063'
OUT = ROOT / 'evaluations/f0_confidence_ablation_v1/holmes_actions'


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', required=True)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if (RUN / 'STATUS.txt').read_text().strip() != 'COMPLETED 24063':
        raise ValueError('F0 answer model incomplete')
    rows = [json.loads(s) for s in DATA.read_text().splitlines()]
    if len(rows) != 1837:
        raise ValueError('Holmes size mismatch')
    config = {'data': str(DATA), 'data_sha256': sha256(DATA),
              'adapter': str(ADAPTER), 'adapter_sha256': sha256(ADAPTER / 'adapter_model.safetensors'),
              'device': args.device, 'rows': len(rows)}
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
    tokens = {n: action_tokens(processor, n) for n in {len(r['choices']) for r in rows}}
    with (OUT / 'predictions.jsonl').open('a' if args.resume else 'w') as stream:
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
                    'data_source': row['data_source'], 'source_answer_fold': 'fold0',
                    'action_index': action, 'gold_action_index': row['correct_choice'],
                    'correct': int(action == row['correct_choice']),
                    'action_probabilities': probs.tolist()}
            stream.write(json.dumps(item, ensure_ascii=False) + '\n')
            stream.flush()
            if (i+1) % 100 == 0 or i+1 == len(rows):
                print(json.dumps({'processed': i+1, 'total': len(rows)}), flush=True)
    (OUT / 'STATUS.txt').write_text(f'COMPLETED {len(rows)}\n')


if __name__ == '__main__':
    main()
