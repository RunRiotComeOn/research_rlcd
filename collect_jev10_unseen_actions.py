"""Cache frozen answer decisions and features for unseen confidence fitting."""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from peft import PeftModel

from core import CONFIDENCES
from decision_policy_v2 import action_tokens, confidence_tokens, candidate_logits
from model import append_tokens, load_model, prepare_video, to_device

ROOT = Path('/pfs/hyx/videojev-rlcd')
START = ROOT / 'runs/jev10_action_sft_v1/checkpoints/step_4813'
DATA = ROOT / 'data/jev10_unseen_calibration_v1'
OUT = ROOT / 'calibration/jev10_unseen_v1'


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', choices=('fit', 'development', 'final'), required=True)
    parser.add_argument('--device', required=True)
    args = parser.parse_args()
    data = DATA / f'{args.split}.jsonl'
    rows = [json.loads(s) for s in data.read_text(encoding='utf-8').splitlines()]
    out = OUT / args.split
    if out.exists():
        raise FileExistsError(out)
    out.mkdir(parents=True)
    config = {'split': args.split, 'data': str(data), 'data_sha256': sha256(data),
              'answer_adapter': str(START),
              'answer_adapter_sha256': sha256(START / 'adapter_model.safetensors'),
              'device': args.device, 'count': len(rows)}
    (out / 'config.json').write_text(json.dumps(config, indent=2) + '\n')
    cfg = yaml.safe_load((ROOT / 'config.yaml').read_text(encoding='utf-8'))
    processor, model = load_model(cfg['model']['path'], args.device)
    model.requires_grad_(False)
    model.model.language_model = PeftModel.from_pretrained(
        model.model.language_model, str(START), is_trainable=False)
    model.to(args.device).eval()
    q_values = torch.tensor(CONFIDENCES, device=args.device)
    features = None
    start = time.monotonic()
    with (out / 'predictions.jsonl').open('w', encoding='utf-8') as stream:
        for i, row in enumerate(rows):
            batch = to_device(prepare_video(processor, row, cfg['media']['max_frames'],
                                            cfg['media']['max_pixels']), args.device)
            if batch['input_ids'].shape[1] > cfg['media']['max_prompt_tokens']:
                raise ValueError(f'Prompt too long: {row["id"]}')
            action_prefix, action_ids = action_tokens(processor, len(row['choices']))
            with torch.inference_mode():
                output = model(**append_tokens(batch, action_prefix), use_cache=False,
                               logits_to_keep=1, output_hidden_states=True)
                action_logits = output.logits[0, -1, action_ids].float()
                action_probs = action_logits.softmax(-1)
                action = int(action_probs.argmax())
                hidden = output.hidden_states[-1][0, -1].detach().cpu().to(torch.float16).numpy()
                digit_prefix, digit_ids = confidence_tokens(processor, action)
                digit_probs = candidate_logits(model, batch, digit_prefix, digit_ids).softmax(-1)
                original_q = float((digit_probs * q_values).sum())
            if features is None:
                features = np.lib.format.open_memmap(out / 'action_hidden.npy', mode='w+',
                                                      dtype=np.float16, shape=(len(rows), len(hidden)))
            features[i] = hidden
            item = {'id': row['id'], 'video_path': row['video_path'],
                    'data_source': row['data_source'], 'action_index': action,
                    'correct': int(action == row['correct_choice']),
                    'gold_action_index': row['correct_choice'],
                    'action_probabilities': action_probs.tolist(),
                    'original_q': original_q,
                    'original_digit_probabilities': digit_probs.tolist()}
            stream.write(json.dumps(item, ensure_ascii=False) + '\n')
            stream.flush()
            if (i + 1) % 100 == 0 or i + 1 == len(rows):
                features.flush()
                print(json.dumps({'split': args.split, 'processed': i+1, 'total': len(rows),
                                  'elapsed_seconds': round(time.monotonic()-start)}), flush=True)
    (out / 'STATUS.txt').write_text(f'COMPLETED {len(rows)}\n')


if __name__ == '__main__':
    main()
