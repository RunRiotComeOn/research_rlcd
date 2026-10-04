"""Train confidence LoRA on genuinely unseen frozen-answer decisions."""
import argparse
import hashlib
import json
import random
import time
from pathlib import Path

import torch
import yaml
from peft import PeftModel

from core import CONFIDENCES
from decision_policy_v2 import candidate_logits, confidence_tokens
from model import load_model, prepare_video, to_device

ROOT = Path('/pfs/hyx/videojev-rlcd')
START = ROOT / 'runs/jev10_action_sft_v1/checkpoints/step_4813'
DATA = ROOT / 'data/jev10_unseen_calibration_v1/fit.jsonl'
LABELS = ROOT / 'calibration/jev10_unseen_v1/fit/predictions.jsonl'
OUT = ROOT / 'runs/jev10_unseen_confidence_v1'
SEED = 20261005
ACCUMULATE = 8
LR = 2e-5


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', required=True)
    args = parser.parse_args()
    if OUT.exists():
        raise FileExistsError(OUT)
    if (LABELS.parent / 'STATUS.txt').read_text().strip() != 'COMPLETED 4000':
        raise ValueError('Frozen-action labels incomplete')
    rows = [json.loads(s) for s in DATA.read_text(encoding='utf-8').splitlines()]
    labels = [json.loads(s) for s in LABELS.read_text(encoding='utf-8').splitlines()]
    if len(rows) != 4000 or len(labels) != 4000 or [r['id'] for r in rows] != [r['id'] for r in labels]:
        raise ValueError('Data/label mismatch')
    order = list(range(len(rows)))
    random.Random(SEED).shuffle(order)
    OUT.mkdir(parents=True)
    config = {'objective': 'Brier of mean confidence token distribution on cached greedy answer',
              'seed': SEED, 'examples': len(rows), 'epochs': 1,
              'effective_batch': ACCUMULATE, 'learning_rate': LR,
              'data': str(DATA), 'data_sha256': sha256(DATA),
              'labels': str(LABELS), 'labels_sha256': sha256(LABELS),
              'initial_adapter': str(START),
              'initial_adapter_sha256': sha256(START / 'adapter_model.safetensors'),
              'device': args.device}
    (OUT / 'config.json').write_text(json.dumps(config, indent=2) + '\n')
    cfg = yaml.safe_load((ROOT / 'config.yaml').read_text(encoding='utf-8'))
    torch.manual_seed(SEED)
    processor, model = load_model(cfg['model']['path'], args.device)
    model.requires_grad_(False)
    model.model.language_model = PeftModel.from_pretrained(
        model.model.language_model, str(START), is_trainable=True)
    model.to(args.device).eval()
    trainable = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    if not trainable or any('lora_' not in n for n, _ in trainable):
        raise RuntimeError('Unexpected trainable parameters')
    optimizer = torch.optim.AdamW((p for _, p in trainable), lr=LR, weight_decay=0)
    q_values = torch.tensor(CONFIDENCES, dtype=torch.float32, device=args.device)
    started = time.monotonic()
    running = []
    optimizer.zero_grad(set_to_none=True)
    with (OUT / 'metrics.jsonl').open('w') as stream:
        for step, index in enumerate(order, 1):
            row, label = rows[index], labels[index]
            batch = to_device(prepare_video(processor, row, cfg['media']['max_frames'],
                                            cfg['media']['max_pixels']), args.device)
            if batch['input_ids'].shape[1] > cfg['media']['max_prompt_tokens']:
                raise ValueError(f'Prompt too long: {row["id"]}')
            digit_prefix, digit_ids = confidence_tokens(processor, label['action_index'])
            digit_probs = candidate_logits(model, batch, digit_prefix, digit_ids).softmax(-1)
            mean_q = (digit_probs * q_values).sum()
            loss = (mean_q - label['correct']).square()
            (loss / ACCUMULATE).backward()
            running.append(float(loss.detach()))
            if step % ACCUMULATE == 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    [p for _, p in trainable], 1.0, error_if_nonfinite=True)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            else:
                grad_norm = None
            if step % 100 == 0:
                entry = {'examples': step, 'optimizer_updates': step // ACCUMULATE,
                         'mean_brier_last_100': sum(running[-100:]) / 100,
                         'mean_q_last': float(mean_q.detach()),
                         'last_correct': label['correct'],
                         'grad_norm': float(grad_norm) if grad_norm is not None else None,
                         'elapsed_seconds': round(time.monotonic() - started)}
                stream.write(json.dumps(entry) + '\n')
                stream.flush()
                print(json.dumps(entry), flush=True)
            if step % 1000 == 0:
                checkpoint = OUT / 'checkpoints' / f'examples_{step:04d}'
                model.model.language_model.save_pretrained(checkpoint)
                (checkpoint / 'STATUS.txt').write_text(f'COMPLETED {step}\n')
    (OUT / 'STATUS.txt').write_text(f'COMPLETED {len(rows)}\n')


if __name__ == '__main__':
    main()
