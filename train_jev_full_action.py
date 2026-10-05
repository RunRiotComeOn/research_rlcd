"""One-pass answer LoRA training on a complete JeV fold or the full pool."""
import argparse
import hashlib
import json
import math
import random
import time
from pathlib import Path

import torch
import yaml
from peft import PeftModel

from decision_policy_v2 import action_tokens, candidate_logits
from model import load_model, prepare_video, to_device

ROOT = Path('/pfs/hyx/videojev-rlcd')
DATA_DIR = ROOT / 'data/jev_full_holmes_v1'
SEED = 20261005
ACCUMULATE = 8
MAX_LR = 1e-5
MIN_LR = 1e-6


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def set_lr(optimizer, update, total):
    phase = (update - 1) / max(1, total - 1)
    value = MIN_LR + (MAX_LR-MIN_LR)*0.5*(1+math.cos(math.pi*phase))
    for group in optimizer.param_groups:
        group['lr'] = value
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=('fold0', 'fold1', 'all'), required=True)
    parser.add_argument('--device', required=True)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--max-examples', type=int, default=0, help='Smoke test only')
    args = parser.parse_args()
    data = DATA_DIR / ('jev_all.jsonl' if args.dataset == 'all' else f'folds/{args.dataset}.jsonl')
    expected = 48126 if args.dataset == 'all' else 24063
    rows = [json.loads(s) for s in data.read_text(encoding='utf-8').splitlines()]
    if len(rows) != expected:
        raise ValueError(f'Expected {expected} rows, got {len(rows)}')
    if args.max_examples:
        if not 1 <= args.max_examples <= expected:
            raise ValueError('Bad smoke-test size')
        rows = rows[:args.max_examples]
    run_name = f'jevfull_action_{args.dataset}_v1' + (f'_smoke_{args.max_examples}' if args.max_examples else '')
    out = ROOT / 'runs' / run_name
    if out.exists() and not args.resume:
        raise FileExistsError(out)
    if args.resume and not out.exists():
        raise FileNotFoundError(out)
    total = len(rows)
    total_updates = math.ceil(total / ACCUMULATE)
    order = list(range(total))
    random.Random(SEED + {'fold0': 0, 'fold1': 1, 'all': 2}[args.dataset]).shuffle(order)
    cfg = yaml.safe_load((ROOT / 'config.yaml').read_text(encoding='utf-8'))
    config = {'dataset': args.dataset, 'data': str(data), 'data_sha256': sha256(data),
              'examples': total, 'model': cfg['model']['path'], 'lora_rank': 8,
              'objective': 'restricted-choice action cross entropy',
              'seed': SEED, 'effective_batch': ACCUMULATE, 'epochs': 1,
              'max_lr': MAX_LR, 'min_lr': MIN_LR, 'device': args.device}
    if not args.resume:
        out.mkdir(parents=True)
        (out / 'config.json').write_text(json.dumps(config, indent=2) + '\n')
    elif json.loads((out / 'config.json').read_text()) != config:
        raise ValueError('Resume config mismatch')
    completed = []
    if args.resume:
        for path in (out / 'checkpoints').glob('examples_*'):
            marker = path / 'STATUS.txt'
            if marker.exists():
                n = int(path.name.split('_')[1])
                if marker.read_text().strip() == f'COMPLETED {n}':
                    completed.append((n, path))
        if not completed:
            raise ValueError('No complete checkpoint to resume')
    start, checkpoint = max(completed) if completed else (0, None)
    torch.manual_seed(SEED)
    processor, model = load_model(cfg['model']['path'], args.device, lora_rank=0 if checkpoint else 8)
    if checkpoint:
        model.requires_grad_(False)
        model.model.language_model = PeftModel.from_pretrained(
            model.model.language_model, str(checkpoint), is_trainable=True)
    model.to(args.device).train()
    trainable = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    if not trainable or any('lora_' not in n for n, _ in trainable):
        raise RuntimeError('Unexpected trainable parameters')
    optimizer = torch.optim.AdamW((p for _, p in trainable), lr=MAX_LR, weight_decay=0)
    if checkpoint:
        state = torch.load(checkpoint / 'optimizer.pt', map_location='cpu', weights_only=True)
        optimizer.load_state_dict(state['optimizer'])
        torch.set_rng_state(state['cpu_rng'])
        torch.cuda.set_rng_state(state['cuda_rng'], device=args.device)
    started = time.monotonic()
    optimizer.zero_grad(set_to_none=True)
    token_cache = {n: action_tokens(processor, n) for n in {len(r['choices']) for r in rows}}
    with (out / 'metrics.jsonl').open('a' if args.resume else 'w') as stream:
        for step in range(start + 1, total + 1):
            row = rows[order[step-1]]
            batch = to_device(prepare_video(processor, row, cfg['media']['max_frames'],
                                            cfg['media']['max_pixels']), args.device)
            if batch['input_ids'].shape[1] > cfg['media']['max_prompt_tokens']:
                raise ValueError(f'Prompt too long: {row["id"]}')
            prefix, ids = token_cache[len(row['choices'])]
            logits = candidate_logits(model, batch, prefix, ids)
            gold = torch.tensor([row['correct_choice']], device=args.device)
            loss = torch.nn.functional.cross_entropy(logits.unsqueeze(0), gold)
            group_size = min(ACCUMULATE, total - ((step-1)//ACCUMULATE)*ACCUMULATE)
            (loss / group_size).backward()
            if step % ACCUMULATE == 0 or step == total:
                update = math.ceil(step / ACCUMULATE)
                lr = set_lr(optimizer, update, total_updates)
                norm = torch.nn.utils.clip_grad_norm_([p for _,p in trainable], 1.0,
                                                       error_if_nonfinite=True)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            else:
                update, lr, norm = None, None, None
            if step % 100 == 0 or step == total:
                entry = {'examples': step, 'updates': math.ceil(step/ACCUMULATE),
                         'loss': float(loss.detach()),
                         'correct_before_update': int(int(logits.argmax()) == row['correct_choice']),
                         'learning_rate': lr, 'grad_norm': float(norm) if norm is not None else None,
                         'elapsed_seconds': round(time.monotonic()-started)}
                stream.write(json.dumps(entry) + '\n')
                stream.flush()
                print(json.dumps(entry), flush=True)
            if step % 4000 == 0 or step == total:
                path = out / 'checkpoints' / f'examples_{step:05d}'
                model.model.language_model.save_pretrained(path)
                torch.save({'optimizer': optimizer.state_dict(),
                            'cpu_rng': torch.get_rng_state(),
                            'cuda_rng': torch.cuda.get_rng_state(device=args.device)}, path / 'optimizer.pt')
                (path / 'STATUS.txt').write_text(f'COMPLETED {step}\n')
    (out / 'STATUS.txt').write_text(f'COMPLETED {total}\n')


if __name__ == '__main__':
    main()
