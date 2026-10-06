"""Train one of four frozen-F0 confidence LoRA ablations on fold1 train only."""
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

from core import CONFIDENCES
from decision_policy_v2 import candidate_logits, confidence_tokens
from model import load_model, prepare_video, to_device


ROOT = Path('/pfs/hyx/videojev-rlcd')
DATA = ROOT / 'data/f0_confidence_ablation_v1/train.jsonl'
ACTIONS = ROOT / 'data/f0_confidence_ablation_v1/train_actions.jsonl'
F0_ADAPTER = ROOT / 'runs/jevfull_action_fold0_v1/checkpoints/examples_24063'
SEED = 20261007
ACCUMULATE = 8
SAVE_EVERY = 4000
VARIANTS = {'base_low': ('base', 2e-6), 'base_high': ('base', 2e-5),
            'f0_low': ('f0', 2e-6), 'f0_high': ('f0', 2e-5)}


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def set_lr(optimizer, update, total, maximum):
    phase = (update-1)/max(1, total-1)
    value = maximum*0.1+maximum*0.9*0.5*(1+math.cos(math.pi*phase))
    for group in optimizer.param_groups:
        group['lr'] = value
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--variant', choices=VARIANTS, required=True)
    parser.add_argument('--device', required=True)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--smoke-examples', type=int, default=0)
    args = parser.parse_args()
    init, max_lr = VARIANTS[args.variant]
    rows = [json.loads(s) for s in DATA.read_text(encoding='utf-8').splitlines()]
    actions = [json.loads(s) for s in ACTIONS.read_text(encoding='utf-8').splitlines()]
    assert len(rows) == len(actions) == 20063
    assert [r['id'] for r in rows] == [r['id'] for r in actions]
    assert all(a['source_answer_fold'] == 'fold0' for a in actions)
    if args.smoke_examples:
        assert 1 <= args.smoke_examples <= len(rows)
        rows = rows[:args.smoke_examples]
        actions = actions[:args.smoke_examples]
    out_name = (f'f0conf_{args.variant}_smoke_{args.smoke_examples}'
                if args.smoke_examples else f'f0conf_{args.variant}_v1')
    out = ROOT / 'runs' / out_name
    if out.exists() and not args.resume:
        raise FileExistsError(out)
    if args.resume and not out.exists():
        raise FileNotFoundError(out)
    total = len(rows)
    total_updates = math.ceil(total/ACCUMULATE)
    order = list(range(total))
    random.Random(SEED).shuffle(order)
    cfg = yaml.safe_load((ROOT / 'config.yaml').read_text(encoding='utf-8'))
    config = {'variant': args.variant, 'initialization': init, 'max_lr': max_lr,
              'min_lr': max_lr*0.1, 'data': str(DATA), 'data_sha256': sha256(DATA),
              'actions_sha256': sha256(ACTIONS), 'f0_adapter': str(F0_ADAPTER),
              'f0_adapter_sha256': sha256(F0_ADAPTER / 'adapter_model.safetensors'),
              'examples': total, 'model': cfg['model']['path'], 'lora_rank': 8,
              'objective': 'squared error of mean ten-bin confidence on frozen F0 action',
              'seed': SEED, 'effective_batch': ACCUMULATE, 'epochs': 1,
              'save_every': SAVE_EVERY, 'device': args.device}
    if not args.resume:
        out.mkdir(parents=True)
        (out / 'config.json').write_text(json.dumps(config, indent=2) + '\n')
    elif json.loads((out / 'config.json').read_text()) != config:
        raise ValueError('Resume config mismatch')
    completed = []
    if args.resume:
        for path in (out / 'checkpoints').glob('examples_*'):
            marker = path / 'STATUS.txt'
            if (marker.exists() and (path / 'optimizer.pt').exists() and
                    marker.read_text().strip() == f'COMPLETED {int(path.name.split("_")[1])}'):
                completed.append((int(path.name.split('_')[1]), path))
        if not completed:
            raise ValueError('No complete checkpoint with optimizer to resume')
    start, checkpoint = max(completed) if completed else (0, None)
    torch.manual_seed(SEED)
    processor, model = load_model(cfg['model']['path'], args.device,
                                  lora_rank=8 if init == 'base' and not checkpoint else 0)
    if checkpoint:
        model.requires_grad_(False)
        model.model.language_model = PeftModel.from_pretrained(
            model.model.language_model, str(checkpoint), is_trainable=True)
    elif init == 'f0':
        model.requires_grad_(False)
        model.model.language_model = PeftModel.from_pretrained(
            model.model.language_model, str(F0_ADAPTER), is_trainable=True)
    model.to(args.device).train()
    trainable = [(n,p) for n,p in model.named_parameters() if p.requires_grad]
    if not trainable or any('lora_' not in n for n,_ in trainable):
        raise RuntimeError('Unexpected trainable parameters')
    optimizer = torch.optim.AdamW((p for _,p in trainable), lr=max_lr, weight_decay=0)
    if checkpoint:
        state = torch.load(checkpoint / 'optimizer.pt', map_location='cpu', weights_only=True)
        optimizer.load_state_dict(state['optimizer'])
        torch.set_rng_state(state['cpu_rng'])
        torch.cuda.set_rng_state(state['cuda_rng'], device=args.device)
    q_values = torch.tensor(CONFIDENCES, dtype=torch.float32, device=args.device)
    tokens = {a: confidence_tokens(processor, a) for a in {r['action_index'] for r in actions}}
    started = time.monotonic()
    optimizer.zero_grad(set_to_none=True)
    running = []
    with (out / 'metrics.jsonl').open('a' if args.resume else 'w') as stream:
        for step in range(start+1, total+1):
            row, action = rows[order[step-1]], actions[order[step-1]]
            batch = to_device(prepare_video(processor, row, cfg['media']['max_frames'],
                                            cfg['media']['max_pixels']), args.device)
            if batch['input_ids'].shape[1] > cfg['media']['max_prompt_tokens']:
                raise ValueError(f'Prompt too long: {row["id"]}')
            prefix, ids = tokens[action['action_index']]
            probs = candidate_logits(model, batch, prefix, ids).softmax(-1)
            mean_q = (probs*q_values).sum()
            loss = (mean_q-action['correct']).square()
            group_size = min(ACCUMULATE, total-((step-1)//ACCUMULATE)*ACCUMULATE)
            (loss/group_size).backward()
            running.append(float(loss.detach()))
            if step % ACCUMULATE == 0 or step == total:
                update = math.ceil(step/ACCUMULATE)
                lr = set_lr(optimizer, update, total_updates, max_lr)
                norm = torch.nn.utils.clip_grad_norm_([p for _,p in trainable], 1.0,
                                                       error_if_nonfinite=True)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            else:
                lr, norm = None, None
            if step % 100 == 0 or step == total:
                n = min(100, len(running))
                entry = {'examples': step, 'updates': math.ceil(step/ACCUMULATE),
                         'mean_online_brier_last_100': sum(running[-n:])/n,
                         'mean_q_last': float(mean_q.detach()), 'learning_rate': lr,
                         'grad_norm': float(norm) if norm is not None else None,
                         'elapsed_seconds': round(time.monotonic()-started)}
                stream.write(json.dumps(entry) + '\n')
                stream.flush()
                print(json.dumps(entry), flush=True)
            if step % SAVE_EVERY == 0 or step == total:
                path = out / 'checkpoints' / f'examples_{step:05d}'
                model.model.language_model.save_pretrained(path)
                torch.save({'optimizer': optimizer.state_dict(),
                            'cpu_rng': torch.get_rng_state(),
                            'cuda_rng': torch.cuda.get_rng_state(device=args.device)}, path / 'optimizer.pt')
                (path / 'STATUS.txt').write_text(f'COMPLETED {step}\n')
                for old in (out / 'checkpoints').glob('examples_*/optimizer.pt'):
                    if old != path / 'optimizer.pt':
                        old.unlink()
    (out / 'STATUS.txt').write_text(f'COMPLETED {total}\n')


if __name__ == '__main__':
    main()
