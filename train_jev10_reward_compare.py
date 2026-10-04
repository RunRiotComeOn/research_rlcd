"""Matched short VideoJev RLCD runs differing only in the reward equation."""
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
from decision_policy_v2 import action_tokens, candidate_logits, confidence_tokens
from model import load_model, prepare_video, to_device

ROOT = Path('/pfs/hyx/videojev-rlcd')
START = ROOT / 'runs/jev10_action_sft_v1/checkpoints/step_4813'
DATA = ROOT / 'data/jev10_v2/train.jsonl'
SEED = 20261003
BETA = 0.2
ACTION_TEMPERATURE = 2.0


def reward(y, q, mode, beta=BETA):
    if mode == 'old_brier':
        return y - beta * (q-y)**2
    if mode == 'proposed':
        return y + beta * (2*y*q-q*q)
    raise ValueError(mode)


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=('old_brier', 'proposed'), required=True)
    parser.add_argument('--device', required=True)
    parser.add_argument('--limit', type=int, default=300)
    parser.add_argument('--beta', type=float, default=BETA)
    parser.add_argument('--run-name')
    args = parser.parse_args()
    if not 1 <= args.limit <= 4813:
        raise ValueError('Invalid limit')
    if not 0 < args.beta <= 1:
        raise ValueError('beta must be in (0,1]')
    run_name = args.run_name or f'jev10_reward_{args.mode}_v2_{args.limit}'
    if Path(run_name).name != run_name:
        raise ValueError('run-name must be a single directory name')
    out = ROOT / 'runs' / run_name
    if out.exists():
        raise FileExistsError(out)
    cfg = yaml.safe_load((ROOT/'config.yaml').read_text(encoding='utf-8'))
    rows = [json.loads(s) for s in DATA.read_text(encoding='utf-8').splitlines()]
    random.Random(SEED).shuffle(rows)
    rows = rows[:args.limit]
    out.mkdir(parents=True)
    run_cfg = {'mode': args.mode, 'beta': args.beta, 'seed': SEED,
               'source_adapter': str(START),
               'source_adapter_weights_sha256': sha256(START/'adapter_model.safetensors'),
               'data': str(DATA), 'data_sha256': sha256(DATA),
               'limit': args.limit, 'num_rollouts': 4,
               'action_sampling_temperature': ACTION_TEMPERATURE,
               'digit_sampling_temperature': 1.0,
               'learning_rate': 2e-5,
               'objective': 'group-centered REINFORCE without reward std division; correct context tokens',
               'device': args.device}
    (out/'run_config.json').write_text(json.dumps(run_cfg, indent=2)+'\n')
    torch.manual_seed(SEED)
    processor, model = load_model(cfg['model']['path'], args.device)
    model.requires_grad_(False)
    model.model.language_model = PeftModel.from_pretrained(
        model.model.language_model, str(START), is_trainable=True)
    model.to(args.device)
    named_trainable = [(n,p) for n,p in model.named_parameters() if p.requires_grad]
    if not named_trainable or any('lora_' not in n for n,_ in named_trainable):
        raise RuntimeError('Only language LoRA weights may train')
    trainable = [p for _,p in named_trainable]
    optimizer = torch.optim.AdamW(trainable, lr=run_cfg['learning_rate'], weight_decay=0)
    started = time.monotonic()
    with (out/'metrics.jsonl').open('w') as stream:
        for step, row in enumerate(rows, 1):
            batch = to_device(prepare_video(processor, row, cfg['media']['max_frames'],
                                            cfg['media']['max_pixels']), args.device)
            if batch['input_ids'].shape[1] > cfg['media']['max_prompt_tokens']:
                raise ValueError(f'Prompt too long: {row["id"]}')
            action_prefix, action_ids = action_tokens(processor, len(row['choices']))
            model.eval()
            with torch.no_grad():
                ap = (candidate_logits(model, batch, action_prefix, action_ids)
                      / ACTION_TEMPERATURE).softmax(-1)
                actions = torch.multinomial(ap, 4, replacement=True).tolist()
                digits = [None]*4
                digit_entropy = {}
                for action in set(actions):
                    prefix, ids = confidence_tokens(processor, action)
                    dp = candidate_logits(model, batch, prefix, ids).softmax(-1)
                    positions = [i for i,a in enumerate(actions) if a == action]
                    drawn = torch.multinomial(dp, len(positions), replacement=True).tolist()
                    for i,digit in zip(positions, drawn):
                        digits[i] = digit
                    digit_entropy[action] = float(-(dp*dp.clamp_min(1e-12).log()).sum())
            correct = [int(a == row['correct_choice']) for a in actions]
            rewards = [reward(y, CONFIDENCES[d], args.mode, args.beta) for y,d in zip(correct,digits)]
            rt = torch.tensor(rewards, dtype=torch.float32, device=args.device)
            adv = rt-rt.mean()
            entry = {'step': step, 'id': row['id'], 'sampled_actions': actions,
                     'sampled_digits': digits, 'correct': correct, 'rewards': rewards,
                     'mean_reward': float(rt.mean()),
                     'reward_std': float(rt.std(unbiased=False)),
                     'action_entropy': float(-(ap*ap.clamp_min(1e-12).log()).sum()),
                     'digit_entropy': sum(digit_entropy.values())/len(digit_entropy),
                     'skipped': bool(float(adv.abs().max()) < 1e-8)}
            if not entry['skipped']:
                model.train()
                optimizer.zero_grad(set_to_none=True)
                if len(set(actions)) > 1:
                    alogp = (candidate_logits(model, batch, action_prefix, action_ids)
                             / ACTION_TEMPERATURE).log_softmax(-1)
                    aloss = -(adv.detach()*alogp[torch.tensor(actions,device=args.device)]).mean()
                    aloss.backward()
                    entry['action_loss'] = float(aloss.detach())
                    del aloss, alogp
                for action in set(actions):
                    prefix, ids = confidence_tokens(processor, action)
                    positions = [i for i,a in enumerate(actions) if a == action]
                    dlogp = candidate_logits(model, batch, prefix, ids).log_softmax(-1)
                    dloss = -sum(adv[i].detach()*dlogp[digits[i]] for i in positions)/4
                    dloss.backward()
                    del dloss, dlogp
                norm = torch.nn.utils.clip_grad_norm_(trainable, 1.0, error_if_nonfinite=True)
                optimizer.step()
                entry['grad_norm'] = float(norm)
            entry['elapsed_seconds'] = time.monotonic()-started
            stream.write(json.dumps(entry)+'\n')
            stream.flush()
            if step % 25 == 0 or step == len(rows):
                print(json.dumps({k:v for k,v in entry.items()
                                  if k not in ('sampled_actions','sampled_digits','correct','rewards')}), flush=True)
            if step % 100 == 0 or step == len(rows):
                model.model.language_model.save_pretrained(out/'checkpoints'/f'step_{step:04d}')
    model.model.language_model.save_pretrained(out/'adapter')
    (out/'STATUS.txt').write_text(f'COMPLETED {len(rows)}\n')
    print(f'COMPLETED {args.mode}', flush=True)


if __name__ == '__main__':
    main()
