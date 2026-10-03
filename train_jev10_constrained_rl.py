"""Matched short RL-only/RLCD pilots with constrained Action and Confidence tokens."""
import argparse
import json
import random
import time
from pathlib import Path

import torch
import yaml
from peft import PeftModel

from constrained_policy import action_prefix, digit_prefix, logits_for, token_ids
from core import CONFIDENCES, assert_correctness_margin
from model import load_model, prepare_video, to_device

ROOT = Path('/pfs/hyx/videojev-rlcd')
START = ROOT / 'runs/jev10_action_sft_v1/checkpoints/step_4813'
ACTION_TEMPERATURE = 2.0


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--mode', choices=('rl_only', 'rlcd'), required=True)
    p.add_argument('--device', required=True)
    p.add_argument('--limit', type=int, default=300)
    p.add_argument('--run-name', required=True)
    args = p.parse_args()
    if not 1 <= args.limit <= 4813:
        raise ValueError('Invalid limit')
    cfg = yaml.safe_load((ROOT / 'config.yaml').read_text())
    out = ROOT / 'runs' / args.run_name
    if out.exists():
        raise FileExistsError(f'Refusing to overwrite {out}')
    out.mkdir(parents=True)
    lambda_cal = 0.0 if args.mode == 'rl_only' else 0.2
    if lambda_cal:
        assert_correctness_margin(lambda_cal)
    config = {
        'mode': args.mode, 'source_adapter': str(START), 'data': 'data/jev10_v2/train.jsonl',
        'limit': args.limit, 'seed': 20261003, 'num_rollouts': 4,
        'lambda_cal': lambda_cal, 'learning_rate': 2e-5,
        'objective': 'group-centered REINFORCE; no reward-std division',
        'action_sampling_temperature': ACTION_TEMPERATURE, 'digit_sampling_temperature': 1.0,
        'digit_gradient': args.mode == 'rlcd', 'device': args.device,
    }
    (out / 'run_config.json').write_text(json.dumps(config, indent=2) + '\n')
    rows = [json.loads(s) for s in (ROOT / 'data/jev10_v2/train.jsonl').read_text().splitlines()]
    random.Random(config['seed']).shuffle(rows)
    rows = rows[:args.limit]
    torch.manual_seed(config['seed'])
    processor, model = load_model(cfg['model']['path'], args.device)
    model.requires_grad_(False)
    model.model.language_model = PeftModel.from_pretrained(model.model.language_model,
                                                           str(START), is_trainable=True)
    model.to(args.device)
    named_trainable = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    if not named_trainable or any('lora_' not in n for n, _ in named_trainable):
        raise RuntimeError('Only language LoRA weights may train')
    trainable = [p for _, p in named_trainable]
    optimizer = torch.optim.AdamW(trainable, lr=config['learning_rate'], weight_decay=0)
    started = time.monotonic()
    with (out / 'metrics.jsonl').open('w') as log:
        for step, row in enumerate(rows, start=1):
            batch = to_device(prepare_video(processor, row, cfg['media']['max_frames'],
                                            cfg['media']['max_pixels']), args.device)
            if batch['input_ids'].shape[1] > cfg['media']['max_prompt_tokens']:
                raise ValueError(f'Prompt too long: {row["id"]}')
            action_ids, digit_ids = token_ids(processor, len(row['choices']))
            model.eval()
            with torch.no_grad():
                action_probs = (logits_for(model, batch, action_prefix(processor), action_ids)
                                / ACTION_TEMPERATURE).softmax(-1)
                sampled_actions = torch.multinomial(action_probs, 4, replacement=True).tolist()
                sampled_digits = [None]*4
                digit_entropy = {}
                for action in set(sampled_actions):
                    digit_probs = logits_for(model, batch, digit_prefix(processor, action), digit_ids).softmax(-1)
                    positions = [i for i, a in enumerate(sampled_actions) if a == action]
                    drawn = torch.multinomial(digit_probs, len(positions), replacement=True).tolist()
                    for pos, digit in zip(positions, drawn):
                        sampled_digits[pos] = digit
                    digit_entropy[action] = float(-(digit_probs*digit_probs.clamp_min(1e-12).log()).sum())
            correct = [int(a == row['correct_choice']) for a in sampled_actions]
            rewards = [y - lambda_cal*(CONFIDENCES[d]-y)**2
                       for y, d in zip(correct, sampled_digits)]
            reward_tensor = torch.tensor(rewards, dtype=torch.float32, device=args.device)
            advantages = reward_tensor - reward_tensor.mean()
            entry = {
                'step': step, 'id': row['id'], 'sampled_actions': sampled_actions,
                'sampled_digits': sampled_digits, 'correct': correct, 'rewards': rewards,
                'mean_reward': float(reward_tensor.mean()),
                'reward_std': float(reward_tensor.std(unbiased=False)),
                'action_entropy': float(-(action_probs*action_probs.clamp_min(1e-12).log()).sum()),
                'digit_entropy': sum(digit_entropy.values())/len(digit_entropy),
                'skipped': bool(float(advantages.abs().max()) < 1e-8),
            }
            if not entry['skipped']:
                model.train()
                optimizer.zero_grad(set_to_none=True)
                if len(set(sampled_actions)) > 1:
                    action_logp = (logits_for(model, batch, action_prefix(processor), action_ids)
                                   / ACTION_TEMPERATURE).log_softmax(-1)
                    selected_logp = action_logp[torch.tensor(sampled_actions, device=args.device)]
                    action_loss = -(advantages.detach()*selected_logp).mean()
                    action_loss.backward()
                    entry['action_loss'] = float(action_loss.detach())
                    del action_loss, selected_logp, action_logp
                if args.mode == 'rlcd':
                    for action in set(sampled_actions):
                        positions = [i for i, a in enumerate(sampled_actions) if a == action]
                        dlogp = logits_for(model, batch, digit_prefix(processor, action), digit_ids).log_softmax(-1)
                        loss = -sum(advantages[i].detach()*dlogp[sampled_digits[i]] for i in positions)/4
                        loss.backward()
                        del loss, dlogp
                norm = torch.nn.utils.clip_grad_norm_(trainable, 1.0, error_if_nonfinite=True)
                optimizer.step()
                entry['grad_norm'] = float(norm)
            entry['elapsed_seconds'] = time.monotonic() - started
            log.write(json.dumps(entry) + '\n')
            log.flush()
            if step % 25 == 0 or step == len(rows):
                print(json.dumps({k: v for k, v in entry.items()
                                  if k not in ('sampled_actions','sampled_digits','correct','rewards')}), flush=True)
            if step % 100 == 0 or step == len(rows):
                model.model.language_model.save_pretrained(out / 'checkpoints' / f'step_{step:04d}')
    model.model.language_model.save_pretrained(out / 'adapter')
    (out / 'STATUS.txt').write_text(f'COMPLETED {len(rows)}\n')


if __name__ == '__main__':
    main()
