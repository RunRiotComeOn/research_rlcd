"""Action REINFORCE with exact expected Brier on the greedy action."""
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
ACTION_TEMPERATURE = 2.0
BETA = 0.2


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def expected_brier(probs, q_values, correct):
    if probs.ndim != 1 or probs.shape != q_values.shape:
        raise ValueError('Expected matching one-dimensional probability and q vectors')
    return (probs * (q_values - correct).square()).sum()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', required=True)
    parser.add_argument('--limit', type=int, default=300)
    parser.add_argument('--run-name', required=True)
    parser.add_argument('--beta', type=float, default=BETA)
    args = parser.parse_args()
    if not 1 <= args.limit <= 300:
        raise ValueError('limit must be in 1..300')
    if args.beta != BETA:
        raise ValueError('Prespecified beta is 0.2')
    if Path(args.run_name).name != args.run_name:
        raise ValueError('run-name must be a single directory name')
    out = ROOT / 'runs' / args.run_name
    if out.exists():
        raise FileExistsError(out)
    cfg = yaml.safe_load((ROOT / 'config.yaml').read_text(encoding='utf-8'))
    rows = [json.loads(line) for line in DATA.read_text(encoding='utf-8').splitlines()]
    random.Random(SEED).shuffle(rows)
    rows = rows[:args.limit]
    out.mkdir(parents=True)
    config = {
        'objective': 'action group-centered correctness REINFORCE plus exact expected Brier on greedy action',
        'beta': args.beta, 'seed': SEED, 'limit': args.limit, 'num_rollouts': 4,
        'action_sampling_temperature': ACTION_TEMPERATURE,
        'learning_rate': 2e-5, 'confidence_bins': list(CONFIDENCES),
        'source_adapter': str(START),
        'source_adapter_weights_sha256': sha256(START / 'adapter_model.safetensors'),
        'data': str(DATA), 'data_sha256': sha256(DATA), 'device': args.device,
        'shared_lora': True, 'greedy_action_for_confidence': True,
    }
    (out / 'run_config.json').write_text(json.dumps(config, indent=2) + '\n')
    torch.manual_seed(SEED)
    processor, model = load_model(cfg['model']['path'], args.device)
    model.requires_grad_(False)
    model.model.language_model = PeftModel.from_pretrained(
        model.model.language_model, str(START), is_trainable=True)
    model.to(args.device)
    trainable = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    if not trainable or any('lora_' not in name for name, _ in trainable):
        raise RuntimeError('Only language LoRA weights may train')
    optimizer = torch.optim.AdamW((p for _, p in trainable), lr=config['learning_rate'], weight_decay=0)
    q_values = torch.tensor(CONFIDENCES, dtype=torch.float32, device=args.device)
    started = time.monotonic()
    with (out / 'metrics.jsonl').open('w') as stream:
        for step, row in enumerate(rows, 1):
            batch = to_device(prepare_video(processor, row, cfg['media']['max_frames'],
                                            cfg['media']['max_pixels']), args.device)
            if batch['input_ids'].shape[1] > cfg['media']['max_prompt_tokens']:
                raise ValueError(f'Prompt too long: {row["id"]}')
            action_prefix, action_ids = action_tokens(processor, len(row['choices']))
            model.eval()  # Keep the confidence context and dropout behavior aligned with evaluation.
            with torch.no_grad():
                action_logits = candidate_logits(model, batch, action_prefix, action_ids)
                greedy_action = int(action_logits.argmax())
                action_probs = (action_logits / ACTION_TEMPERATURE).softmax(-1)
                actions = torch.multinomial(action_probs, 4, replacement=True).tolist()
            action_correct = [int(a == row['correct_choice']) for a in actions]
            greedy_correct = int(greedy_action == row['correct_choice'])
            advantages = torch.tensor(action_correct, dtype=torch.float32, device=args.device)
            advantages -= advantages.mean()
            optimizer.zero_grad(set_to_none=True)
            action_loss_value = 0.0
            if float(advantages.abs().max()) > 0:
                action_log_probs = (candidate_logits(model, batch, action_prefix, action_ids)
                                    / ACTION_TEMPERATURE).log_softmax(-1)
                action_loss = -(advantages.detach() * action_log_probs[
                    torch.tensor(actions, device=args.device)]).mean()
                action_loss.backward()
                action_loss_value = float(action_loss.detach())
                del action_loss, action_log_probs
            digit_prefix, digit_ids = confidence_tokens(processor, greedy_action)
            digit_probs = candidate_logits(model, batch, digit_prefix, digit_ids).softmax(-1)
            conf_brier = expected_brier(digit_probs, q_values, greedy_correct)
            (args.beta * conf_brier).backward()
            with torch.no_grad():
                mean_q = float((digit_probs * q_values).sum())
                conf_entropy = float(-(digit_probs * digit_probs.clamp_min(1e-12).log()).sum())
                top_bin = int(digit_probs.argmax())
                conf_loss_value = float(conf_brier)
            norm = torch.nn.utils.clip_grad_norm_([p for _, p in trainable], 1.0,
                                                  error_if_nonfinite=True)
            optimizer.step()
            entry = {
                'step': step, 'id': row['id'], 'sampled_actions': actions,
                'sampled_correct': action_correct, 'greedy_action': greedy_action,
                'greedy_correct': greedy_correct, 'action_advantage_nonzero': bool(float(advantages.abs().max()) > 0),
                'action_loss': action_loss_value, 'confidence_expected_brier': conf_loss_value,
                'confidence_weighted_loss': args.beta * conf_loss_value,
                'confidence_mean_q': mean_q, 'confidence_argmax_bin': top_bin,
                'confidence_entropy': conf_entropy,
                'action_entropy': float(-(action_probs * action_probs.clamp_min(1e-12).log()).sum()),
                'grad_norm': float(norm), 'elapsed_seconds': time.monotonic() - started,
            }
            stream.write(json.dumps(entry) + '\n')
            stream.flush()
            if step % 25 == 0 or step == len(rows):
                print(json.dumps({k: v for k, v in entry.items()
                                  if k not in ('sampled_actions', 'sampled_correct')}), flush=True)
            if step % 100 == 0 or step == len(rows):
                model.model.language_model.save_pretrained(out / 'checkpoints' / f'step_{step:04d}')
    model.model.language_model.save_pretrained(out / 'adapter')
    (out / 'STATUS.txt').write_text(f'COMPLETED {len(rows)}\n')
    print('COMPLETED expected_brier', flush=True)


if __name__ == '__main__':
    main()
