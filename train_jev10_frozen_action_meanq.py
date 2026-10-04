"""Train an independent confidence LoRA against Brier of its mean probability."""
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


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', required=True)
    parser.add_argument('--limit', type=int, default=300)
    parser.add_argument('--run-name', required=True)
    args = parser.parse_args()
    if not 1 <= args.limit <= 300:
        raise ValueError('limit must be in 1..300')
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
    run_cfg = {
        'objective': 'frozen greedy action; separate confidence LoRA; Brier of softmax mean q',
        'seed': SEED, 'limit': args.limit, 'learning_rate': 2e-5,
        'action_adapter': str(START), 'confidence_initial_adapter': str(START),
        'source_adapter_weights_sha256': sha256(START / 'adapter_model.safetensors'),
        'data': str(DATA), 'data_sha256': sha256(DATA), 'device': args.device,
        'confidence_bins': list(CONFIDENCES), 'action_policy_frozen': True,
        'separate_confidence_model': True, 'greedy_action_for_confidence': True,
    }
    (out / 'run_config.json').write_text(json.dumps(run_cfg, indent=2) + '\n')
    torch.manual_seed(SEED)
    processor, action_model = load_model(cfg['model']['path'], args.device)
    action_model.requires_grad_(False)
    action_model.model.language_model = PeftModel.from_pretrained(
        action_model.model.language_model, str(START), is_trainable=False)
    action_model.to(args.device).eval()
    _, conf_model = load_model(cfg['model']['path'], args.device)
    conf_model.requires_grad_(False)
    conf_model.model.language_model = PeftModel.from_pretrained(
        conf_model.model.language_model, str(START), is_trainable=True)
    conf_model.to(args.device).eval()
    trainable = [(name, param) for name, param in conf_model.named_parameters() if param.requires_grad]
    if not trainable or any('lora_' not in name for name, _ in trainable):
        raise RuntimeError('Only confidence-model language LoRA weights may train')
    if any(p.requires_grad for p in action_model.parameters()):
        raise RuntimeError('Action model was not frozen')
    optimizer = torch.optim.AdamW((p for _, p in trainable), lr=run_cfg['learning_rate'], weight_decay=0)
    q_values = torch.tensor(CONFIDENCES, dtype=torch.float32, device=args.device)
    started = time.monotonic()
    with (out / 'metrics.jsonl').open('w') as stream:
        for step, row in enumerate(rows, 1):
            batch = to_device(prepare_video(processor, row, cfg['media']['max_frames'],
                                            cfg['media']['max_pixels']), args.device)
            if batch['input_ids'].shape[1] > cfg['media']['max_prompt_tokens']:
                raise ValueError(f'Prompt too long: {row["id"]}')
            action_prefix, action_ids = action_tokens(processor, len(row['choices']))
            with torch.no_grad():
                action_probs = candidate_logits(action_model, batch, action_prefix, action_ids).softmax(-1)
                action = int(action_probs.argmax())
            correct = int(action == row['correct_choice'])
            digit_prefix, digit_ids = confidence_tokens(processor, action)
            optimizer.zero_grad(set_to_none=True)
            digit_probs = candidate_logits(conf_model, batch, digit_prefix, digit_ids).softmax(-1)
            mean_q = (digit_probs * q_values).sum()
            loss = (mean_q - correct).square()
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                [p for _, p in trainable], 1.0, error_if_nonfinite=True)
            optimizer.step()
            with torch.no_grad():
                entry = {
                    'step': step, 'id': row['id'], 'greedy_action': action,
                    'correct': correct, 'mean_q': float(mean_q),
                    'brier': float(loss), 'argmax_bin': int(digit_probs.argmax()),
                    'digit_entropy': float(-(digit_probs * digit_probs.clamp_min(1e-12).log()).sum()),
                    'action_entropy': float(-(action_probs * action_probs.clamp_min(1e-12).log()).sum()),
                    'grad_norm': float(grad_norm), 'elapsed_seconds': time.monotonic() - started,
                }
            stream.write(json.dumps(entry) + '\n')
            stream.flush()
            if step % 25 == 0 or step == len(rows):
                print(json.dumps(entry), flush=True)
            if step % 100 == 0 or step == len(rows):
                conf_model.model.language_model.save_pretrained(out / 'checkpoints' / f'step_{step:04d}')
    conf_model.model.language_model.save_pretrained(out / 'adapter')
    (out / 'STATUS.txt').write_text(f'COMPLETED {len(rows)}\n')
    print('COMPLETED frozen_action_meanq', flush=True)


if __name__ == '__main__':
    main()
