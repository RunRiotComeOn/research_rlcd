"""Evaluate a frozen action adapter plus a separately trained confidence adapter."""
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import torch
import yaml
from peft import PeftModel

from core import CONFIDENCES, calibration_metrics
from decision_policy_v2 import action_tokens, candidate_logits, confidence_tokens
from model import load_model, prepare_video, to_device

ROOT = Path('/pfs/hyx/videojev-rlcd')
ACTION_ADAPTER = ROOT / 'runs/jev10_action_sft_v1/checkpoints/step_4813'
DATA = ROOT / 'data/jev10_v2/validation.jsonl'
REFERENCE = ROOT / 'evaluations/jev10_reward_compare_v2/baseline/predictions.jsonl'
OUT = ROOT / 'evaluations/jev10_frozen_action_meanq_v1'


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', required=True)
    parser.add_argument('--step', type=int, choices=(100, 200, 300), required=True)
    args = parser.parse_args()
    adapter = ROOT / f'runs/jev10_frozen_action_meanq_v1_300/checkpoints/step_{args.step:04d}'
    if (adapter.parent.parent / 'STATUS.txt').read_text().strip() != 'COMPLETED 300':
        raise ValueError('Training incomplete')
    out = OUT / f'step_{args.step:04d}'
    if out.exists():
        raise FileExistsError(out)
    cfg = yaml.safe_load((ROOT / 'config.yaml').read_text(encoding='utf-8'))
    rows = [json.loads(line) for line in DATA.read_text(encoding='utf-8').splitlines()]
    reference = {r['id']: r for r in [json.loads(line) for line in REFERENCE.read_text().splitlines()]}
    if len(rows) != 481 or len(reference) != 481 or {r['id'] for r in rows} != set(reference):
        raise ValueError('Validation rows or reference mismatch')
    out.mkdir(parents=True)
    config = {'action_adapter': str(ACTION_ADAPTER),
              'action_adapter_sha256': sha256(ACTION_ADAPTER / 'adapter_model.safetensors'),
              'confidence_adapter': str(adapter),
              'confidence_adapter_sha256': sha256(adapter / 'adapter_model.safetensors'),
              'data': str(DATA), 'data_sha256': sha256(DATA),
              'reference_sha256': sha256(REFERENCE), 'step': args.step, 'device': args.device}
    (out / 'run_config.json').write_text(json.dumps(config, indent=2) + '\n')
    processor, action_model = load_model(cfg['model']['path'], args.device)
    action_model.requires_grad_(False)
    action_model.model.language_model = PeftModel.from_pretrained(
        action_model.model.language_model, str(ACTION_ADAPTER), is_trainable=False)
    action_model.to(args.device).eval()
    _, conf_model = load_model(cfg['model']['path'], args.device)
    conf_model.requires_grad_(False)
    conf_model.model.language_model = PeftModel.from_pretrained(
        conf_model.model.language_model, str(adapter), is_trainable=False)
    conf_model.to(args.device).eval()
    q_values = torch.tensor(CONFIDENCES, dtype=torch.float32, device=args.device)
    results = []
    with (out / 'predictions.jsonl').open('w', encoding='utf-8') as stream:
        for i, row in enumerate(rows, 1):
            batch = to_device(prepare_video(processor, row, cfg['media']['max_frames'],
                                            cfg['media']['max_pixels']), args.device)
            action_prefix, action_ids = action_tokens(processor, len(row['choices']))
            with torch.no_grad():
                action_probs = candidate_logits(action_model, batch, action_prefix, action_ids).softmax(-1)
                action = int(action_probs.argmax())
                digit_prefix, digit_ids = confidence_tokens(processor, action)
                digit_probs = candidate_logits(conf_model, batch, digit_prefix, digit_ids).softmax(-1)
                q_mean = float((digit_probs * q_values).sum())
                digit = int(digit_probs.argmax())
            if action != reference[row['id']]['action_index']:
                raise ValueError(f'Frozen action changed for {row["id"]}')
            correct = int(action == row['correct_choice'])
            percent = round(100 * q_mean)
            item = {'id': row['id'], 'video_path': row['video_path'],
                    'data_source': row['data_source'], 'gold_action': chr(65 + row['correct_choice']),
                    'predicted_action': chr(65 + action), 'action_index': action, 'correct': correct,
                    'confidence': q_mean, 'confidence_percent': percent,
                    'confidence_rounded': percent / 100, 'confidence_argmax_bin': digit,
                    'digit_probabilities': digit_probs.tolist(),
                    'action_probabilities': action_probs.tolist(),
                    'output': f'Action: {chr(65 + action)}\nConfidence: {percent}%'}
            results.append(item)
            stream.write(json.dumps(item, ensure_ascii=False) + '\n')
            stream.flush()
            if i % 100 == 0 or i == len(rows):
                print(json.dumps({'processed': i, 'total': len(rows)}), flush=True)
    summary = {'rows': len(results), 'step': args.step,
               'correct': sum(r['correct'] for r in results),
               'accuracy': sum(r['correct'] for r in results) / len(results),
               'mean_q': sum(r['confidence'] for r in results) / len(results),
               'continuous': {k: v for k, v in calibration_metrics(results, 'confidence').items()
                              if k in ('brier', 'ece', 'nll')},
               'rounded_percent': {k: v for k, v in calibration_metrics(results, 'confidence_rounded').items()
                                   if k in ('brier', 'ece', 'nll')},
               'argmax_bins': dict(sorted(Counter(r['confidence_argmax_bin'] for r in results).items()))}
    (out / 'summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False) + '\n')
    (out / 'STATUS.txt').write_text('COMPLETED 481\n')
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
