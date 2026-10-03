"""Run one frozen model on the prespecified fresh JeV blind set."""
import argparse
import hashlib
import json
import math
from pathlib import Path

import torch
import yaml
from peft import PeftModel

from model import load_model, native_action_scores, prepare_video, to_device

ROOT = Path('/pfs/hyx/videojev-rlcd')
DATA = ROOT / 'data/jev10_blind_v1'
OUT = ROOT / 'evaluations/jev10_blind_v1'
CONFIGS = {
    'existing': ('runs/jev10_action_sft_v1/checkpoints/step_4813',
                 'calibration/jev10_action/selected_step_4813_context/summary.json'),
    'ce_brier': ('runs/jev10_decision_context_full_ce_brier_v1/checkpoints/step_2000',
                 'calibration/jev10_action/full_ce_brier_step_2000/summary.json'),
}


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def calibrated(p, slope, intercept):
    p = max(1e-6, min(1-1e-6, p))
    z = slope*math.log(p/(1-p))+intercept
    return 1/(1+math.exp(-max(-30, min(30, z))))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=CONFIGS, required=True)
    parser.add_argument('--device', required=True)
    args = parser.parse_args()
    out = OUT / args.model
    if out.exists():
        raise FileExistsError(out)
    manifest = json.loads((DATA / 'manifest.json').read_text(encoding='utf-8'))
    data_file = DATA / 'blind.jsonl'
    if sha256(data_file) != manifest['output_sha256'] or manifest['rows'] != 2000:
        raise ValueError('Blind dataset checksum or row count changed')
    rows = [json.loads(s) for s in data_file.read_text(encoding='utf-8').splitlines()]
    if len(rows) != 2000 or len({r['id'] for r in rows}) != 2000:
        raise ValueError('Blind IDs or row count invalid')
    adapter_rel, calibration_rel = CONFIGS[args.model]
    adapter, calibration_file = ROOT / adapter_rel, ROOT / calibration_rel
    calibration = json.loads(calibration_file.read_text(encoding='utf-8'))
    if calibration['fit_rows'] != 481:
        raise ValueError('Unexpected calibration fit set')
    cfg = yaml.safe_load((ROOT / 'config.yaml').read_text(encoding='utf-8'))
    out.mkdir(parents=True)
    run = {'model': args.model, 'adapter': str(adapter),
           'adapter_config_sha256': sha256(adapter/'adapter_config.json'),
           'adapter_weights_sha256': sha256(adapter/'adapter_model.safetensors'),
           'calibration': str(calibration_file),
           'calibration_sha256': sha256(calibration_file),
           'blind_data_sha256': sha256(data_file),
           'base_model': cfg['model']['path'], 'device': args.device,
           'max_frames': cfg['media']['max_frames'],
           'max_pixels': cfg['media']['max_pixels']}
    (out/'run_config.json').write_text(json.dumps(run, indent=2)+'\n')
    processor, model = load_model(cfg['model']['path'], args.device)
    model.model.language_model = PeftModel.from_pretrained(
        model.model.language_model, str(adapter), is_trainable=False)
    model.to(args.device)
    model.eval()
    with (out/'predictions.jsonl').open('w', encoding='utf-8') as stream:
        for i, row in enumerate(rows, 1):
            batch = to_device(prepare_video(processor, row, cfg['media']['max_frames'],
                                            cfg['media']['max_pixels']), args.device)
            with torch.no_grad():
                scores = native_action_scores(model, processor, batch, len(row['choices']))
            probs = scores['probabilities']
            predicted = max(range(len(probs)), key=probs.__getitem__)
            p = probs[predicted]
            q = calibrated(p, calibration['slope'], calibration['intercept'])
            result = {'id': row['id'], 'video_path': row['video_path'],
                      'data_source': row['data_source'],
                      'gold_action': chr(65+row['correct_choice']),
                      'predicted_action': chr(65+predicted),
                      'correct': int(predicted == row['correct_choice']),
                      'native_probabilities': probs, 'label_mass': scores['label_mass'],
                      'selected_p': p, 'platt_q': q,
                      'percent': round(100*q)}
            stream.write(json.dumps(result, ensure_ascii=False)+'\n')
            stream.flush()
            if i % 100 == 0 or i == len(rows):
                print(json.dumps({'processed': i, 'total': len(rows)}), flush=True)
    (out/'STATUS.txt').write_text('COMPLETED 2000\n')
    print(f'COMPLETED {args.model}', flush=True)


if __name__ == '__main__':
    main()
