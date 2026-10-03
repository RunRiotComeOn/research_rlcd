"""Run the selected JeV action adapter with its validation-fitted confidence map."""
import argparse
import json
import math
from pathlib import Path

import torch
import yaml
from peft import PeftModel

from core import parse_output
from model import load_model, native_action_scores, prepare_video, to_device

ROOT = Path('/pfs/hyx/videojev-rlcd')
ADAPTER = ROOT / 'runs/jev10_action_sft_v1/checkpoints/step_4813'
CALIBRATION = ROOT / 'calibration/jev10_action/selected_step_4813_context/summary.json'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input', required=True, help='Converted VideoJev JSONL with video_path and choices')
    p.add_argument('--output', required=True, help='JSONL destination under /pfs/hyx/videojev-rlcd')
    p.add_argument('--limit', type=int)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--confidence-format', choices=('percent', 'bin'), default='percent',
                   help='Show calibrated confidence as an integer percent or legacy 0-9 bin')
    args = p.parse_args()
    destination = Path(args.output).resolve()
    if not destination.is_relative_to(ROOT.resolve()):
        raise ValueError('Output must stay inside the user project')
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    cfg = yaml.safe_load((ROOT / 'config.yaml').read_text())
    calibration = json.loads(CALIBRATION.read_text())
    if calibration.get('validation_name') != 'selected_step_4813_context_validation':
        raise ValueError('Calibration was not fitted on context-correct action tokens')
    rows = [json.loads(s) for s in Path(args.input).read_text().splitlines() if s.strip()]
    if args.limit is not None:
        rows = rows[:args.limit]
    processor, model = load_model(cfg['model']['path'], args.device)
    model.model.language_model = PeftModel.from_pretrained(model.model.language_model,
                                                           str(ADAPTER), is_trainable=False)
    model.to(args.device).eval()
    with destination.open('w') as f:
        for row in rows:
            batch = to_device(prepare_video(processor, row, cfg['media']['max_frames'],
                                            cfg['media']['max_pixels']), args.device)
            with torch.no_grad():
                scores = native_action_scores(model, processor, batch, len(row['choices']))
            probabilities = scores['probabilities']
            selected = max(range(len(probabilities)), key=probabilities.__getitem__)
            p_selected = max(1e-6, min(1-1e-6, probabilities[selected]))
            z = calibration['slope']*math.log(p_selected/(1-p_selected)) + calibration['intercept']
            q = 1/(1+math.exp(-max(-30, min(30, z))))
            bin_index = max(0, min(9, int(q*10)))
            percent = max(0, min(100, round(100*q)))
            output = (f'Action: {chr(65+selected)}\nConfidence: {percent}%'
                      if args.confidence_format == 'percent' else
                      f'Action: {chr(65+selected)}\nConfidence: {bin_index}')
            if args.confidence_format == 'bin':
                parsed = parse_output(output, len(row['choices']))
                if not parsed.valid_format:
                    raise ValueError(f'Invalid output: {row["id"]}')
                confidence = parsed.confidence
            else:
                confidence = percent/100
            f.write(json.dumps({'id': row['id'], 'output': output,
                                'action': chr(65+selected), 'confidence_bin': bin_index,
                                'confidence_percent': percent, 'confidence': confidence,
                                'continuous_q': q,
                                'confidence_source': 'context_correct_action_logits_plus_platt',
                                'native_probabilities': probabilities,
                                'label_mass': scores['label_mass']}) + '\n')


if __name__ == '__main__':
    main()
