"""Select each variant's confidence checkpoint by development Brier only."""
import json
from pathlib import Path


ROOT = Path('/pfs/hyx/videojev-rlcd')
EVAL = ROOT / 'evaluations/f0_confidence_ablation_v1'
STEPS = (4000, 8000, 12000, 16000, 20000, 20063)
VARIANTS = ('base_low', 'base_high', 'f0_low', 'f0_high')


def main():
    path = EVAL / 'selection.json'
    if path.exists():
        raise FileExistsError(path)
    result = {'criterion': 'minimum development Brier of confidence LoRA; earlier step breaks ties',
              'dev_rows': 2000, 'variants': {}}
    reference = None
    for variant in VARIANTS:
        trials = []
        for step in STEPS:
            checkpoint = EVAL / 'dev' / variant / f'examples_{step:05d}'
            if (checkpoint / 'STATUS.txt').read_text().strip() != 'COMPLETED 2000':
                raise ValueError(f'Incomplete dev result {variant} {step}')
            summary = json.loads((checkpoint / 'summary.json').read_text())
            common = (summary['accuracy'], summary['methods']['platt'])
            if reference is None:
                reference = common
            elif common != reference:
                raise ValueError('Development labels or Platt baseline differ across runs')
            trials.append({'step': step,
                           'brier': summary['methods']['confidence_lora']['brier'],
                           'auc': summary['methods']['confidence_lora']['auc'],
                           'ece': summary['methods']['confidence_lora']['ece_10_equal_width']})
        best = min(trials, key=lambda r: (r['brier'], r['step']))
        result['variants'][variant] = {'selected_step': best['step'],
                                       'selected_dev': best, 'trials': trials}
    result['dev_platt'] = reference[1]
    path.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
