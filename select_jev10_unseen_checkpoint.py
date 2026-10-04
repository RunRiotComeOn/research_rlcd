"""Choose confidence LoRA weights solely from the development set."""
import json
from pathlib import Path

ROOT = Path('/pfs/hyx/videojev-rlcd')
EVAL = ROOT / 'evaluations/jev10_unseen_confidence_v1'
OUT = ROOT / 'calibration/jev10_unseen_v1/selected_checkpoint.json'


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    records = []
    for examples in (1000, 2000, 3000, 4000):
        base = EVAL / f'examples_{examples:04d}' / 'development'
        if (base / 'STATUS.txt').read_text().strip() != 'COMPLETED 500':
            raise ValueError(f'Incomplete development evaluation: {examples}')
        summary = json.loads((base / 'summary.json').read_text())
        if summary['rows'] != 500 or summary['training_examples'] != examples:
            raise ValueError(f'Unexpected summary: {examples}')
        records.append({'examples': examples, 'development_brier': summary['brier'],
                        'development_mean_q': summary['mean_q']})
    chosen = min(records, key=lambda r: (r['development_brier'], r['examples']))
    baselines = json.loads((ROOT / 'calibration/jev10_unseen_v1/models/fit_development_summary.json').read_text())
    result = {'examples': chosen['examples'], 'selection_rule': 'minimum development Brier, earliest tie',
              'checkpoints': records,
              'development_platt_brier': baselines['development']['methods']['platt']['brier'],
              'development_constant_oracle_brier': baselines['development']['constant_oracle_brier']}
    OUT.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
