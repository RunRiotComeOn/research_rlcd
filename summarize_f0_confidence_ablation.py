"""Aggregate selected F0 confidence ablations and video bootstrap intervals."""
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


ROOT = Path('/pfs/hyx/videojev-rlcd')
EVAL = ROOT / 'evaluations/f0_confidence_ablation_v1'


def read_jsonl(path):
    with path.open() as stream:
        return [json.loads(line) for line in stream]


def bootstrap(rows, seed):
    groups = defaultdict(list)
    for row in rows:
        groups[Path(row['video_path']).name].append(
            (row['confidence_lora_q']-row['correct'])**2-
            (row['platt_q']-row['correct'])**2)
    keys = sorted(groups)
    sums = np.array([sum(groups[k]) for k in keys])
    counts = np.array([len(groups[k]) for k in keys])
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(5000):
        picks = rng.integers(0, len(keys), len(keys))
        draws.append(float(sums[picks].sum()/counts[picks].sum()))
    return np.quantile(draws, [0.025, 0.975]).tolist()


def main():
    selection = json.loads((EVAL / 'selection.json').read_text())
    output = {'answer_model': 'jevfull_action_fold0_v1/checkpoints/examples_24063',
              'data_rows': {'confidence_train': 20063, 'development': 2000,
                            'jev_sealed_test': 2000, 'holmes': 1837},
              'selection_criterion': selection['criterion'],
              'results': {}}
    for split, expected in [('test', 2000), ('holmes', 1837)]:
        output['results'][split] = {}
        reference = None
        for variant, chosen in selection['variants'].items():
            step = chosen['selected_step']
            path = EVAL / split / variant / f'examples_{step:05d}'
            if (path / 'STATUS.txt').read_text().strip() != f'COMPLETED {expected}':
                raise ValueError(f'Incomplete {split} {variant}')
            summary = json.loads((path / 'summary.json').read_text())
            rows = read_jsonl(path / 'predictions.jsonl')
            assert len(rows) == expected
            common = [(r['id'], r['correct'], r['action_index'],
                       r['selected_action_p'], r['platt_q']) for r in rows]
            if reference is None:
                reference = common
            elif common != reference:
                raise ValueError(f'Frozen F0 actions or Platt differ across {split} runs')
            lo = summary['methods']['confidence_lora']
            pl = summary['methods']['platt']
            output['results'][split][variant] = {
                'selected_step': step, 'accuracy': summary['accuracy'],
                'constant_oracle_brier': summary['constant_oracle_brier'],
                'videos': summary['videos'], 'confidence_lora': lo,
                'platt': pl, 'selected_action_p': summary['methods']['selected_action_p'],
                'lora_minus_platt_brier': lo['brier']-pl['brier'],
                'video_bootstrap_95pct_interval': bootstrap(rows, 20261007)}
    path = EVAL / 'final_summary.json'
    path.write_text(json.dumps(output, indent=2) + '\n')
    print(json.dumps(output, indent=2))


if __name__ == '__main__':
    main()
