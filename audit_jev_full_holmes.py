"""Independent row and metric audit of the completed full-JeV experiment."""
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score


ROOT = Path('/pfs/hyx/videojev-rlcd')
DATA = ROOT / 'data/jev_full_holmes_v1'
OOF = ROOT / 'calibration/jevfull_oof_v1'
EVAL = ROOT / 'evaluations/jevfull_holmes_v1'


def rows(path):
    with path.open(encoding='utf-8') as stream:
        return [json.loads(line) for line in stream]


def names(items):
    return {Path(row['video_path']).name for row in items}


def main():
    jev = rows(DATA / 'jev_all.jsonl')
    holmes = rows(DATA / 'holmes_all.jsonl')
    folds = [rows(DATA / 'folds' / f'fold{i}.jsonl') for i in (0, 1)]
    oof = [rows(OOF / f'fold{i}' / 'predictions.jsonl') for i in (0, 1)]
    actions = rows(EVAL / 'actions/predictions.jsonl')
    final = rows(EVAL / 'confidence/predictions.jsonl')
    assert [len(jev), len(holmes), *map(len, folds), *map(len, oof),
            len(actions), len(final)] == [48126, 1837, 24063, 24063,
                                          24063, 24063, 1837, 1837]
    assert {r['id'] for r in jev} == {r['id'] for fold in folds for r in fold}
    assert len({r['id'] for r in jev}) == 48126
    assert not names(jev) & names(holmes)
    assert not names(folds[0]) & names(folds[1])
    for i in (0, 1):
        assert [r['id'] for r in folds[i]] == [r['id'] for r in oof[i]]
        assert all(r['source_answer_fold'] == f'fold{1-i}' for r in oof[i])
    assert [r['id'] for r in holmes] == [r['id'] for r in actions] == [r['id'] for r in final]
    assert all(a['correct'] == f['correct'] and
               a['action_index'] == f['action_index']
               for a, f in zip(actions, final))
    y = np.array([r['correct'] for r in final])
    measures = {}
    for name, field in [('selected_action_p', 'chosen_action_p'),
                        ('platt', 'platt_q'),
                        ('confidence_lora', 'confidence_lora_q')]:
        q = np.array([r[field] for r in final])
        measures[name] = {'brier': float(np.mean((q-y)**2)),
                          'mean_q': float(np.mean(q)),
                          'auc': float(roc_auc_score(y, q))}
    summary = json.loads((EVAL / 'confidence/summary.json').read_text())
    for name, result in measures.items():
        for metric, value in result.items():
            assert abs(value-summary['methods'][name][metric]) < 1e-10
    accuracy = float(y.mean())
    assert abs(accuracy-summary['accuracy']) < 1e-10
    assert abs(accuracy*(1-accuracy)-summary['constant_oracle_brier']) < 1e-10
    delta = measures['confidence_lora']['brier']-measures['platt']['brier']
    assert abs(delta-summary['lora_minus_platt']['brier_delta']) < 1e-10
    print(json.dumps({'data_rows': len(jev), 'fold_rows': list(map(len, folds)),
                      'oof_rows': list(map(len, oof)), 'holmes_rows': len(holmes),
                      'holmes_videos': len(names(holmes)),
                      'jev_holmes_video_filename_overlap': len(names(jev)&names(holmes)),
                      'cross_fold_video_filename_overlap': len(names(folds[0])&names(folds[1])),
                      'accuracy': accuracy, 'constant_oracle_brier': accuracy*(1-accuracy),
                      'methods': measures, 'lora_minus_platt_brier': delta,
                      'video_bootstrap_95pct_interval': summary['lora_minus_platt']['video_bootstrap_95pct_interval']},
                     indent=2))


if __name__ == '__main__':
    main()
