"""Audit frozen-F0 splits/results and paired 2x2 Brier contrasts."""
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score


ROOT = Path('/pfs/hyx/videojev-rlcd')
DATA = ROOT / 'data/f0_confidence_ablation_v1'
HOLMES = ROOT / 'data/jev_full_holmes_v1/holmes_all.jsonl'
FOLD0 = ROOT / 'data/jev_full_holmes_v1/folds/fold0.jsonl'
EVAL = ROOT / 'evaluations/f0_confidence_ablation_v1'
VARIANTS = ('base_low', 'base_high', 'f0_low', 'f0_high')
CONTRASTS = (('base_low', 'base_high', 'higher_lr_from_base'),
             ('f0_low', 'f0_high', 'higher_lr_from_f0'),
             ('base_low', 'f0_low', 'f0_start_at_low_lr'),
             ('base_high', 'f0_high', 'f0_start_at_high_lr'))


def read_jsonl(path):
    with path.open(encoding='utf-8') as stream:
        return [json.loads(line) for line in stream]


def names(rows):
    return {Path(r['video_path']).name for r in rows}


def paired_interval(left, right, seed):
    grouped = defaultdict(list)
    for a, b in zip(left, right):
        assert a['id'] == b['id'] and a['correct'] == b['correct']
        y = a['correct']
        grouped[Path(a['video_path']).name].append(
            (a['confidence_lora_q']-y)**2-(b['confidence_lora_q']-y)**2)
    keys = sorted(grouped)
    sums = np.array([sum(grouped[k]) for k in keys])
    counts = np.array([len(grouped[k]) for k in keys])
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(5000):
        picks = rng.integers(0, len(keys), len(keys))
        draws.append(float(sums[picks].sum()/counts[picks].sum()))
    return {'brier_first_minus_second': float(sums.sum()/counts.sum()),
            'video_bootstrap_95pct_interval': np.quantile(draws, [0.025,0.975]).tolist()}


def main():
    source = {s: read_jsonl(DATA / f'{s}.jsonl') for s in ('train', 'dev', 'test')}
    actions = {s: read_jsonl(DATA / f'{s}_actions.jsonl') for s in source}
    source['holmes'] = read_jsonl(HOLMES)
    actions['holmes'] = read_jsonl(EVAL / 'holmes_actions/predictions.jsonl')
    assert [len(source[s]) for s in ('train','dev','test','holmes')] == [20063,2000,2000,1837]
    fold0 = read_jsonl(FOLD0)
    all_sets = {'fold0': names(fold0), **{s: names(rows) for s, rows in source.items()}}
    assert all(not all_sets[a] & all_sets[b] for i,a in enumerate(all_sets)
               for b in list(all_sets)[i+1:])
    assert all([r['id'] for r in source[s]] == [r['id'] for r in actions[s]]
               for s in source)
    assert all(a['source_answer_fold'] == 'fold0' for s in actions for a in actions[s])
    selection = json.loads((EVAL / 'selection.json').read_text())
    final = json.loads((EVAL / 'final_summary.json').read_text())
    assert len(selection['variants']) == 4
    result = {'rows': {s: len(rows) for s,rows in source.items()},
              'videos': {s: len(names(rows)) for s,rows in source.items()},
              'all_pairwise_video_filename_overlaps': 0,
              'selected_steps': {v: selection['variants'][v]['selected_step'] for v in VARIANTS},
              'contrasts': {}}
    for split in ('test', 'holmes'):
        predictions = {}
        for variant in VARIANTS:
            step = selection['variants'][variant]['selected_step']
            path = EVAL / split / variant / f'examples_{step:05d}'
            expected = len(source[split])
            assert (path / 'STATUS.txt').read_text().strip() == f'COMPLETED {expected}'
            predictions[variant] = read_jsonl(path / 'predictions.jsonl')
            assert [r['id'] for r in predictions[variant]] == [r['id'] for r in source[split]]
            assert all(p['action_index'] == a['action_index'] and p['correct'] == a['correct']
                       for p,a in zip(predictions[variant],actions[split]))
            y = np.asarray([r['correct'] for r in predictions[variant]])
            q = np.asarray([r['confidence_lora_q'] for r in predictions[variant]])
            metric = final['results'][split][variant]['confidence_lora']
            assert abs(float(np.mean((q-y)**2))-metric['brier']) < 1e-10
            assert abs(float(roc_auc_score(y,q))-metric['auc']) < 1e-10
        result['contrasts'][split] = {
            label: paired_interval(predictions[a],predictions[b],20261007)
            for a,b,label in CONTRASTS}
    out = EVAL / 'factorial_contrasts.json'
    out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    main()
