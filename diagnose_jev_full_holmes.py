"""Diagnose online JeV confidence and Holmes calibration without model reruns."""
import json
import math
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit


ROOT = Path('/pfs/hyx/videojev-rlcd')
DATA = ROOT / 'data/jev_full_holmes_v1'
OOF = ROOT / 'calibration/jevfull_oof_v1'
CONF = ROOT / 'runs/jevfull_confidence_oof_v1'
HOLMES = ROOT / 'evaluations/jevfull_holmes_v1/confidence'


def read_jsonl(path):
    with path.open(encoding='utf-8') as stream:
        return [json.loads(line) for line in stream]


def platt_fit(p, y):
    p = np.clip(p, 1e-6, 1-1e-6)
    z = np.log(p/(1-p))
    def value(theta):
        t = theta[0]*z+theta[1]
        loss = np.mean(np.logaddexp(0, t)-y*t)+1e-4*theta[0]**2
        grad = np.array([np.mean((expit(t)-y)*z)+2e-4*theta[0],
                         np.mean(expit(t)-y)])
        return loss, grad
    result = minimize(value, [1.0, 0.0], jac=True, method='L-BFGS-B',
                      options={'maxiter': 1000, 'ftol': 1e-12})
    if not result.success:
        raise RuntimeError(result.message)
    return result.x


def brier(q, y):
    return float(np.mean((q-y)**2))


def reliability(items, field):
    bins = []
    n = len(items)
    for i in range(10):
        selected = [r for r in items if min(9, int(r[field]*10)) == i]
        bins.append({'range': [i/10, (i+1)/10], 'count': len(selected),
                     'mean_q': sum(r[field] for r in selected)/len(selected) if selected else None,
                     'accuracy': sum(r['correct'] for r in selected)/len(selected) if selected else None})
    ece = sum(row['count']/n*abs(row['mean_q']-row['accuracy'])
              for row in bins if row['count'])
    return {'ece_10_equal_width': ece, 'bins': bins}


def group_summary(rows):
    n = len(rows)
    return {'n': n, 'accuracy': sum(r['correct'] for r in rows)/n,
            'mean_platt_q': sum(r['platt_q'] for r in rows)/n,
            'mean_lora_q': sum(r['confidence_lora_q'] for r in rows)/n,
            'platt_brier': sum((r['platt_q']-r['correct'])**2 for r in rows)/n,
            'lora_brier': sum((r['confidence_lora_q']-r['correct'])**2 for r in rows)/n}


def main():
    rows = read_jsonl(DATA / 'jev_all.jsonl')
    labels = {r['id']: r for fold in ('fold0', 'fold1')
              for r in read_jsonl(OOF / fold / 'predictions.jsonl')}
    assert len(rows) == len(labels) == 48126
    order = list(range(len(rows)))
    random.Random(20261005).shuffle(order)
    ordered = [labels[rows[i]['id']] for i in order]
    metrics = read_jsonl(CONF / 'metrics.jsonl')
    blocks = [r for r in metrics if 43100 < r['examples'] <= 48100
              and r['examples'] % 100 == 0]
    assert len(blocks) == 50 and blocks[0]['examples'] == 43200
    online_brier = sum(r['mean_online_brier_last_100'] for r in blocks)/50
    prefix = ordered[:43100]
    window = ordered[43100:48100]
    assert len(window) == 5000
    def arrays(items):
        return (np.array([r['action_probabilities'][r['action_index']] for r in items]),
                np.array([r['correct'] for r in items]))
    p_train, y_train = arrays(prefix)
    p_test, y_test = arrays(window)
    slope, intercept = platt_fit(p_train, y_train)
    full_fit = json.loads((OOF / 'platt/parameters.json').read_text())
    logit_test = np.log(np.clip(p_test,1e-6,1-1e-6)/
                        np.clip(1-p_test,1e-6,1-1e-6))
    prefix_q = expit(slope*logit_test+intercept)
    full_q = expit(full_fit['slope']*logit_test+full_fit['intercept'])
    prefix_videos = {Path(r['video_path']).name for r in prefix}
    window_videos = {Path(r['video_path']).name for r in window}
    online = {'window_training_positions_1based': [43101, 48100],
              'examples': 5000, 'lora_online_brier': online_brier,
              'platt_prefix_fit_brier': brier(prefix_q, y_test),
              'platt_full_fit_brier_optimistic': brier(full_q, y_test),
              'raw_action_probability_brier': brier(p_test, y_test),
              'window_accuracy': float(y_test.mean()),
              'platt_prefix_slope': float(slope),
              'platt_prefix_intercept': float(intercept),
              'window_video_filenames': len(window_videos),
              'window_videos_seen_in_confidence_prefix': len(window_videos & prefix_videos)}
    holmes_data = read_jsonl(DATA / 'holmes_all.jsonl')
    final = read_jsonl(HOLMES / 'predictions.jsonl')
    assert [r['id'] for r in holmes_data] == [r['id'] for r in final]
    merged = [{**prediction,
               'choice_count': len(source['choices']),
               'question_word': (re.match(r'\w+', source['question'].strip().lower()) or ['other'])[0]}
              for source, prediction in zip(holmes_data, final)]
    by_options = defaultdict(list)
    by_question = defaultdict(list)
    by_video = defaultdict(list)
    for row in merged:
        by_options[str(row['choice_count'])].append(row)
        by_question[row['question_word']].append(row)
        by_video[Path(row['video_path']).name].append(row)
    video_deltas = [group_summary(v)['lora_brier']-group_summary(v)['platt_brier']
                    for v in by_video.values()]
    reliability_result = {name: reliability(merged, field) for name, field in
                          [('platt', 'platt_q'), ('confidence_lora', 'confidence_lora_q')]}
    diagnostics = {'jev_online_last_5000': online,
                   'holmes_reliability': reliability_result,
                   'holmes_by_choice_count': {k: group_summary(v) for k,v in sorted(by_options.items())},
                   'holmes_by_question_first_word': {k: group_summary(v) for k,v in
                                                     sorted(by_question.items(), key=lambda kv: -len(kv[1]))
                                                     if len(v) >= 30},
                   'holmes_videos': {'count': len(by_video),
                                    'lora_worse_than_platt_count': sum(x > 0 for x in video_deltas),
                                    'lora_better_than_platt_count': sum(x < 0 for x in video_deltas),
                                    'question_count_min': min(map(len, by_video.values())),
                                    'question_count_max': max(map(len, by_video.values()))}}
    out = HOLMES / 'diagnostics.json'
    out.write_text(json.dumps(diagnostics, indent=2) + '\n')
    print(json.dumps(diagnostics, indent=2))


if __name__ == '__main__':
    main()
