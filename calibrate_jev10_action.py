"""Fit a two-parameter Platt calibrator on JeV validation and apply to test."""
import argparse
import json
import math
from collections import Counter
from pathlib import Path

import torch

from core import calibration_metrics, parse_output

ROOT = Path('/pfs/hyx/videojev-rlcd')


def records(path):
    return [json.loads(s) for s in path.read_text().splitlines() if s.strip()]


def feature(p):
    p = max(1e-6, min(1-1e-6, p))
    return math.log(p/(1-p))


def digit(q):
    return max(0, min(9, int(q*10)))


def scores(rows, key):
    data = [{'correct': r['correct'], 'confidence': r[key]} for r in rows]
    metric = calibration_metrics(data)
    pos = [r[key] for r in rows if r['correct']]
    neg = [r[key] for r in rows if not r['correct']]
    auc = sum((a > b) + 0.5*(a == b) for a in pos for b in neg)/(len(pos)*len(neg))
    return {'brier': metric['brier'], 'ece': metric['ece'], 'nll': metric['nll'],
            'rank_auc': auc, 'bins': dict(sorted(Counter(digit(r[key]) for r in rows).items()))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--validation-name', required=True)
    parser.add_argument('--test-name', required=True)
    parser.add_argument('--name', required=True)
    args = parser.parse_args()
    parent = ROOT / 'evaluations/jev10_action'
    out = ROOT / 'calibration/jev10_action' / args.name
    if out.exists():
        raise FileExistsError(f'Refusing to overwrite {out}')
    validation = records(parent / args.validation_name / 'predictions.jsonl')
    test = records(parent / args.test_name / 'predictions.jsonl')
    if len(validation) != 481 or len(test) != 481:
        raise ValueError('Expected 481 validation and 481 test predictions')
    if {r['id'] for r in validation} & {r['id'] for r in test}:
        raise ValueError('Validation/test IDs overlap')
    out.mkdir(parents=True)
    x = torch.tensor([feature(r['selected_p']) for r in validation])
    y = torch.tensor([r['correct'] for r in validation], dtype=torch.float32)
    theta = torch.nn.Parameter(torch.tensor([1.0, 0.0]))
    optimizer = torch.optim.LBFGS([theta], lr=1.0, max_iter=100,
                                  line_search_fn='strong_wolfe')
    def closure():
        optimizer.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(theta[0]*x+theta[1], y)
        loss = loss + 1e-4*theta[0].square()
        loss.backward()
        return loss
    optimizer.step(closure)
    slope, intercept = [float(v) for v in theta.detach()]
    result = {'validation_name': args.validation_name, 'test_name': args.test_name,
              'fit_rows': 481, 'slope': slope, 'intercept': intercept}
    for split, group in [('validation', validation), ('test', test)]:
        enriched = []
        for r in group:
            q = 1/(1+math.exp(-max(-30, min(30, slope*feature(r['selected_p'])+intercept))))
            bin_index = digit(q)
            output = f"Action: {r['predicted_action']}\nConfidence: {bin_index}"
            parsed = parse_output(output, len(r['native_probabilities']))
            if not parsed.valid_format or parsed.action_index != ord(r['predicted_action'])-65:
                raise ValueError(f'Invalid output for {r["id"]}')
            enriched.append({**r, 'platt_q': q, 'platt_bin': bin_index,
                             'platt_discrete': parsed.confidence, 'output': output})
        with (out / f'{split}_predictions.jsonl').open('w') as f:
            for row in enriched:
                f.write(json.dumps(row) + '\n')
        result[split] = {'rows': len(group), 'accuracy': sum(r['correct'] for r in group)/len(group),
                         'native': scores(enriched, 'selected_p'),
                         'platt': scores(enriched, 'platt_discrete')}
    (out / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
    (out / 'STATUS.txt').write_text('COMPLETED\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
