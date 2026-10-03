import json
from pathlib import Path

from core import parse_output

root = Path('/pfs/hyx/videojev-rlcd')
output = root / 'calibration/confidence_lora_v1/evals/selected'
source = {
    'blind': root / 'calibration/pool_predictions.jsonl',
    'holmes': root / 'evaluations/ablation_none_200_holmes92/predictions.jsonl',
}
for split in ('blind', 'holmes'):
    rows = [json.loads(x) for x in (output / f'{split}_predictions.jsonl').read_text().splitlines()]
    reference = {r['id']: r for r in (json.loads(x) for x in source[split].read_text().splitlines())}
    assert len(rows) == (300 if split == 'blind' else 92)
    assert len({r['id'] for r in rows}) == len(rows)
    for r in rows:
        original = reference[r['id']]
        if split == 'blind':
            assert original['split'] == 'blind'
        assert r['predicted_action'] == original['predicted_action']
        assert r['correct'] == original['correct']
        d = parse_output(r['output'], original['num_choices'] if split == 'blind' else len(original['native_probabilities']))
        assert d.valid_format and d.confidence_bin == r['digit']
        assert d.action_index == ord(r['predicted_action']) - 65
        assert r['confidence'] == d.confidence
        assert r['digit'] == max(0, min(9, int(r['q'] * 10)))
    print(split, 'PASS', len(rows))
