"""One-shot blind comparison of the fixed step-300 reward models."""
import hashlib
import json
from pathlib import Path

from analyze_jev10_reward_compare import paired, records

ROOT = Path('/pfs/hyx/videojev-rlcd')
DATA = ROOT/'data/jev10_reward_blind_v1'
EVAL = ROOT/'evaluations/jev10_reward_blind_v1'
OUT = ROOT/'comparisons/jev10_reward_blind_v1'
NAMES = ('baseline','old_brier_step_0300','proposed_step_0300')


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    manifest = json.loads((DATA/'manifest.json').read_text())
    blind = DATA/'blind.jsonl'
    if manifest['rows'] != 1000 or sha256(blind) != manifest['output_sha256']:
        raise ValueError('Blind data invalid')
    source = {r['id']:r for r in records(blind)}
    if len(source) != 1000:
        raise ValueError('Blind IDs not unique')
    groups, summaries, hashes = {}, {}, {}
    for name in NAMES:
        folder = EVAL/name
        if (folder/'STATUS.txt').read_text().strip() != 'COMPLETED 1000':
            raise ValueError(f'{name} incomplete')
        cfg = json.loads((folder/'run_config.json').read_text())
        if cfg['dataset'] != 'blind' or cfg['data_sha256'] != manifest['output_sha256']:
            raise ValueError(f'{name} used different data')
        if sha256(Path(cfg['adapter'])/'adapter_model.safetensors') != cfg['adapter_weights_sha256']:
            raise ValueError(f'{name} weights changed')
        file = folder/'predictions.jsonl'
        rows = records(file)
        group = {r['id']:r for r in rows}
        if len(rows) != 1000 or set(group) != set(source):
            raise ValueError(f'{name} IDs mismatch')
        for key,row in group.items():
            original = source[key]
            if (row['video_path'],row['data_source'],row['gold_action']) != (
                original['video_path'],original['data_source'],chr(65+original['correct_choice'])):
                raise ValueError(f'{name} metadata mismatch')
            if row['correct'] != int(row['predicted_action'] == row['gold_action']):
                raise ValueError(f'{name} correctness mismatch')
        groups[name] = group
        summaries[name] = json.loads((folder/'summary.json').read_text())
        hashes[name] = sha256(file)
    old,new = groups[NAMES[1]],groups[NAMES[2]]
    result = {'protocol':str(ROOT/'JEV10_REWARD_20261003_PROTOCOL.md'),
              'data_manifest_sha256':sha256(DATA/'manifest.json'),
              'data_sha256':manifest['output_sha256'],
              'rows':1000,'video_clusters':len({r['video_path'] for r in old.values()}),
              'summaries':summaries,'prediction_sha256':hashes,
              'proposed_minus_old_accuracy':paired(old,new,'accuracy',20261004),
              'proposed_minus_old_direct_brier':paired(old,new,'brier',20261005),
              'proposed_gained_questions':sum(old[i]['correct']==0 and new[i]['correct']==1 for i in old),
              'proposed_lost_questions':sum(old[i]['correct']==1 and new[i]['correct']==0 for i in old)}
    OUT.mkdir(parents=True)
    (OUT/'comparison.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
    (OUT/'STATUS.txt').write_text('COMPLETED\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('summaries','prediction_sha256')},indent=2))
    for name in NAMES:
        s=summaries[name]
        print(json.dumps({'name':name,'correct':s['correct'],'accuracy':s['accuracy'],
                          'direct_brier':s['direct']['brier'],
                          'expected_brier':s['expected']['brier'],
                          'bins':s['confidence_bins']}))


if __name__ == '__main__':
    main()
