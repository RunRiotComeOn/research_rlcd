"""Matched validation comparison of proposed reward beta=0.2 versus beta=0.4."""
import hashlib
import json
from pathlib import Path

from analyze_jev10_reward_compare import paired, records

ROOT = Path('/pfs/hyx/videojev-rlcd')
EVAL = ROOT/'evaluations/jev10_reward_compare_v2'
OUT = ROOT/'comparisons/jev10_reward_beta040_v1'
RUN = ROOT/'runs/jev10_reward_proposed_beta040_v1_300'


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    if (RUN/'STATUS.txt').read_text().strip() != 'COMPLETED 300':
        raise ValueError('Beta=.4 training incomplete')
    config = json.loads((RUN/'run_config.json').read_text())
    reference = json.loads((ROOT/'runs/jev10_reward_proposed_v2_300/run_config.json').read_text())
    if config['mode'] != reference['mode'] or config['beta'] != .4 or reference['beta'] != .2:
        raise ValueError('Mode or beta mismatch')
    for key in ('seed','source_adapter_weights_sha256','data_sha256','limit','num_rollouts',
                'action_sampling_temperature','digit_sampling_temperature','learning_rate','objective'):
        if config[key] != reference[key]:
            raise ValueError(f'Training setting mismatch: {key}')
    result = {'protocol':str(ROOT/'JEV10_REWARD_BETA040_PROTOCOL.md'),
              'beta020_run_config_sha256':sha256(ROOT/'runs/jev10_reward_proposed_v2_300/run_config.json'),
              'beta040_run_config_sha256':sha256(RUN/'run_config.json'),
              'rows':481,'steps':{}}
    for step in (100,200,300):
        a=EVAL/f'proposed_step_{step:04d}'
        b=EVAL/f'beta040_step_{step:04d}'
        for folder in (a,b):
            if (folder/'STATUS.txt').read_text().strip() != 'COMPLETED 481':
                raise ValueError(f'Evaluation incomplete: {folder}')
        old={r['id']:r for r in records(a/'predictions.jsonl')}
        new={r['id']:r for r in records(b/'predictions.jsonl')}
        if len(old)!=481 or set(old)!=set(new):
            raise ValueError(f'IDs mismatch at {step}')
        for key in old:
            if (old[key]['video_path'],old[key]['gold_action']) != (new[key]['video_path'],new[key]['gold_action']):
                raise ValueError(f'Metadata mismatch at {step}')
        old_summary=json.loads((a/'summary.json').read_text())
        new_summary=json.loads((b/'summary.json').read_text())
        result['steps'][str(step)]={
            'beta020':{'correct':old_summary['correct'],'accuracy':old_summary['accuracy'],
                       'direct_brier':old_summary['direct']['brier'],
                       'mean_q':old_summary['mean_reported_confidence'],
                       'q9_count':old_summary['confidence_bins'].get('9',0)},
            'beta040':{'correct':new_summary['correct'],'accuracy':new_summary['accuracy'],
                       'direct_brier':new_summary['direct']['brier'],
                       'mean_q':new_summary['mean_reported_confidence'],
                       'q9_count':new_summary['confidence_bins'].get('9',0)},
            'beta040_minus_beta020_accuracy':paired(old,new,'accuracy',20261004+step),
            'beta040_minus_beta020_brier':paired(old,new,'brier',20261005+step),
            'gained_questions':sum(old[k]['correct']==0 and new[k]['correct']==1 for k in old),
            'lost_questions':sum(old[k]['correct']==1 and new[k]['correct']==0 for k in old),
            'beta020_predictions_sha256':sha256(a/'predictions.jsonl'),
            'beta040_predictions_sha256':sha256(b/'predictions.jsonl'),
        }
    OUT.mkdir(parents=True)
    (OUT/'validation.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
    (OUT/'STATUS.txt').write_text('COMPLETED VALIDATION\n')
    print(json.dumps({'rows':result['rows'],'steps':{k:{field:value for field,value in v.items()
            if field not in ('beta020_predictions_sha256','beta040_predictions_sha256')}
            for k,v in result['steps'].items()}},indent=2))


if __name__=='__main__':
    main()
