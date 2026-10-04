"""Inspect reward/advantage composition and validation trajectories for beta scheduling."""
import json
import math
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BETA = .2
Q = tuple(.05+.1*i for i in range(10))


def read(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]


def avg(values):
    return sum(values)/len(values) if values else None


def centered(values):
    mean = avg(values)
    return [v-mean for v in values]


def stage(rows):
    ys = [y for row in rows for y in row['correct']]
    qs = [Q[d] for row in rows for d in row['sampled_digits']]
    rewards = [r for row in rows for r in row['rewards']]
    bins = Counter(d for row in rows for d in row['sampled_digits'])
    kinds = Counter()
    correctness_energy = calibration_energy = total_energy = 0.
    for row in rows:
        y = row['correct']
        q = [Q[d] for d in row['sampled_digits']]
        n_correct = sum(y)
        if n_correct == 4:
            kinds['all_correct'] += 1
        elif n_correct == 0:
            kinds['all_wrong'] += 1
        else:
            kinds['mixed'] += 1
        if n_correct in (0,4) and not row['skipped']:
            kinds['same_y_confidence_only_updates'] += 1
        yc = centered(y)
        rc = centered(row['rewards'])
        cc = [r-v for r,v in zip(rc,yc)]
        correctness_energy += sum(v*v for v in yc)
        calibration_energy += sum(v*v for v in cc)
        total_energy += sum(v*v for v in rc)
    return {
        'steps': [rows[0]['step'],rows[-1]['step']],
        'mean_reward': avg(rewards),
        'sampled_correct_rate': avg(ys),
        'mean_q': avg(qs),
        'mean_q_when_correct': avg([q for q,y in zip(qs,ys) if y]),
        'mean_q_when_wrong': avg([q for q,y in zip(qs,ys) if not y]),
        'sampled_binary_brier': avg([(q-y)**2 for q,y in zip(qs,ys)]),
        'mean_reward_std_within_group': avg([r['reward_std'] for r in rows]),
        'mean_action_entropy': avg([r['action_entropy'] for r in rows]),
        'mean_digit_entropy': avg([r['digit_entropy'] for r in rows]),
        'mean_grad_norm_when_updated': avg([r['grad_norm'] for r in rows if not r['skipped']]),
        'updates':sum(not r['skipped'] for r in rows),
        'skipped':sum(r['skipped'] for r in rows),
        'group_types':dict(kinds),
        'confidence_advantage_rms_to_correctness_advantage_rms':
            math.sqrt(calibration_energy/correctness_energy) if correctness_energy else None,
        'confidence_advantage_energy_share_of_total': calibration_energy/total_energy if total_energy else None,
        'q_bins':{str(k):bins[k] for k in range(10)},
    }


def main():
    summary=json.loads((ROOT/'jev10_reward_compare_v2.json').read_text(encoding='utf-8'))['summaries']
    out={}
    for mode,file in (('old_brier','reward_old_metrics.jsonl'),
                      ('proposed','reward_new_metrics.jsonl')):
        rows=read(ROOT/file)
        if len(rows)!=300 or [r['step'] for r in rows]!=list(range(1,301)):
            raise ValueError(f'{mode}: bad steps')
        for row in rows:
            for y,d,r in zip(row['correct'],row['sampled_digits'],row['rewards']):
                q=Q[d]
                expected=(y-BETA*(q-y)**2 if mode=='old_brier'
                          else y+BETA*(2*y*q-q*q))
                if abs(r-expected)>1e-8:
                    raise ValueError(f'{mode}: reward mismatch at {row["step"]}')
        windows=[stage(rows[i:i+50]) for i in range(0,300,50)]
        windows25=[stage(rows[i:i+25]) for i in range(0,300,25)]
        checkpoints=[]
        for step in (100,200,300):
            s=summary[f'{mode}_step_{step:04d}']
            checkpoints.append({'step':step,'validation_correct':s['correct'],
                                'validation_accuracy':s['accuracy'],
                                'validation_direct_brier':s['direct']['brier'],
                                'validation_mean_q':s['mean_reported_confidence'],
                                'validation_q9_count':s['confidence_bins'].get('9',0),
                                'training_previous_100':stage(rows[step-100:step])})
        out[mode]={'windows_25':windows25,'windows_50':windows,
                   'checkpoints':checkpoints,'whole_run':stage(rows)}
    (ROOT/'jev10_reward_beta_dynamics.json').write_text(json.dumps(out,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    for mode in out:
        print(mode)
        for c in out[mode]['checkpoints']:
            t=c['training_previous_100']
            print(json.dumps({'step':c['step'],'train_reward':round(t['mean_reward'],4),
                              'train_y':round(t['sampled_correct_rate'],4),
                              'train_q':round(t['mean_q'],4),
                              'train_q_correct':round(t['mean_q_when_correct'],4),
                              'train_q_wrong':round(t['mean_q_when_wrong'],4),
                              'train_brier':round(t['sampled_binary_brier'],4),
                              'train_q9':t['q_bins']['9'],
                              'same_y_conf_only_updates':t['group_types'].get('same_y_confidence_only_updates',0),
                              'updates':t['updates'],
                              'mean_action_entropy':round(t['mean_action_entropy'],4),
                              'mean_digit_entropy':round(t['mean_digit_entropy'],4),
                              'confidence_advantage_rms_ratio':round(t['confidence_advantage_rms_to_correctness_advantage_rms'],4),
                              'val_accuracy':round(c['validation_accuracy'],4),
                              'val_brier':round(c['validation_direct_brier'],4),
                              'val_q9':c['validation_q9_count']}))
        print('windows',json.dumps([{'steps':w['steps'],'train_y':round(w['sampled_correct_rate'],3),
                                    'train_q':round(w['mean_q'],3),
                                    'train_q9':w['q_bins']['9'],'updates':w['updates'],
                                    'same_y_only':w['group_types'].get('same_y_confidence_only_updates',0)}
                                   for w in out[mode]['windows_50']]))
        print('windows25',json.dumps([{'steps':w['steps'],'train_y':round(w['sampled_correct_rate'],3),
                                      'train_q':round(w['mean_q'],3),'train_q9':w['q_bins']['9'],
                                      'updates':w['updates'],'digit_entropy':round(w['mean_digit_entropy'],3)}
                                     for w in out[mode]['windows_25']]))


if __name__=='__main__':
    main()
