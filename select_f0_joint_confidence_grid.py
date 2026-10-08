"""Select KL strength, teacher weight and checkpoint using development only."""
import json
from pathlib import Path


ROOT = Path('/pfs/hyx/videojev-rlcd')
EVAL = ROOT / 'evaluations/f0_joint_confidence_v1'
STEPS = (4000,8000,12000,16000,16063)
VARIANTS = ('lam0','lam05','lam1')
KL_WEIGHTS = (5,50)
MIN_AGREEMENT = 0.99


def main():
    output = EVAL / 'selection_grid.json'
    if output.exists():
        raise FileExistsError(output)
    result = {'criterion': 'min dev direct-mean-q Brier among checkpoints with >=99% F0 action agreement and no dev answer-accuracy loss; if none eligible, select min Brier for diagnosis only',
              'dev_rows':2000,'min_action_agreement':MIN_AGREEMENT,
              'kl_weights':[5,50],'teacher_weights':[0,0.5,1],
              'runs':{}}
    candidates = []
    for kl in KL_WEIGHTS:
        suffix = '' if kl==5 else '_kl50'
        for variant in VARIANTS:
            key = f'{variant}{suffix}'
            trials = []
            for step in STEPS:
                base = EVAL / 'dev' / key / f'examples_{step:05d}'
                if (base/'STATUS.txt').read_text().strip() != 'COMPLETED 2000':
                    raise ValueError(f'Incomplete dev {key} {step}')
                summary = json.loads((base/'summary.json').read_text())
                eligible = (summary['action_agreement_f0'] >= MIN_AGREEMENT and
                            summary['action_accuracy'] >= summary['f0_action_accuracy'])
                trials.append({'step':step,'eligible':eligible,
                               'action_agreement_f0':summary['action_agreement_f0'],
                               'action_accuracy':summary['action_accuracy'],
                               'f0_action_accuracy':summary['f0_action_accuracy'],
                               'brier':summary['direct_mean_q']['brier'],
                               'auc':summary['direct_mean_q']['auc'],
                               'ece':summary['direct_mean_q']['ece_10_equal_width']})
            pool = [r for r in trials if r['eligible']]
            best = min(pool if pool else trials,key=lambda r:(r['brier'],r['step']))
            result['runs'][key] = {'variant':variant,'action_kl_weight':kl,
                                   'selected_step':best['step'],
                                   'selected_dev':best,'any_eligible':bool(pool),
                                   'trials':trials}
            candidates.append((key,best))
    eligible = [(key,r) for key,r in candidates if r['eligible']]
    key,chosen = min(eligible if eligible else candidates,
                     key=lambda entry:(entry[1]['brier'],entry[1]['step']))
    result['global_selection'] = {'run':key,
                                  'variant':result['runs'][key]['variant'],
                                  'action_kl_weight':result['runs'][key]['action_kl_weight'],
                                  'step':chosen['step'],'eligible':bool(eligible),
                                  'selected_dev':chosen}
    output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    main()
