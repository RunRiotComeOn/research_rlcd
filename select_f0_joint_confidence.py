"""Choose joint confidence checkpoint and teacher weight using development only."""
import json
from pathlib import Path


ROOT = Path('/pfs/hyx/videojev-rlcd')
EVAL = ROOT / 'evaluations/f0_joint_confidence_v1'
STEPS = (4000,8000,12000,16000,16063)
VARIANTS = ('lam0','lam05','lam1')
MIN_AGREEMENT = 0.99


def main():
    output = EVAL / 'selection.json'
    if output.exists():
        raise FileExistsError(output)
    result = {'criterion': 'min dev direct-mean-q Brier among checkpoints with >=99% F0 action agreement and no dev answer-accuracy loss; if none eligible, select min Brier for diagnosis only',
              'dev_rows':2000,'min_action_agreement':MIN_AGREEMENT,'variants':{}}
    candidates = []
    for variant in VARIANTS:
        trials = []
        for step in STEPS:
            base = EVAL / 'dev' / variant / f'examples_{step:05d}'
            if (base/'STATUS.txt').read_text().strip() != 'COMPLETED 2000':
                raise ValueError(f'Incomplete dev {variant} {step}')
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
        result['variants'][variant] = {'selected_step':best['step'],
                                       'selected_dev':best,'any_eligible':bool(pool),
                                       'trials':trials}
        candidates.append((variant,best))
    eligible = [(name,r) for name,r in candidates if r['eligible']]
    chosen_variant, chosen = min(eligible if eligible else candidates,
                                 key=lambda entry:(entry[1]['brier'],entry[1]['step']))
    result['global_selection'] = {'variant':chosen_variant,'step':chosen['step'],
                                  'eligible':bool(eligible),
                                  'selected_dev':chosen}
    output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    main()
