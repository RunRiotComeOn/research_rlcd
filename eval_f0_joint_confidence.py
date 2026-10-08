"""Evaluate action and confidence directly from one F0-initialized LoRA."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import yaml
from peft import PeftModel
from sklearn.metrics import roc_auc_score

from core import CONFIDENCES
from decision_policy_v2 import greedy_decision
from model import load_model, prepare_video, to_device


ROOT = Path('/pfs/hyx/videojev-rlcd')
DATA = ROOT / 'data/f0_joint_confidence_v1'
EVAL = ROOT / 'evaluations/f0_joint_confidence_v1'
F0_HOLMES = ROOT / 'evaluations/f0_confidence_ablation_v1/holmes_actions/predictions.jsonl'
VARIANTS = ('lam0','lam05','lam1')


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def ece(q,y):
    bins = np.minimum((q*10).astype(int),9)
    return float(sum(mask.mean()*abs(float(q[mask].mean()-y[mask].mean()))
                     for i in range(10) if (mask := bins==i).any()))


def score(q,y):
    q = np.asarray(q,dtype=np.float64)
    return {'brier':float(np.mean((q-y)**2)),
            'auc':float(roc_auc_score(y,q)),
            'ece_10_equal_width':ece(q,y),
            'mean_q':float(q.mean())}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--split',choices=('dev','test','holmes'),required=True)
    parser.add_argument('--variant',choices=VARIANTS,required=True)
    parser.add_argument('--step',type=int,required=True)
    parser.add_argument('--device',required=True)
    parser.add_argument('--resume',action='store_true')
    args = parser.parse_args()
    if args.split == 'holmes':
        data = ROOT / 'data/jev_full_holmes_v1/holmes_all.jsonl'
        action_file = F0_HOLMES
        if (action_file.parent/'STATUS.txt').read_text().strip() != 'COMPLETED 1837':
            raise ValueError('F0 Holmes actions incomplete')
        expected = 1837
    else:
        data = DATA / f'{args.split}.jsonl'
        action_file = DATA / f'{args.split}_actions.jsonl'
        expected = 2000
    run = ROOT / 'runs' / f'f0joint_{args.variant}_v1'
    adapter = run / 'checkpoints' / f'examples_{args.step:05d}'
    if (adapter/'STATUS.txt').read_text().strip() != f'COMPLETED {args.step}':
        raise ValueError('Incomplete adapter checkpoint')
    rows = [json.loads(s) for s in data.read_text(encoding='utf-8').splitlines()]
    f0 = [json.loads(s) for s in action_file.read_text().splitlines()]
    if len(rows) != expected or len(f0) != expected or [r['id'] for r in rows] != [r['id'] for r in f0]:
        raise ValueError('Data/F0 row mismatch')
    if any(a['source_answer_fold'] != 'fold0' for a in f0):
        raise ValueError('Reference answers are not F0')
    out = EVAL / args.split / args.variant / f'examples_{args.step:05d}'
    config = {'split':args.split,'variant':args.variant,'step':args.step,
              'data_sha256':sha256(data),'f0_actions_sha256':sha256(action_file),
              'adapter_sha256':sha256(adapter/'adapter_model.safetensors'),
              'device':args.device}
    if out.exists() and not args.resume:
        raise FileExistsError(out)
    if args.resume and not out.exists():
        raise FileNotFoundError(out)
    if not args.resume:
        out.mkdir(parents=True)
        (out/'config.json').write_text(json.dumps(config,indent=2)+'\n')
        start = 0
    else:
        if json.loads((out/'config.json').read_text()) != config:
            raise ValueError('Resume config mismatch')
        old = [json.loads(s) for s in (out/'predictions.jsonl').read_text().splitlines()]
        start = len(old)
        if [r['id'] for r in old] != [r['id'] for r in rows[:start]]:
            raise ValueError('Resume row mismatch')
    cfg = yaml.safe_load((ROOT/'config.yaml').read_text(encoding='utf-8'))
    processor, model = load_model(cfg['model']['path'],args.device)
    model.requires_grad_(False)
    model.model.language_model = PeftModel.from_pretrained(
        model.model.language_model,str(adapter),is_trainable=False)
    model.to(args.device).eval()
    with (out/'predictions.jsonl').open('a' if args.resume else 'w') as stream:
        for i in range(start,len(rows)):
            row,reference = rows[i],f0[i]
            batch = to_device(prepare_video(processor,row,cfg['media']['max_frames'],
                                            cfg['media']['max_pixels']),args.device)
            if batch['input_ids'].shape[1] > cfg['media']['max_prompt_tokens']:
                raise ValueError(f'Prompt too long: {row["id"]}')
            decision = greedy_decision(model,processor,batch,len(row['choices']))
            action = decision['action_index']
            student_p = np.asarray(decision['action_probabilities'],dtype=np.float64)
            f0_p = np.asarray(reference['action_probabilities'],dtype=np.float64)
            kl = float(np.sum(f0_p*(np.log(np.clip(f0_p,1e-8,1))-
                                     np.log(np.clip(student_p,1e-8,1)))))
            item = {'id':row['id'],'video_path':row['video_path'],
                    'action_index':action,'gold_action_index':row['correct_choice'],
                    'correct':int(action==row['correct_choice']),
                    'selected_action_p':float(student_p[action]),
                    'confidence_bin':decision['confidence_bin'],
                    'confidence_bin_q':decision['confidence'],
                    'mean_q':decision['expected_q'],
                    'output':decision['output'],
                    'action_probabilities':decision['action_probabilities'],
                    'digit_probabilities':decision['digit_probabilities'],
                    'f0_action_index':reference['action_index'],
                    'f0_correct':reference['correct'],'action_kl_to_f0':kl}
            stream.write(json.dumps(item)+'\n')
            stream.flush()
            if (i+1)%100==0 or i+1==len(rows):
                print(json.dumps({'split':args.split,'variant':args.variant,
                                  'step':args.step,'processed':i+1,'total':len(rows)}),flush=True)
    predictions = [json.loads(s) for s in (out/'predictions.jsonl').read_text().splitlines()]
    if len(predictions) != expected:
        raise ValueError('Incomplete predictions')
    y = np.asarray([r['correct'] for r in predictions],dtype=np.float64)
    f0_y = np.asarray([r['f0_correct'] for r in predictions],dtype=np.float64)
    summary = {'split':args.split,'variant':args.variant,'step':args.step,
               'rows':expected,'videos':len({Path(r['video_path']).name for r in predictions}),
               'action_agreement_f0':sum(r['action_index']==r['f0_action_index']
                                         for r in predictions)/expected,
               'action_accuracy':float(y.mean()),'f0_action_accuracy':float(f0_y.mean()),
               'action_accuracy_delta':float(y.mean()-f0_y.mean()),
               'mean_action_kl_to_f0':float(np.mean([r['action_kl_to_f0'] for r in predictions])),
               'constant_oracle_brier':float(y.mean()*(1-y.mean())),
               'direct_mean_q':score([r['mean_q'] for r in predictions],y),
               'direct_argmax_digit_q':score([r['confidence_bin_q'] for r in predictions],y),
               'direct_selected_action_p':score([r['selected_action_p'] for r in predictions],y)}
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    (out/'STATUS.txt').write_text(f'COMPLETED {expected}\n')
    print(json.dumps(summary,indent=2))


if __name__ == '__main__':
    main()
