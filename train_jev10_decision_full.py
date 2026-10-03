"""Full JeV 10% LoRA training with context-correct action labels."""
import argparse
import json
import random
import time
from pathlib import Path

import torch
import yaml

from model import action_label_tokens, append_tokens, load_model, prepare_video, to_device

ROOT=Path('/pfs/hyx/videojev-rlcd')
DATA=ROOT/'data/jev10_v2/train.jsonl'


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--mode',choices=('ce','ce_brier'),required=True)
    p.add_argument('--device',required=True)
    p.add_argument('--run-name',required=True)
    p.add_argument('--limit',type=int,default=4813,help='Debug only; formal runs use 4813')
    args=p.parse_args()
    if not 1<=args.limit<=4813: raise ValueError('Invalid limit')
    out=ROOT/'runs'/args.run_name
    if out.exists(): raise FileExistsError(out)
    rows=[json.loads(s) for s in DATA.read_text().splitlines()]
    if len(rows)!=4813: raise ValueError(f'Expected 4813 training rows; got {len(rows)}')
    random.Random(20261002).shuffle(rows)
    rows=rows[:args.limit]
    cfg=yaml.safe_load((ROOT/'config.yaml').read_text())
    out.mkdir(parents=True)
    config={'data':str(DATA),'source_model':cfg['model']['path'],'mode':args.mode,
            'objective':'CE(gold action) + weight * multiclass Brier(action probabilities)',
            'brier_weight':1.0 if args.mode=='ce_brier' else 0.0,
            'tokenization':'full Action: <LETTER> context; one token per option',
            'lora_rank':8,'learning_rate':2e-5,'seed':20261002,'steps':len(rows),
            'device':args.device}
    (out/'run_config.json').write_text(json.dumps(config,indent=2)+'\n')
    torch.manual_seed(20261002)
    processor,model=load_model(cfg['model']['path'],args.device,lora_rank=8)
    named=[(n,p) for n,p in model.named_parameters() if p.requires_grad]
    if not named or any('lora_' not in n for n,_ in named):
        raise RuntimeError('Only LoRA parameters may train')
    trainable=[p for _,p in named]
    optimizer=torch.optim.AdamW(trainable,lr=config['learning_rate'],weight_decay=0)
    prefix,action_ids=action_label_tokens(processor,26)
    started=time.monotonic()
    model.train()
    with (out/'metrics.jsonl').open('w') as log:
        for step,row in enumerate(rows,1):
            batch=to_device(prepare_video(processor,row,cfg['media']['max_frames'],
                                          cfg['media']['max_pixels']),args.device)
            if batch['input_ids'].shape[1]>cfg['media']['max_prompt_tokens']:
                raise ValueError(f'Prompt too long: {row["id"]}')
            logits=model(**append_tokens(batch,prefix),use_cache=False,
                         logits_to_keep=1).logits[0,-1,action_ids[:len(row['choices'])]].float()
            gold=row['correct_choice']
            probs=logits.softmax(-1)
            target=torch.nn.functional.one_hot(torch.tensor(gold,device=args.device),
                                                num_classes=len(row['choices'])).float()
            ce=torch.nn.functional.cross_entropy(logits.unsqueeze(0),
                                                  torch.tensor([gold],device=args.device))
            multiclass_brier=(probs-target).square().sum()
            loss=ce+config['brier_weight']*multiclass_brier
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            norm=torch.nn.utils.clip_grad_norm_(trainable,1.0,error_if_nonfinite=True)
            optimizer.step()
            entry={'step':step,'id':row['id'],'loss':float(loss.detach()),
                   'ce':float(ce.detach()),'multiclass_brier':float(multiclass_brier.detach()),
                   'correct_before_update':int(int(logits.argmax())==gold),
                   'selected_p':float(probs.max().detach()),
                   'grad_norm':float(norm),'elapsed_seconds':time.monotonic()-started}
            log.write(json.dumps(entry)+'\n');log.flush()
            if step%100==0 or step==len(rows): print(json.dumps(entry),flush=True)
            if step in (500,1000,2000,3000,4000,len(rows)):
                model.model.language_model.save_pretrained(out/'checkpoints'/f'step_{step:04d}')
    (out/'STATUS.txt').write_text(f'COMPLETED {len(rows)}\n')


if __name__=='__main__': main()
