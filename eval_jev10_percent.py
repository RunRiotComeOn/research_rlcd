"""Evaluate exact percent outputs on video-disjoint JeV validation/test."""
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import torch
import yaml
from peft import PeftModel

from core import calibration_metrics
from model import load_model, prepare_video, to_device
from percent_policy import greedy_percent, percent_prompt

ROOT = Path('/pfs/hyx/videojev-rlcd')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run-name',required=True)
    p.add_argument('--adapter',required=True)
    p.add_argument('--split',choices=('validation','test'),default='validation')
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--limit',type=int)
    args = p.parse_args()
    out = ROOT/'evaluations/jev10_percent'/args.run_name
    if out.exists():
        raise FileExistsError(out)
    out.mkdir(parents=True)
    cfg = yaml.safe_load((ROOT/'config.yaml').read_text())
    data = [json.loads(s) for s in (ROOT/f'data/jev10_v2/{args.split}.jsonl').read_text().splitlines()]
    if args.limit: data=data[:args.limit]
    processor,model = load_model(cfg['model']['path'],args.device)
    model.requires_grad_(False)
    model.model.language_model=PeftModel.from_pretrained(model.model.language_model,args.adapter,is_trainable=False)
    model.to(args.device).eval()
    rows=[]
    with (out/'predictions.jsonl').open('w') as f:
        for i,row in enumerate(data,1):
            batch=to_device(prepare_video(processor,row,cfg['media']['max_frames'],
                                          cfg['media']['max_pixels'],percent_prompt(row)),args.device)
            with torch.no_grad(): decision=greedy_percent(model,processor,batch,len(row['choices']))
            item={'id':row['id'],'data_source':row['data_source'],
                  'correct':int(decision['action_index']==row['correct_choice']),
                  'gold_action':chr(65+row['correct_choice']),
                  'predicted_action':chr(65+decision['action_index']),**decision}
            rows.append(item); f.write(json.dumps(item)+'\n'); f.flush()
            if i%50==0 or i==len(data): print(json.dumps({'processed':i,'total':len(data)}),flush=True)
    m=calibration_metrics(rows)
    grouped=defaultdict(list)
    for r in rows: grouped[r['percent']//10].append(r)
    summary={'count':len(rows),'split':args.split,'adapter':args.adapter,
             'accuracy':sum(r['correct'] for r in rows)/len(rows),
             'brier':m['brier'],'ece':m['ece'],'nll':m['nll'],
             'percent_counts':dict(sorted(Counter(r['percent'] for r in rows).items())),
             'by_decile':{str(k):{'count':len(v),'mean_reported_percent':sum(r['percent'] for r in v)/len(v),
                                  'accuracy':sum(r['correct'] for r in v)/len(v)}
                          for k,v in sorted(grouped.items())}}
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    (out/'STATUS.txt').write_text('COMPLETED\n')
    print(json.dumps({k:v for k,v in summary.items() if k!='percent_counts'},indent=2),flush=True)


if __name__=='__main__': main()
