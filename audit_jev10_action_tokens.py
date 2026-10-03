"""Inspect full-vocabulary next-token mass at the action decision position."""
import json
from pathlib import Path

import torch
import yaml
from peft import PeftModel

from model import append_tokens, load_model, prepare_video, to_device

ROOT=Path('/pfs/hyx/videojev-rlcd')
cfg=yaml.safe_load((ROOT/'config.yaml').read_text())
processor,model=load_model(cfg['model']['path'],'cuda:6')
model.model.language_model=PeftModel.from_pretrained(
    model.model.language_model,
    str(ROOT/'runs/jev10_action_sft_v1/checkpoints/step_4813'),is_trainable=False)
model.to('cuda:6').eval()
t=processor.tokenizer
prefix=t.encode('Action: ',add_special_tokens=False)
for s in ('Action: ','Action: A','Action: B','A',' A','B',' B','<LETTER>'):
    print('ENC',repr(s),t.encode(s,add_special_tokens=False),flush=True)
rows=[json.loads(s) for s in (ROOT/'data/jev10_v2/validation.jsonl').read_text().splitlines()[:5]]
for row in rows:
    batch=to_device(prepare_video(processor,row,cfg['media']['max_frames'],
                                  cfg['media']['max_pixels']),'cuda:6')
    with torch.no_grad():
        logits=model(**append_tokens(batch,prefix),use_cache=False,
                     logits_to_keep=1).logits[0,-1].float()
        probs=logits.softmax(-1)
        values,indices=probs.topk(15)
    print('ROW',row['id'],'gold',chr(65+row['correct_choice']),flush=True)
    print('TOP',[(int(i),repr(t.decode([int(i)])),float(v)) for i,v in zip(indices,values)],flush=True)
    ids=[t.encode(chr(65+i),add_special_tokens=False)[0] for i in range(len(row['choices']))]
    print('A_D_MASS',float(probs[ids].sum()),'label_probs',[float(probs[i]) for i in ids],flush=True)
