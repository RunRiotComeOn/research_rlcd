"""Short RLCD pilot with a directly generated integer percentage."""
import argparse
import json
import random
import time
from pathlib import Path

import torch
import yaml
from peft import PeftModel

from constrained_policy import action_prefix, logits_for, token_ids
from model import load_model, prepare_video, to_device
from percent_policy import percent_logprob, percent_prompt, sample_percent

ROOT = Path('/pfs/hyx/videojev-rlcd')
START = ROOT / 'runs/jev10_action_sft_v1/checkpoints/step_4813'
ACTION_TEMPERATURE = 2.0


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--limit', type=int, default=300)
    p.add_argument('--run-name', default='jev10_percent_rlcd_300')
    args = p.parse_args()
    if not 1 <= args.limit <= 4813:
        raise ValueError('Invalid limit')
    out = ROOT / 'runs' / args.run_name
    if out.exists():
        raise FileExistsError(out)
    out.mkdir(parents=True)
    config = {'source_adapter': str(START), 'data': 'data/jev10_v2/train.jsonl',
              'limit': args.limit, 'seed': 20261003, 'num_rollouts': 4,
              'lambda_cal': 0.2, 'learning_rate': 2e-5,
              'objective': 'group-centered REINFORCE; no reward-std division',
              'action_sampling_temperature': ACTION_TEMPERATURE,
              'confidence_support': '000% through 100%', 'device': args.device}
    (out / 'run_config.json').write_text(json.dumps(config, indent=2)+'\n')
    cfg = yaml.safe_load((ROOT / 'config.yaml').read_text())
    rows = [json.loads(s) for s in (ROOT / 'data/jev10_v2/train.jsonl').read_text().splitlines()]
    random.Random(config['seed']).shuffle(rows)
    rows = rows[:args.limit]
    torch.manual_seed(config['seed'])
    processor, model = load_model(cfg['model']['path'], args.device)
    model.requires_grad_(False)
    model.model.language_model = PeftModel.from_pretrained(model.model.language_model,
                                                           str(START), is_trainable=True)
    model.to(args.device)
    trainable = [p for n,p in model.named_parameters() if p.requires_grad and 'lora_' in n]
    if not trainable or len(trainable) != sum(p.requires_grad for p in model.parameters()):
        raise RuntimeError('Only language LoRA weights may train')
    optimizer = torch.optim.AdamW(trainable, lr=config['learning_rate'], weight_decay=0)
    started = time.monotonic()
    with (out / 'metrics.jsonl').open('w') as log:
        for step,row in enumerate(rows,1):
            batch = to_device(prepare_video(processor, row, cfg['media']['max_frames'],
                                            cfg['media']['max_pixels'], percent_prompt(row)), args.device)
            if batch['input_ids'].shape[1] > cfg['media']['max_prompt_tokens']:
                raise ValueError(f'Prompt too long: {row["id"]}')
            action_ids,_ = token_ids(processor, len(row['choices']))
            model.eval()
            with torch.no_grad():
                ap = (logits_for(model, batch, action_prefix(processor), action_ids)
                      / ACTION_TEMPERATURE).softmax(-1)
                actions = torch.multinomial(ap, 4, replacement=True).tolist()
                percents = [sample_percent(model, processor, batch, a) for a in actions]
            ys = [int(a == row['correct_choice']) for a in actions]
            rewards = [y - 0.2*(v/100-y)**2 for y,v in zip(ys,percents)]
            rt = torch.tensor(rewards, dtype=torch.float32, device=args.device)
            adv = rt-rt.mean()
            entry = {'step':step,'id':row['id'],'actions':actions,'percents':percents,
                     'correct':ys,'rewards':rewards,'mean_reward':float(rt.mean()),
                     'reward_std':float(rt.std(unbiased=False)),
                     'skipped':bool(float(adv.abs().max()) < 1e-8)}
            if not entry['skipped']:
                model.train()
                optimizer.zero_grad(set_to_none=True)
                action_logp = (logits_for(model, batch, action_prefix(processor), action_ids)
                               / ACTION_TEMPERATURE).log_softmax(-1)
                aloss = -(adv.detach()*action_logp[torch.tensor(actions,device=args.device)]).mean()
                aloss.backward()
                del aloss, action_logp
                for i,(a,v) in enumerate(zip(actions,percents)):
                    loss = -adv[i].detach()*percent_logprob(model, processor, batch, a, v)/4
                    loss.backward()
                    del loss
                norm = torch.nn.utils.clip_grad_norm_(trainable,1.0,error_if_nonfinite=True)
                optimizer.step()
                entry['grad_norm'] = float(norm)
            entry['elapsed_seconds'] = time.monotonic()-started
            log.write(json.dumps(entry)+'\n'); log.flush()
            if step % 25 == 0 or step == len(rows):
                print(json.dumps({k:v for k,v in entry.items()
                                  if k not in ('actions','percents','correct','rewards')}),flush=True)
            if step % 100 == 0 or step == len(rows):
                model.model.language_model.save_pretrained(out/'checkpoints'/f'step_{step:04d}')
    (out/'STATUS.txt').write_text(f'COMPLETED {len(rows)}\n')


if __name__ == '__main__':
    main()
