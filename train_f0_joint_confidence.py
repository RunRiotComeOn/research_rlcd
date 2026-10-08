"""Train a single F0-initialized action+confidence LoRA with action KL."""
import argparse
import hashlib
import json
import math
import random
import time
from pathlib import Path

import torch
import yaml
from peft import PeftModel

from core import CONFIDENCES
from decision_policy_v2 import action_tokens, candidate_logits, confidence_tokens
from model import load_model, prepare_video, to_device


ROOT = Path('/pfs/hyx/videojev-rlcd')
DATA = ROOT / 'data/f0_joint_confidence_v1/train.jsonl'
ACTIONS = ROOT / 'data/f0_joint_confidence_v1/train_actions.jsonl'
PLATT = ROOT / 'calibration/f0_joint_confidence_v1/platt/parameters.json'
F0 = ROOT / 'runs/jevfull_action_fold0_v1/checkpoints/examples_24063'
SEED = 20261008
ACCUMULATE = 8
SAVE_EVERY = 4000
MAX_LR = 2e-5
MIN_LR = 2e-6
ACTION_KL_WEIGHT = 5.0
LAMBDAS = {'lam0': 0.0, 'lam05': 0.5, 'lam1': 1.0}


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def set_lr(optimizer, update, total):
    phase = (update-1)/max(1,total-1)
    value = MIN_LR+(MAX_LR-MIN_LR)*0.5*(1+math.cos(math.pi*phase))
    for group in optimizer.param_groups:
        group['lr'] = value
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--variant', choices=LAMBDAS, required=True)
    parser.add_argument('--device', required=True)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--smoke-examples', type=int, default=0)
    args = parser.parse_args()
    teacher_weight = LAMBDAS[args.variant]
    rows = [json.loads(s) for s in DATA.read_text(encoding='utf-8').splitlines()]
    actions = [json.loads(s) for s in ACTIONS.read_text(encoding='utf-8').splitlines()]
    assert len(rows) == len(actions) == 16063
    assert [r['id'] for r in rows] == [a['id'] for a in actions]
    assert all(a['source_answer_fold'] == 'fold0' for a in actions)
    if (ROOT / 'calibration/f0_joint_confidence_v1/platt/STATUS.txt').read_text().strip() != 'COMPLETED 16063':
        raise ValueError('Training-only teacher Platt incomplete')
    if args.smoke_examples:
        assert 1 <= args.smoke_examples <= len(rows)
        rows = rows[:args.smoke_examples]
        actions = actions[:args.smoke_examples]
    out_name = (f'f0joint_{args.variant}_smoke_{args.smoke_examples}'
                if args.smoke_examples else f'f0joint_{args.variant}_v1')
    out = ROOT / 'runs' / out_name
    if out.exists() and not args.resume:
        raise FileExistsError(out)
    if args.resume and not out.exists():
        raise FileNotFoundError(out)
    total = len(rows)
    total_updates = math.ceil(total/ACCUMULATE)
    order = list(range(total))
    random.Random(SEED).shuffle(order)
    cfg = yaml.safe_load((ROOT / 'config.yaml').read_text(encoding='utf-8'))
    platt = json.loads(PLATT.read_text())
    config = {'variant': args.variant, 'teacher_weight': teacher_weight,
              'action_kl_weight': ACTION_KL_WEIGHT, 'max_lr': MAX_LR, 'min_lr': MIN_LR,
              'data': str(DATA), 'data_sha256': sha256(DATA),
              'actions_sha256': sha256(ACTIONS), 'teacher_sha256': sha256(PLATT),
              'f0_adapter': str(F0), 'f0_adapter_sha256': sha256(F0 / 'adapter_model.safetensors'),
              'examples': total, 'model': cfg['model']['path'], 'lora_rank': 8,
              'objective': 'Brier of mean confidence on F0 action + teacher MSE + action KL',
              'seed': SEED, 'effective_batch': ACCUMULATE, 'epochs': 1,
              'save_every': SAVE_EVERY, 'device': args.device}
    if not args.resume:
        out.mkdir(parents=True)
        (out / 'config.json').write_text(json.dumps(config, indent=2)+'\n')
    elif json.loads((out / 'config.json').read_text()) != config:
        raise ValueError('Resume config mismatch')
    completed = []
    if args.resume:
        for path in (out / 'checkpoints').glob('examples_*'):
            marker = path / 'STATUS.txt'
            if marker.exists() and (path / 'optimizer.pt').exists():
                n = int(path.name.split('_')[1])
                if marker.read_text().strip() == f'COMPLETED {n}':
                    completed.append((n,path))
        if not completed:
            raise ValueError('No complete checkpoint with optimizer')
    start, checkpoint = max(completed) if completed else (0,None)
    torch.manual_seed(SEED)
    processor, model = load_model(cfg['model']['path'], args.device)
    model.requires_grad_(False)
    model.model.language_model = PeftModel.from_pretrained(
        model.model.language_model, str(checkpoint if checkpoint else F0), is_trainable=True)
    model.to(args.device).train()
    trainable = [(n,p) for n,p in model.named_parameters() if p.requires_grad]
    if not trainable or any('lora_' not in n for n,_ in trainable):
        raise RuntimeError('Unexpected trainable parameters')
    optimizer = torch.optim.AdamW((p for _,p in trainable), lr=MAX_LR, weight_decay=0)
    if checkpoint:
        state = torch.load(checkpoint / 'optimizer.pt', map_location='cpu', weights_only=True)
        optimizer.load_state_dict(state['optimizer'])
        torch.set_rng_state(state['cpu_rng'])
        torch.cuda.set_rng_state(state['cuda_rng'], device=args.device)
    q_values = torch.tensor(CONFIDENCES, dtype=torch.float32, device=args.device)
    action_token_sets = {n: action_tokens(processor,n) for n in {len(r['choices']) for r in rows}}
    confidence_token_sets = {a: confidence_tokens(processor,a)
                             for a in {r['action_index'] for r in actions}}
    optimizer.zero_grad(set_to_none=True)
    started = time.monotonic()
    running = {'brier': [], 'teacher': [], 'action_kl': [], 'action_agree': []}
    with (out / 'metrics.jsonl').open('a' if args.resume else 'w') as stream:
        for step in range(start+1,total+1):
            row, label = rows[order[step-1]], actions[order[step-1]]
            batch = to_device(prepare_video(processor,row,cfg['media']['max_frames'],
                                            cfg['media']['max_pixels']),args.device)
            if batch['input_ids'].shape[1] > cfg['media']['max_prompt_tokens']:
                raise ValueError(f'Prompt too long: {row["id"]}')
            action_prefix, action_ids = action_token_sets[len(row['choices'])]
            student_action_logits = candidate_logits(model,batch,action_prefix,action_ids)
            student_log_probs = student_action_logits.log_softmax(-1)
            target_probs = torch.tensor(label['action_probabilities'],dtype=torch.float32,
                                        device=args.device)
            target_probs = target_probs/target_probs.sum()
            action_kl = torch.sum(target_probs*(target_probs.clamp_min(1e-8).log()-student_log_probs))
            conf_prefix, conf_ids = confidence_token_sets[label['action_index']]
            digit_probs = candidate_logits(model,batch,conf_prefix,conf_ids).softmax(-1)
            q = (digit_probs*q_values).sum()
            brier = (q-label['correct']).square()
            chosen_p = float(label['action_probabilities'][label['action_index']])
            p = min(max(chosen_p,1e-6),1-1e-6)
            teacher_q = torch.sigmoid(torch.tensor(
                platt['slope']*math.log(p/(1-p))+platt['intercept'],
                dtype=torch.float32,device=args.device))
            teacher_loss = (q-teacher_q).square()
            loss = brier+teacher_weight*teacher_loss+ACTION_KL_WEIGHT*action_kl
            group_size = min(ACCUMULATE,total-((step-1)//ACCUMULATE)*ACCUMULATE)
            (loss/group_size).backward()
            running['brier'].append(float(brier.detach()))
            running['teacher'].append(float(teacher_loss.detach()))
            running['action_kl'].append(float(action_kl.detach()))
            running['action_agree'].append(int(student_action_logits.argmax() == label['action_index']))
            if step % ACCUMULATE == 0 or step == total:
                update = math.ceil(step/ACCUMULATE)
                lr = set_lr(optimizer,update,total_updates)
                norm = torch.nn.utils.clip_grad_norm_([p for _,p in trainable],1.0,
                                                       error_if_nonfinite=True)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            else:
                lr,norm = None,None
            if step % 100 == 0 or step == total:
                n = min(100,len(running['brier']))
                entry = {'examples':step,'updates':math.ceil(step/ACCUMULATE),
                         'brier_last_100':sum(running['brier'][-n:])/n,
                         'teacher_mse_last_100':sum(running['teacher'][-n:])/n,
                         'action_kl_last_100':sum(running['action_kl'][-n:])/n,
                         'action_agreement_last_100':sum(running['action_agree'][-n:])/n,
                         'mean_q_last':float(q.detach()),'learning_rate':lr,
                         'grad_norm':float(norm) if norm is not None else None,
                         'elapsed_seconds':round(time.monotonic()-started)}
                stream.write(json.dumps(entry)+'\n')
                stream.flush()
                print(json.dumps(entry),flush=True)
            if step % SAVE_EVERY == 0 or step == total:
                path = out / 'checkpoints' / f'examples_{step:05d}'
                model.model.language_model.save_pretrained(path)
                torch.save({'optimizer':optimizer.state_dict(),
                            'cpu_rng':torch.get_rng_state(),
                            'cuda_rng':torch.cuda.get_rng_state(device=args.device)},path/'optimizer.pt')
                (path/'STATUS.txt').write_text(f'COMPLETED {step}\n')
                for old in (out/'checkpoints').glob('examples_*/optimizer.pt'):
                    if old != path/'optimizer.pt':
                        old.unlink()
    (out/'STATUS.txt').write_text(f'COMPLETED {total}\n')


if __name__ == '__main__':
    main()
