"""Continue the JeV action adapter with a lower learning rate after step 500."""
import json
import random
import time
from pathlib import Path

import torch
import yaml
from peft import PeftModel

from model import append_tokens, load_model, prepare_video, to_device

ROOT = Path('/pfs/hyx/videojev-rlcd')
OUT = ROOT / 'runs/jev10_action_sft_low_lr_v1'
SOURCE = ROOT / 'runs/jev10_action_sft_v1/checkpoints/step_0500'
DEVICE = 'cuda:6'


def main():
    if OUT.exists():
        raise FileExistsError(f'Refusing to overwrite {OUT}')
    cfg = yaml.safe_load((ROOT / 'config.yaml').read_text())
    rows = [json.loads(s) for s in (ROOT / 'data/jev10_v2/train.jsonl').read_text().splitlines()]
    if len(rows) != 4813:
        raise ValueError('Wrong train size')
    random.Random(20261002).shuffle(rows)
    rows = rows[500:]
    OUT.mkdir(parents=True)
    (OUT / 'run_config.json').write_text(json.dumps({
        'source_adapter': str(SOURCE), 'start_step': 500, 'end_step': 4813,
        'learning_rate': 2e-6, 'objective': 'restricted-option action cross entropy',
        'device': DEVICE, 'data': str(ROOT / 'data/jev10_v2/train.jsonl'),
    }, indent=2) + '\n')
    torch.manual_seed(20261002)
    processor, model = load_model(cfg['model']['path'], DEVICE)
    model.requires_grad_(False)
    model.model.language_model = PeftModel.from_pretrained(model.model.language_model,
                                                           str(SOURCE), is_trainable=True)
    model.to(DEVICE)
    named_trainable = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    if not named_trainable or any('lora_' not in n for n, _ in named_trainable):
        raise ValueError('Only LoRA parameters may train')
    trainable = [p for _, p in named_trainable]
    optimizer = torch.optim.AdamW(trainable, lr=2e-6, weight_decay=0)
    prefix = processor.tokenizer.encode('Action: ', add_special_tokens=False)
    option_ids = [processor.tokenizer.encode(chr(65+i), add_special_tokens=False) for i in range(26)]
    if any(len(x) != 1 for x in option_ids):
        raise ValueError('Action letters must be one token')
    option_ids = [x[0] for x in option_ids]
    started = time.monotonic()
    model.train()
    with (OUT / 'metrics.jsonl').open('w') as log:
        for step, row in enumerate(rows, start=501):
            batch = to_device(prepare_video(processor, row, cfg['media']['max_frames'],
                                            cfg['media']['max_pixels']), DEVICE)
            if batch['input_ids'].shape[1] > cfg['media']['max_prompt_tokens']:
                raise ValueError(f'Prompt too long: {row["id"]}')
            logits = model(**append_tokens(batch, prefix), use_cache=False,
                           logits_to_keep=1).logits[0, -1, option_ids[:len(row['choices'])]].float()
            target = torch.tensor([row['correct_choice']], device=DEVICE)
            loss = torch.nn.functional.cross_entropy(logits.unsqueeze(0), target)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(trainable, 1.0, error_if_nonfinite=True)
            optimizer.step()
            entry = {'step': step, 'id': row['id'], 'loss': float(loss.detach()),
                     'correct_before_update': int(int(logits.argmax()) == row['correct_choice']),
                     'grad_norm': float(norm), 'elapsed_seconds': time.monotonic()-started}
            log.write(json.dumps(entry) + '\n')
            log.flush()
            if step % 100 == 0 or step == 4813:
                print(json.dumps(entry), flush=True)
            if step in (1000, 2000, 3000, 4000, 4813):
                model.model.language_model.save_pretrained(OUT / 'checkpoints' / f'step_{step:04d}')
    model.model.language_model.save_pretrained(OUT / 'adapter')
    (OUT / 'STATUS.txt').write_text('COMPLETED 4813\n')


if __name__ == '__main__':
    main()
