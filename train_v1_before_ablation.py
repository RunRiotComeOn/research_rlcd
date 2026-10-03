"""Small VideoJev SFT bootstrap and grouped policy optimization trainer."""
import argparse
import json
import math
import random
import time
from pathlib import Path

import torch
import yaml
from peft import PeftModel

from core import assert_correctness_margin, parse_output, reward
from model import load_model, prepare_video, to_device, generate_completion, completion_logprobs

ROOT = Path("/pfs/hyx/videojev-rlcd")


def records(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def attach_adapter(model, path, trainable=True):
    model.requires_grad_(False)
    model.model.language_model = PeftModel.from_pretrained(
        model.model.language_model, str(path), is_trainable=trainable
    )
    if trainable and not any(p.requires_grad for p in model.parameters()):
        raise RuntimeError("No adapter parameters are trainable")
    return model


def save_adapter(model, path):
    path.mkdir(parents=True, exist_ok=True)
    model.model.language_model.save_pretrained(path)


def sft_step(model, processor, batch, gold, confidence_bin, optimizer):
    # Balanced arbitrary bins teach the output vocabulary without inventing calibrated labels.
    target = f"Action: {chr(65+gold)}\nConfidence: {confidence_bin}"
    completion = processor.tokenizer.encode(target, add_special_tokens=False)
    eos = processor.tokenizer.eos_token_id
    if eos is not None:
        completion.append(eos)
    optimizer.zero_grad(set_to_none=True)
    loss = -completion_logprobs(model, batch, completion).mean()
    loss.backward()
    grad = torch.nn.utils.clip_grad_norm_(
        [p for p in model.parameters() if p.requires_grad], 1.0, error_if_nonfinite=True
    )
    optimizer.step()
    return {"loss": float(loss.detach()), "grad_norm": float(grad)}


def grpo_step(model, processor, batch, gold, optimizer, config, lambda_cal):
    model.eval()
    completions = []
    with torch.no_grad():
        for _ in range(config["num_generations"]):
            ids, text = generate_completion(
                model, processor, batch,
                max_new_tokens=config["max_new_tokens"],
                temperature=config["temperature"],
                top_p=config["top_p"],
                sample=True,
            )
            decision = parse_output(text, config["num_choices"])
            score = reward(decision, gold, lambda_cal, config["format_penalty"])
            old = completion_logprobs(model, batch, ids).detach()
            completions.append((ids, text, decision, score, old))
    values = torch.tensor([x[3] for x in completions], device=batch["input_ids"].device)
    std = values.std(unbiased=False)
    advantages = (values - values.mean()) / (std + 1e-6) if std > 1e-8 else torch.zeros_like(values)
    info = {
        "rewards": [float(x[3]) for x in completions],
        "raw_generations": [x[1] for x in completions],
        "valid_format_rate": sum(x[2].valid_format for x in completions) / len(completions),
        "correct_rate": sum(x[2].valid_format and x[2].action_index == gold for x in completions) / len(completions),
        "reward_std": float(std),
    }
    if not bool((advantages != 0).any()):
        info["skipped_zero_advantage"] = True
        return info
    model.train()
    optimizer.zero_grad(set_to_none=True)
    losses = []
    for (ids, _, _, _, old), adv in zip(completions, advantages):
        logp = completion_logprobs(model, batch, ids)
        ratio = (logp - old).clamp(-20, 20).exp()
        clipped = ratio.clamp(1-config["clip_epsilon"], 1+config["clip_epsilon"])
        objective = torch.minimum(ratio * adv, clipped * adv).mean()
        loss = -objective / len(completions)
        loss.backward()
        losses.append(float(loss.detach()))
    grad = torch.nn.utils.clip_grad_norm_(
        [p for p in model.parameters() if p.requires_grad], config["max_grad_norm"],
        error_if_nonfinite=True,
    )
    optimizer.step()
    info.update(loss=sum(losses), grad_norm=float(grad), skipped_zero_advantage=False)
    return info


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "config.yaml"))
    parser.add_argument("--mode", choices=("sft", "rl_only", "rlcd"), required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--adapter", help="A saved common SFT adapter for RL runs")
    parser.add_argument("--limit", type=int, help="Maximum training examples for a smoke run")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    if args.mode != "sft" and not args.adapter:
        parser.error("RL modes require --adapter from the common SFT run")
    lambda_cal = 0.0 if args.mode == "rl_only" else cfg["reward"]["calibration_weight"]
    if args.mode == "rlcd":
        assert_correctness_margin(lambda_cal)
    out = ROOT / "runs" / args.run_name
    out.mkdir(parents=True, exist_ok=False)
    (out / "run_config.json").write_text(json.dumps(
        {"mode": args.mode, "config": cfg, "adapter": args.adapter, "limit": args.limit}, indent=2
    ))
    torch.manual_seed(cfg["seed"])
    random.seed(cfg["seed"])
    processor, model = load_model(cfg["model"]["path"], args.device, 0 if args.adapter else cfg["training"]["lora_rank"])
    if args.adapter:
        model = attach_adapter(model, args.adapter)
    trainable = [p for p in model.parameters() if p.requires_grad]
    if not trainable:
        raise RuntimeError("No trainable parameters")
    optimizer = torch.optim.AdamW(trainable, lr=cfg["training"]["learning_rate"], weight_decay=0)
    data = records(cfg["data"]["train"])
    random.Random(cfg["seed"]).shuffle(data)
    if args.limit:
        data = data[:args.limit]
    elif cfg["training"].get("max_examples"):
        data = data[:cfg["training"]["max_examples"]]
    start = time.monotonic()
    with (out / "metrics.jsonl").open("w") as log:
        for step, row in enumerate(data):
            batch = to_device(
                prepare_video(processor, row, cfg["media"]["max_frames"], cfg["media"]["max_pixels"]),
                args.device,
            )
            if batch["input_ids"].shape[1] > cfg["media"]["max_prompt_tokens"]:
                entry = {"step": step, "id": row["id"], "skipped_too_long": int(batch["input_ids"].shape[1])}
            else:
                if args.mode == "sft":
                    model.train()
                    entry = sft_step(model, processor, batch, row["correct_choice"], step % 10, optimizer)
                else:
                    generation = dict(cfg["generation"], num_choices=len(row["choices"]),
                                      format_penalty=cfg["reward"]["format_penalty"])
                    entry = grpo_step(model, processor, batch, row["correct_choice"], optimizer, generation, lambda_cal)
                entry.update(step=step, id=row["id"], prompt_tokens=int(batch["input_ids"].shape[1]))
            entry["elapsed_seconds"] = time.monotonic() - start
            log.write(json.dumps(entry, ensure_ascii=False) + "\n")
            log.flush()
            if (step+1) % cfg["training"]["save_every"] == 0:
                save_adapter(model, out / "adapter")
            print(json.dumps({k: v for k, v in entry.items() if k not in ("raw_generations", "rewards")}), flush=True)
    save_adapter(model, out / "adapter")
    (out / "STATUS.txt").write_text(f"COMPLETED {len(data)} examples\n")


if __name__ == "__main__":
    main()
