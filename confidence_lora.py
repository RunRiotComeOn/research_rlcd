"""Train a Qwen-native confidence adapter against frozen action correctness."""
import argparse
import json
import random
import time
from collections import Counter
from pathlib import Path

import torch
import yaml
from peft import PeftModel

from core import calibration_metrics
from model import append_tokens, load_model, prepare_video, to_device

ROOT = Path("/pfs/hyx/videojev-rlcd")
ACTION_ADAPTER = ROOT / "runs/ablation_none_200/checkpoints/step_0200"
OUT = ROOT / "calibration/confidence_lora_v1"


def records(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def score_metrics(predictions):
    rows = [{"correct": r["correct"], "confidence": r["confidence"]} for r in predictions]
    base = calibration_metrics(rows)
    pos = [r["confidence"] for r in predictions if r["correct"]]
    neg = [r["confidence"] for r in predictions if not r["correct"]]
    auc = sum((a > b) + 0.5 * (a == b) for a in pos for b in neg) / (len(pos) * len(neg))
    return {"count": len(rows), "accuracy": sum(r["correct"] for r in rows) / len(rows),
            "brier": base["brier"], "ece": base["ece"], "nll": base["nll"],
            "rank_auc": auc, "bins": dict(sorted(Counter(r["digit"] for r in predictions).items()))}


def confidence(model, processor, batch, action, digit_ids):
    prefix = processor.tokenizer.encode(f"Action: {action}\nConfidence: ", add_special_tokens=False)
    logits = model(**append_tokens(batch, prefix), use_cache=False, logits_to_keep=1).logits[0, -1].float()
    digit_logits = logits[digit_ids]
    probs = digit_logits.softmax(-1)
    centers = torch.arange(10, device=probs.device, dtype=probs.dtype) * 0.1 + 0.05
    return (probs * centers).sum(), probs


def main():
    global OUT
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--loss", choices=("brier", "bce"), default="brier")
    parser.add_argument("--output", default=str(OUT))
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--checkpoint", default=str(OUT / "adapter/confidence"))
    parser.add_argument("--splits", default="select,blind")
    parser.add_argument("--eval-name", default="selected")
    args = parser.parse_args()
    OUT = Path(args.output)
    if args.limit < 1 or args.limit > 1000:
        raise ValueError("--limit must be in 1..1000")
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text())
    pool = records(ROOT / "calibration/pool_1600.jsonl")
    predictions = records(ROOT / "calibration/pool_predictions.jsonl")
    if len(pool) != 1600 or len(predictions) != 1600:
        raise ValueError("Expected the fixed 1600-row calibration pool")
    if [r["id"] for r in pool] != [r["id"] for r in predictions]:
        raise ValueError("Pool/prediction ID order mismatch")
    grouped = {key: [(r, p) for r, p in zip(pool, predictions) if r["split"] == key]
               for key in ("fit", "select", "blind")}
    if [len(grouped[key]) for key in ("fit", "select", "blind")] != [1000, 300, 300]:
        raise ValueError("Split count mismatch")
    torch.manual_seed(20261002)
    random.seed(20261002)
    processor, model = load_model(cfg["model"]["path"], args.device)
    language = PeftModel.from_pretrained(model.model.language_model, str(ACTION_ADAPTER),
                                          adapter_name="answer", is_trainable=False)
    if args.eval:
        language.load_adapter(args.checkpoint, adapter_name="confidence", is_trainable=False)
    else:
        language.load_adapter(str(ACTION_ADAPTER), adapter_name="confidence", is_trainable=True)
    model.model.language_model = language
    language.set_adapter("confidence")
    # PEFT may toggle adapter flags when switching; enforce the frozen answer policy.
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(not args.eval and "confidence" in name and "lora_" in name)
    model.to(args.device)
    digits = [processor.tokenizer.encode(str(i), add_special_tokens=False) for i in range(10)]
    if any(len(x) != 1 for x in digits):
        raise ValueError("Confidence digits must each have one token")
    digit_ids = [x[0] for x in digits]
    if len(set(digit_ids)) != 10:
        raise ValueError("Confidence digit token IDs must be distinct")
    if not args.eval:
        trainable = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
        if not trainable or any("confidence" not in name for name, _ in trainable):
            raise RuntimeError("Only the confidence adapter may be trained")
        (OUT / "run_config.json").write_text(json.dumps({
            "source_adapter": str(ACTION_ADAPTER), "source_pool": "calibration/pool_1600.jsonl",
            "label": "frozen selected action correctness", "train_split": "fit", "limit": args.limit,
            "learning_rate": args.lr, "loss": args.loss + " on expectation over 10 digit tokens",
            "digit_ids": digit_ids, "seed": 20261002,
            "trainable_parameters": sum(p.numel() for _, p in trainable),
        }, indent=2) + "\n")
        optimizer = torch.optim.AdamW([p for _, p in trainable], lr=args.lr, weight_decay=0)
        examples = grouped["fit"][:args.limit]
        random.shuffle(examples)
        model.train()
        start = time.monotonic()
        with (OUT / "train_metrics.jsonl").open("w") as log:
            for step, (row, pred) in enumerate(examples, start=1):
                batch = to_device(prepare_video(processor, row, cfg["media"]["max_frames"],
                                                cfg["media"]["max_pixels"]), args.device)
                if batch["input_ids"].shape[1] > cfg["media"]["max_prompt_tokens"]:
                    raise ValueError(f"Prompt too long: {row['id']}")
                q, probs = confidence(model, processor, batch, pred["predicted_action"], digit_ids)
                target = torch.tensor(float(pred["correct"]), device=q.device)
                if args.loss == "brier":
                    loss = (q - target).square()
                else:
                    loss = torch.nn.functional.binary_cross_entropy(q, target)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_([p for _, p in trainable], 1.0, error_if_nonfinite=True)
                optimizer.step()
                entry = {"step": step, "id": row["id"], "correct": pred["correct"],
                         "q": float(q.detach()), "loss": float(loss.detach()),
                         "grad_norm": float(norm), "elapsed_seconds": time.monotonic() - start}
                log.write(json.dumps(entry) + "\n")
                log.flush()
                if step % 25 == 0 or step == len(examples):
                    print(json.dumps(entry), flush=True)
                if step % 250 == 0 or step == len(examples):
                    checkpoint = OUT / "checkpoints" / f"step_{step:04d}"
                    language.save_pretrained(checkpoint, selected_adapters=["confidence"])
        language.save_pretrained(OUT / "adapter", selected_adapters=["confidence"])
        (OUT / "train.STATUS.txt").write_text(f"COMPLETED {len(examples)}\n")
    else:
        model.eval()
        eval_out = OUT / "evals" / args.eval_name
        eval_out.mkdir(parents=True, exist_ok=True)
        requested = args.splits.split(",")
        if not requested or any(s not in ("select", "blind", "holmes") for s in requested):
            raise ValueError("Invalid evaluation split")
        if "holmes" in requested:
            test_rows = records(ROOT / "data/test.jsonl")
            test_predictions = records(ROOT / "evaluations/ablation_none_200_holmes92/predictions.jsonl")
            by_id = {r["id"]: r for r in test_rows}
            if len(by_id) != 92 or len(test_predictions) != 92:
                raise ValueError("Holmes 92-row evaluation mismatch")
            grouped["holmes"] = [(by_id[p["id"]], p) for p in test_predictions]
        summary = {}
        for split in requested:
            output = []
            with torch.no_grad():
                for i, (row, pred) in enumerate(grouped[split], start=1):
                    batch = to_device(prepare_video(processor, row, cfg["media"]["max_frames"],
                                                    cfg["media"]["max_pixels"]), args.device)
                    q, _ = confidence(model, processor, batch, pred["predicted_action"], digit_ids)
                    q = float(q)
                    digit = max(0, min(9, int(q * 10)))
                    output.append({"id": row["id"], "correct": pred["correct"],
                                   "predicted_action": pred["predicted_action"], "q": q,
                                   "digit": digit, "confidence": 0.05 + 0.1 * digit,
                                   "output": f"Action: {pred['predicted_action']}\nConfidence: {digit}"})
                    if i % 50 == 0:
                        print(json.dumps({"split": split, "processed": i}), flush=True)
            with (eval_out / f"{split}_predictions.jsonl").open("w") as f:
                for item in output:
                    f.write(json.dumps(item) + "\n")
            summary[split] = score_metrics(output)
            print(json.dumps({split: summary[split]}), flush=True)
        (eval_out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        (eval_out / "STATUS.txt").write_text("COMPLETED\n")


if __name__ == "__main__":
    main()
