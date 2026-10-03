"""Collect frozen VideoJev action predictions and native option probabilities."""
import argparse
import json
from pathlib import Path

import torch
import yaml
from peft import PeftModel

from core import parse_output
from model import load_model, prepare_video, to_device, generate_completion, native_action_probs

ROOT = Path("/pfs/hyx/videojev-rlcd")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=str(ROOT / "config.yaml"))
    p.add_argument("--data", default=str(ROOT / "calibration/pool_1600.jsonl"))
    p.add_argument("--output", default=str(ROOT / "calibration/pool_predictions.jsonl"))
    p.add_argument("--adapter", default=str(ROOT / "runs/ablation_none_200/checkpoints/step_0200"))
    p.add_argument("--limit", type=int)
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    rows = [json.loads(s) for s in Path(args.data).read_text().splitlines() if s.strip()]
    if args.limit:
        rows = rows[:args.limit]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    previous = [json.loads(s) for s in output.read_text().splitlines() if s.strip()] if output.exists() else []
    if previous:
        expected = [r["id"] for r in rows[:len(previous)]]
        actual = [r["id"] for r in previous]
        if actual != expected:
            raise ValueError("Existing output does not match the fixed pool order")
    processor, model = load_model(cfg["model"]["path"], args.device)
    model.model.language_model = PeftModel.from_pretrained(
        model.model.language_model, args.adapter, is_trainable=False
    )
    model.to(args.device).eval()
    with output.open("a", encoding="utf-8") as log:
        for i, row in enumerate(rows[len(previous):], start=len(previous)):
            batch = to_device(
                prepare_video(processor, row, cfg["media"]["max_frames"], cfg["media"]["max_pixels"]),
                args.device,
            )
            if batch["input_ids"].shape[1] > cfg["media"]["max_prompt_tokens"]:
                raise ValueError(f"Prompt exceeds budget: {row['id']}")
            with torch.no_grad():
                _, raw = generate_completion(
                    model, processor, batch,
                    max_new_tokens=cfg["evaluation"]["max_new_tokens"], sample=False,
                )
                native = native_action_probs(model, processor, batch, len(row["choices"]))
            decision = parse_output(raw, len(row["choices"]))
            if not decision.valid_format:
                raise ValueError(f"Invalid output on calibration pool: {row['id']}: {raw!r}")
            chosen = decision.action_index
            item = {
                "id": row["id"], "split": row["split"],
                "gold_action": chr(65+row["correct_choice"]),
                "predicted_action": chr(65+chosen),
                "correct": int(chosen == row["correct_choice"]),
                "explicit_confidence": decision.confidence,
                "native_probabilities": native,
                "selected_native_p": native[chosen],
                "raw_generation": raw,
                "num_choices": len(row["choices"]),
            }
            log.write(json.dumps(item, ensure_ascii=False) + "\n")
            log.flush()
            if (i+1) % 25 == 0 or i+1 == len(rows):
                print(json.dumps({"collected": i+1, "total": len(rows), "split": row["split"]}), flush=True)
    (output.parent / (output.stem + ".STATUS.txt")).write_text(f"COMPLETED {len(rows)}\n")


if __name__ == "__main__":
    main()
