"""Evaluate VideoJev decisions, explicit and native confidence, and calibration."""
import argparse
import json
from pathlib import Path

import torch
import yaml
from peft import PeftModel

from core import parse_output, prompt_text, summarize
from model import load_model, prepare_video, to_device, generate_completion, native_action_probs

ROOT = Path("/pfs/hyx/videojev-rlcd")


def svg_plot(points, path, title, x_label, y_label, ideal=False):
    width, height = 640, 440
    left, right, top, bottom = 70, 35, 45, 65
    x0, x1 = left, width-right
    y0, y1 = top, height-bottom
    def px(x): return x0 + x * (x1-x0)
    def py(y): return y1 - y * (y1-y0)
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
             '<rect width="100%" height="100%" fill="white"/>',
             f'<text x="{width/2}" y="26" text-anchor="middle" font-size="18">{title}</text>',
             f'<path d="M{x0},{y0} V{y1} H{x1}" fill="none" stroke="#333" stroke-width="2"/>']
    for i in range(6):
        value = i/5
        parts += [
            f'<line x1="{px(value)}" y1="{y1}" x2="{px(value)}" y2="{y1+5}" stroke="#333"/>',
            f'<text x="{px(value)}" y="{y1+21}" text-anchor="middle" font-size="11">{value:.1f}</text>',
            f'<line x1="{x0-5}" y1="{py(value)}" x2="{x0}" y2="{py(value)}" stroke="#333"/>',
            f'<text x="{x0-10}" y="{py(value)+4}" text-anchor="end" font-size="11">{value:.1f}</text>',
        ]
    parts += [
        f'<text x="{width/2}" y="{height-15}" text-anchor="middle" font-size="13">{x_label}</text>',
        f'<text x="19" y="{height/2}" text-anchor="middle" font-size="13" transform="rotate(-90 19 {height/2})">{y_label}</text>',
    ]
    if ideal:
        parts.append(f'<path d="M{px(0)},{py(0)} L{px(1)},{py(1)}" stroke="#999" stroke-dasharray="5 5" fill="none"/>')
    values = [(x,y) for x,y in points if x is not None and y is not None]
    if values:
        data = " ".join(f"{px(x):.1f},{py(y):.1f}" for x,y in values)
        parts.append(f'<polyline points="{data}" fill="none" stroke="#0068b5" stroke-width="2.5"/>')
        parts.extend(f'<circle cx="{px(x):.1f}" cy="{py(y):.1f}" r="4" fill="#0068b5"/>'
                     for x,y in values)
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=str(ROOT / "config.yaml"))
    p.add_argument("--run-name", required=True)
    p.add_argument("--adapter", help="Adapter directory; omit for vanilla Qwen")
    p.add_argument("--limit", type=int)
    p.add_argument("--data", help="JSONL evaluation set; defaults to configured test")
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    out = ROOT / "evaluations" / args.run_name
    out.mkdir(parents=True, exist_ok=False)
    processor, model = load_model(cfg["model"]["path"], args.device)
    if args.adapter:
        model.model.language_model = PeftModel.from_pretrained(
            model.model.language_model, args.adapter, is_trainable=False
        )
        model.to(args.device)
    model.eval()
    with open(args.data or cfg["data"]["test"], encoding="utf-8") as f:
        data = [json.loads(line) for line in f if line.strip()]
    if args.limit:
        data = data[:args.limit]
    results = []
    with (out / "predictions.jsonl").open("w", encoding="utf-8") as f:
        for row in data:
            batch = to_device(
                prepare_video(processor, row, cfg["media"]["max_frames"], cfg["media"]["max_pixels"]),
                args.device,
            )
            if batch["input_ids"].shape[1] > cfg["media"]["max_prompt_tokens"]:
                raise ValueError(f"Prompt exceeds budget: {row['id']}")
            with torch.no_grad():
                completion, raw = generate_completion(
                    model, processor, batch,
                    max_new_tokens=cfg["evaluation"]["max_new_tokens"], sample=False
                )
                native = native_action_probs(model, processor, batch, len(row["choices"]))
            decision = parse_output(raw, len(row["choices"]))
            native_index = max(range(len(native)), key=native.__getitem__)
            result = {
                "id": row["id"], "video_path": row["video_path"],
                "prompt": prompt_text(row), "gold_action": chr(65+row["correct_choice"]),
                "predicted_action": chr(65+decision.action_index) if decision.valid_format else None,
                "correct": int(decision.valid_format and decision.action_index == row["correct_choice"]),
                "valid_format": decision.valid_format,
                "confidence_bin": decision.confidence_bin, "confidence": decision.confidence,
                "native_action": chr(65+native_index), "native_correct": int(native_index == row["correct_choice"]),
                "native_confidence": native[native_index],
                "native_probabilities": native,
                "raw_generation": raw,
                "prompt_tokens": int(batch["input_ids"].shape[1]),
            }
            results.append(result)
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
            f.flush()
            print(json.dumps({k: result[k] for k in ("id", "predicted_action", "correct", "confidence", "native_action")}), flush=True)
    summary = summarize(results)
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    table = summary["explicit"]["table"]
    svg_plot([(r["mean_confidence"], r["accuracy"]) for r in table if r["count"]],
             out / "reliability.svg", "Explicit confidence", "Mean confidence", "Empirical accuracy", ideal=True)
    curve = summary["explicit"]["risk_coverage"]
    svg_plot([(r["coverage_all"], r["selective_accuracy"]) for r in curve],
             out / "risk_coverage.svg", "Selective decisions", "Coverage", "Selective accuracy")
    print(json.dumps({k: v for k, v in summary.items() if k not in ("explicit", "native")}, indent=2))


if __name__ == "__main__":
    main()
