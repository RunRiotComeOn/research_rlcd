"""Compare confidence collapse and held-out calibration for advantage ablation."""
import json
import re
from pathlib import Path

ROOT = Path("/pfs/hyx/videojev-rlcd")
OUT = ROOT / "comparisons" / "advantage_ablation_200"
OUT.mkdir(parents=True, exist_ok=True)
PAT = re.compile(r"\AAction: ([A-Z])\nConfidence: ([0-9])\Z")
TAGS = ("std", "none")
STEPS = (25, 50, 75, 100, 200)


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


result = {"training": {}, "validation": {}}
for tag in TAGS:
    run = ROOT / "runs" / f"ablation_{tag}_200"
    records = rows(run / "metrics.jsonl")
    if len(records) != 200:
        raise ValueError(f"Expected 200 steps for {tag}; got {len(records)}")
    windows = []
    for start in range(0, 200, 25):
        chunk = records[start:start+25]
        bins = []
        for record in chunk:
            for raw in record.get("raw_generations", []):
                match = PAT.fullmatch(raw.strip())
                if match:
                    bins.append(int(match.group(2)))
        windows.append({
            "steps": [start, start+24],
            "valid_rollouts": len(bins),
            "fraction_confidence_9": sum(x == 9 for x in bins) / len(bins),
            "mean_confidence_bin": sum(bins) / len(bins),
            "skipped_updates": sum(bool(r.get("skipped_zero_advantage")) for r in chunk),
            "mean_rollout_correct_rate": sum(r.get("correct_rate", 0) for r in chunk) / 25,
        })
    result["training"][tag] = windows
    result["validation"][tag] = {}
    for step in STEPS:
        summary = ROOT / "evaluations" / f"ablation_{tag}_{step}_dev92" / "summary.json"
        if summary.is_file():
            data = json.loads(summary.read_text())
            result["validation"][tag][str(step)] = {
                "accuracy": data["accuracy"],
                "invalid_format_rate": data["invalid_format_rate"],
                "ece": data["explicit"]["ece"],
                "brier": data["explicit"]["brier"],
                "nll": data["explicit"]["nll"],
                "confidence_bin_counts": {
                    str(i): sum(x["count"] for x in data["explicit"]["table"] if x["lower"] <= 0.05+0.1*i < x["upper"])
                    for i in range(10)
                },
            }
(OUT / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
W, H = 720, 430
left, right, top, bottom = 80, 35, 45, 70
x0, x1, y0, y1 = left, W-right, top, H-bottom
px = lambda step: x0 + step / 200 * (x1-x0)
py = lambda frac: y1 - frac * (y1-y0)
parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">',
         '<rect width="100%" height="100%" fill="white"/>',
         '<text x="360" y="28" text-anchor="middle" font-size="18">Rollouts reporting confidence 9</text>',
         f'<path d="M{x0},{y0} V{y1} H{x1}" fill="none" stroke="#333" stroke-width="2"/>']
for tick in (0, 0.25, 0.5, 0.75, 1):
    parts.append(f'<text x="{x0-12}" y="{py(tick)+4:.1f}" text-anchor="end" font-size="12">{tick:.0%}</text>')
    parts.append(f'<line x1="{x0}" y1="{py(tick):.1f}" x2="{x1}" y2="{py(tick):.1f}" stroke="#ddd"/>')
for tick in (0, 50, 100, 150, 200):
    parts.append(f'<text x="{px(tick):.1f}" y="{y1+20}" text-anchor="middle" font-size="12">{tick}</text>')
parts.append(f'<text x="360" y="{H-18}" text-anchor="middle" font-size="13">Training step</text>')
for tag, color in (("std", "#c43c39"), ("none", "#126ca4")):
    points = " ".join(f'{px(w["steps"][1]+1):.1f},{py(w["fraction_confidence_9"]):.1f}' for w in result["training"][tag])
    parts.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="3"/>')
    parts.append(f'<text x="{x1-130}" y="{65 if tag=="std" else 88}" fill="{color}" font-size="13">{tag}</text>')
parts.append("</svg>")
(OUT / "confidence9_by_step.svg").write_text("\n".join(parts))
print(OUT)
