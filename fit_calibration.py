"""Fit frozen-action confidence baselines from binary VideoJev outcomes."""
import copy
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch import nn

from core import calibration_metrics

ROOT = Path("/pfs/hyx/videojev-rlcd")
OUT = ROOT / "calibration" / "confidence_v2"
OUT.mkdir(parents=True, exist_ok=True)
torch.set_num_threads(4)


def normalize_row(row):
    if "selected_native_p" not in row:
        idx = ord(row["predicted_action"]) - 65
        row = dict(row, selected_native_p=row["native_probabilities"][idx])
    if "explicit_confidence" not in row:
        row = dict(row, explicit_confidence=row["confidence"])
    return row


def feature(row):
    probs = np.asarray(row["native_probabilities"], dtype=np.float64)
    p = float(row["selected_native_p"])
    p = min(max(p, 1e-6), 1 - 1e-6)
    ranked = np.sort(probs)[::-1]
    entropy = -float(np.sum(probs * np.log(np.maximum(probs, 1e-12)))) / math.log(len(probs))
    return np.array([
        math.log(p / (1-p)), p, ranked[0], ranked[1],
        ranked[0] - p, ranked[0] - ranked[1],
        entropy, len(probs) / 6,
    ], dtype=np.float32)


def digit(q):
    return max(0, min(9, int(float(q) * 10)))


def auc(qs, ys):
    pos = [q for q, y in zip(qs, ys) if y]
    neg = [q for q, y in zip(qs, ys) if not y]
    return sum((a > b) + 0.5 * (a == b) for a in pos for b in neg) / (len(pos) * len(neg))


def metrics(qs, ys, discrete=False):
    if discrete:
        qs = [0.05 + 0.1 * digit(q) for q in qs]
    items = [{"confidence": float(q), "correct": int(y)} for q, y in zip(qs, ys)]
    result = calibration_metrics(items)
    result["rank_auc"] = auc(qs, ys)
    result["confidence_bins"] = dict(sorted(Counter(digit(q) for q in qs).items()))
    return {k: v for k, v in result.items() if k not in ("table", "risk_coverage")}


def evaluate(name, rows, platt, head, mu, scale, base_rate):
    x = np.stack([feature(r) for r in rows])
    ys = np.array([r["correct"] for r in rows], dtype=np.float32)
    with torch.no_grad():
        z = torch.tensor(x[:, 0], dtype=torch.float32)
        platt_q = torch.sigmoid(platt[0] * z + platt[1]).numpy()
        head_q = torch.sigmoid(head(torch.tensor((x-mu)/scale, dtype=torch.float32)).squeeze(-1)).numpy()
    methods = {
        "constant": np.full(len(rows), base_rate),
        "explicit": np.array([r["explicit_confidence"] for r in rows]),
        "selected_native": np.array([r["selected_native_p"] for r in rows]),
        "platt": platt_q,
        "head": head_q,
    }
    output = {
        "count": len(rows), "accuracy": float(ys.mean()),
        "methods": {
            method: {"continuous": metrics(q, ys), "discrete": metrics(q, ys, True)}
            for method, q in methods.items()
        },
    }
    path = OUT / f"{name}_predictions.jsonl"
    with path.open("w") as f:
        for i, row in enumerate(rows):
            item = {
                "id": row["id"], "correct": int(ys[i]),
                "predicted_action": row["predicted_action"],
                "explicit_q": float(methods["explicit"][i]),
                "selected_native_p": float(methods["selected_native"][i]),
                "platt_q": float(platt_q[i]), "platt_bin": digit(platt_q[i]),
                "head_q": float(head_q[i]), "head_bin": digit(head_q[i]),
                "platt_output": f'Action: {row["predicted_action"]}\nConfidence: {digit(platt_q[i])}',
                "head_output": f'Action: {row["predicted_action"]}\nConfidence: {digit(head_q[i])}',
            }
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    return output


def main():
    pool_path = ROOT / "calibration" / "pool_predictions.jsonl"
    rows = [normalize_row(json.loads(s)) for s in pool_path.read_text().splitlines() if s.strip()]
    if len(rows) != 1600:
        raise ValueError(f"Expected 1600 collected rows, got {len(rows)}")
    fit = [r for r in rows if r["split"] == "fit"]
    select = [r for r in rows if r["split"] == "select"]
    blind = [r for r in rows if r["split"] == "blind"]
    if [len(fit), len(select), len(blind)] != [1000, 300, 300]:
        raise ValueError("Calibration split count mismatch")
    x_fit = np.stack([feature(r) for r in fit])
    y_fit = torch.tensor([r["correct"] for r in fit], dtype=torch.float32)
    base_rate = float(y_fit.mean())
    x_sel = np.stack([feature(r) for r in select])
    y_sel = torch.tensor([r["correct"] for r in select], dtype=torch.float32)

    # Two-parameter Platt scaling on the probability of the chosen action.
    platt = nn.Parameter(torch.tensor([1.0, 0.0]))
    optimizer = torch.optim.LBFGS([platt], lr=1.0, max_iter=100, line_search_fn="strong_wolfe")
    z_fit = torch.tensor(x_fit[:, 0])
    def closure():
        optimizer.zero_grad()
        loss = nn.functional.binary_cross_entropy_with_logits(platt[0] * z_fit + platt[1], y_fit)
        loss = loss + 1e-4 * (platt[0] ** 2)
        loss.backward()
        return loss
    optimizer.step(closure)
    frozen_platt = platt.detach().clone()

    # Independent 8-feature confidence head. Action policy remains frozen.
    mu = x_fit.mean(axis=0)
    scale = np.maximum(x_fit.std(axis=0), 1e-5)
    train_x = torch.tensor((x_fit - mu)/scale, dtype=torch.float32)
    select_x = torch.tensor((x_sel - mu)/scale, dtype=torch.float32)
    torch.manual_seed(20261002)
    head = nn.Sequential(nn.Linear(8, 16), nn.Tanh(), nn.Linear(16, 1))
    opt = torch.optim.AdamW(head.parameters(), lr=0.005, weight_decay=0.001)
    best = None
    best_epoch = 0
    best_score = float("inf")
    patience = 0
    for epoch in range(500):
        head.train()
        opt.zero_grad()
        logits = head(train_x).squeeze(-1)
        loss = nn.functional.binary_cross_entropy_with_logits(logits, y_fit)
        loss.backward()
        opt.step()
        head.eval()
        with torch.no_grad():
            q = torch.sigmoid(head(select_x).squeeze(-1))
            score = float(((q - y_sel) ** 2).mean())
        if score < best_score - 1e-6:
            best_score = score
            best_epoch = epoch + 1
            best = copy.deepcopy(head.state_dict())
            patience = 0
        else:
            patience += 1
            if patience >= 50:
                break
    head.load_state_dict(best)
    head.eval()
    torch.save(head.state_dict(), OUT / "head_state.pt")
    (OUT / "parameters.json").write_text(json.dumps({
        "source_model": "runs/ablation_none_200/checkpoints/step_0200",
        "fit_count": 1000, "select_count": 300, "blind_count": 300,
        "fit_base_rate": base_rate,
        "platt_slope": float(frozen_platt[0]), "platt_intercept": float(frozen_platt[1]),
        "feature_mean": mu.tolist(), "feature_scale": scale.tolist(),
        "head_hidden": 16, "head_best_epoch": best_epoch,
        "head_select_continuous_brier": best_score,
    }, indent=2) + "\n")

    results = {}
    for name, group in (("fit", fit), ("select", select), ("blind", blind)):
        results[name] = evaluate(name, group, frozen_platt, head, mu, scale, base_rate)
    for name, path in (
        ("historical_dev92", ROOT / "evaluations/ablation_none_200_dev92/predictions.jsonl"),
        ("holmes92", ROOT / "evaluations/ablation_none_200_holmes92/predictions.jsonl"),
    ):
        group = [normalize_row(json.loads(s)) for s in path.read_text().splitlines() if s.strip()]
        results[name] = evaluate(name, group, frozen_platt, head, mu, scale, base_rate)
    (OUT / "summary.json").write_text(json.dumps(results, indent=2) + "\n")
    (OUT / "STATUS.txt").write_text("COMPLETE\n")
    for name, data in results.items():
        print(name, "accuracy", round(data["accuracy"], 3),
              {method: round(item["discrete"]["brier"], 3) for method, item in data["methods"].items()})


if __name__ == "__main__":
    main()
