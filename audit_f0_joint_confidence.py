"""Independently audit the completed frozen-F0 joint confidence experiment."""

import hashlib
import json
from itertools import combinations
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score


ROOT = Path("/pfs/hyx/videojev-rlcd")
DATA = ROOT / "data/f0_joint_confidence_v1"
EVAL = ROOT / "evaluations/f0_joint_confidence_v1"
FOLD0 = ROOT / "data/jev_full_holmes_v1/folds/fold0.jsonl"
HOLMES = ROOT / "data/jev_full_holmes_v1/holmes_all.jsonl"
HOLMES_ACTIONS = ROOT / "evaluations/f0_confidence_ablation_v1/holmes_actions/predictions.jsonl"
STEPS = (4000, 8000, 12000, 16000, 16063)


def read_jsonl(path):
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream]


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def video_names(rows):
    return {Path(row["video_path"]).name for row in rows}


def ece(q, y):
    bins = np.minimum((q * 10).astype(int), 9)
    return float(sum(np.mean(bins == i) * abs(float(q[bins == i].mean() - y[bins == i].mean()))
                     for i in range(10) if np.any(bins == i)))


def check_metric(q, y, metric):
    assert abs(float(np.mean((q - y) ** 2)) - metric["brier"]) < 1e-10
    assert abs(float(roc_auc_score(y, q)) - metric["auc"]) < 1e-10
    assert abs(ece(q, y) - metric["ece_10_equal_width"]) < 1e-10


def main():
    manifest = json.loads((DATA / "manifest.json").read_text())
    selection = json.loads((EVAL / "selection_grid.json").read_text())
    final = json.loads((EVAL / "final_summary_grid.json").read_text())
    sources = {split: read_jsonl(DATA / f"{split}.jsonl") for split in ("train", "dev", "test")}
    sources["holmes"] = read_jsonl(HOLMES)
    sources["fold0"] = read_jsonl(FOLD0)
    actions = {split: read_jsonl(DATA / f"{split}_actions.jsonl") for split in ("train", "dev", "test")}
    actions["holmes"] = read_jsonl(HOLMES_ACTIONS)
    expected = {"train": 16063, "dev": 2000, "test": 2000, "holmes": 1837, "fold0": 24063}
    assert {split: len(rows) for split, rows in sources.items()} == expected
    assert sha256(Path(manifest["source"])) == manifest["source_sha256"]
    assert sha256(Path(manifest["source_actions"])) == manifest["source_actions_sha256"]

    for split in ("train", "dev", "test"):
        item = manifest["splits"][split]
        assert len(sources[split]) == item["rows"]
        assert len(video_names(sources[split])) == item["videos"]
        assert sha256(DATA / f"{split}.jsonl") == item["data_sha256"]
        assert sha256(DATA / f"{split}_actions.jsonl") == item["actions_sha256"]
    for split in actions:
        assert [r["id"] for r in sources[split]] == [r["id"] for r in actions[split]]
        assert all(r.get("source_answer_fold") == "fold0" for r in actions[split])
    for left, right in combinations(sources, 2):
        assert not video_names(sources[left]) & video_names(sources[right]), (left, right)
        assert not {r["id"] for r in sources[left]} & {r["id"] for r in sources[right]}, (left, right)

    assert len(selection["runs"]) == 6
    eligible_count = 0
    for run, item in selection["runs"].items():
        assert [r["step"] for r in item["trials"]] == list(STEPS)
        for trial in item["trials"]:
            folder = EVAL / "dev" / run / f'examples_{trial["step"]:05d}'
            assert (folder / "STATUS.txt").read_text().strip() == "COMPLETED 2000"
            summary = json.loads((folder / "summary.json").read_text())
            assert summary["rows"] == 2000
            assert abs(summary["direct_mean_q"]["brier"] - trial["brier"]) < 1e-12
            assert abs(summary["action_agreement_f0"] - trial["action_agreement_f0"]) < 1e-12
            eligible = summary["action_agreement_f0"] >= 0.99 and summary["action_accuracy"] >= summary["f0_action_accuracy"]
            assert eligible == trial["eligible"]
            eligible_count += eligible
        pool = [r for r in item["trials"] if r["eligible"]] or item["trials"]
        assert item["selected_step"] == min(pool, key=lambda r: (r["brier"], r["step"]))["step"]
    assert eligible_count == 0
    chosen = selection["global_selection"]
    assert chosen["run"] == "lam1" and chosen["step"] == 16000 and not chosen["eligible"]
    assert final["selection"] == chosen

    result = {"rows": expected,
              "videos": {split: len(video_names(rows)) for split, rows in sources.items()},
              "pairwise_video_filename_overlap": 0,
              "pairwise_question_id_overlap": 0,
              "development_checkpoints": 30,
              "eligible_checkpoints": eligible_count,
              "diagnostic_selection": {"run": chosen["run"], "step": chosen["step"]},
              "final": {}}
    for split in ("test", "holmes"):
        folder = EVAL / split / chosen["run"] / f'examples_{chosen["step"]:05d}'
        assert (folder / "STATUS.txt").read_text().strip() == f'COMPLETED {expected[split]}'
        predictions = read_jsonl(folder / "predictions.jsonl")
        reference = actions[split]
        assert [r["id"] for r in predictions] == [r["id"] for r in sources[split]]
        assert len(predictions) == expected[split]
        assert all(r["action_index"] == int(np.argmax(r["action_probabilities"])) for r in predictions)
        assert all(r["correct"] == int(r["action_index"] == r["gold_action_index"]) for r in predictions)
        assert all(r["f0_action_index"] == a["action_index"] and r["f0_correct"] == a["correct"]
                   for r, a in zip(predictions, reference))
        assert all(abs(sum(r["digit_probabilities"]) - 1) < 1e-4 for r in predictions)
        assert all(abs(r["mean_q"] - sum((i + .5) / 10 * p for i, p in enumerate(r["digit_probabilities"]))) < 1e-5
                   for r in predictions)
        y = np.array([r["correct"] for r in predictions], dtype=float)
        f0_y = np.array([r["f0_correct"] for r in predictions], dtype=float)
        q = np.array([r["mean_q"] for r in predictions], dtype=float)
        argmax_q = np.array([r["confidence_bin_q"] for r in predictions], dtype=float)
        raw_p = np.array([r["selected_action_p"] for r in predictions], dtype=float)
        metrics = final["results"][split]
        check_metric(q, y, metrics["joint_direct_mean_q"])
        check_metric(argmax_q, y, metrics["joint_direct_argmax_digit_q"])
        check_metric(raw_p, y, metrics["joint_direct_selected_action_p"])
        assert abs(float(y.mean()) - metrics["joint_accuracy"]) < 1e-12
        assert abs(float(f0_y.mean()) - metrics["f0_accuracy"]) < 1e-12
        assert abs(float(np.mean([r["action_index"] == a["action_index"] for r, a in zip(predictions, reference)]))
                   - metrics["action_agreement_f0"]) < 1e-12
        assert abs(float(y.mean() * (1 - y.mean())) - metrics["joint_constant_oracle_brier"]) < 1e-12
        result["final"][split] = {"rows": len(predictions), "videos": len(video_names(predictions)),
                                   "action_agreement_f0": metrics["action_agreement_f0"],
                                   "joint_accuracy": metrics["joint_accuracy"],
                                   "direct_q_brier": metrics["joint_direct_mean_q"]["brier"]}
    (EVAL / "audit_grid.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
