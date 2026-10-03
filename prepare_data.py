"""Create deterministic 5% VideoJev train/test subsets inside this project."""
import argparse
import hashlib
import json
import random
import re
from pathlib import Path

TRAIN_SOURCE = Path("/pfs/qcy/JeV-Data/video_multiple_choice_clean_48126.json")
TEST_SOURCE = Path("/pfs/qcy/JeV-Data/holmes.json")
ROOT = Path("/pfs/hyx/videojev-rlcd")
ANSWER = re.compile(r"<answer>\s*([A-Z])\s*</answer>", re.I)
OPTION = re.compile(r"^\s*[A-Z][.、:)\s-]+", re.I)


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def convert(row, split):
    options = [OPTION.sub("", x).strip() for x in row["options"]]
    match = ANSWER.fullmatch((row.get("answer") or row.get("solution") or "").strip())
    if not match or not 2 <= len(options) <= 26:
        raise ValueError(f"Bad answer/options for {row.get('problem_id')}")
    gold = ord(match.group(1).upper()) - 65
    if gold >= len(options) or gold < 0 or any(not x for x in options):
        raise ValueError(f"Answer outside options for {row.get('problem_id')}")
    video_path = (row.get("videos") or [row.get("path")])[0]
    if not video_path or not Path(video_path).is_file():
        raise FileNotFoundError(video_path)
    cache = row.get("preprocessed_video")
    if cache and not Path(cache).is_file():
        raise FileNotFoundError(cache)
    return {
        "id": f"{split}-{row['problem_id']}",
        "question": row["problem"].replace("<video>", "").strip(),
        "choices": options,
        "correct_choice": gold,
        "video_path": video_path,
        "frame_cache_path": cache,
        "data_source": row.get("data_source"),
        "reasoning_effort": row.get("reasoning_effort"),
    }


def write_jsonl(path, rows):
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=20261001)
    parser.add_argument("--fraction", type=float, default=0.05)
    args = parser.parse_args()
    if not 0 < args.fraction <= 1:
        parser.error("fraction must be in (0, 1]")
    out = ROOT / "data"
    out.mkdir(parents=True, exist_ok=True)
    sources = {
        "train": (TRAIN_SOURCE, json.loads(TRAIN_SOURCE.read_text(encoding="utf-8"))),
        "test": (TEST_SOURCE, json.loads(TEST_SOURCE.read_text(encoding="utf-8"))),
    }
    result = {}
    selected = {}
    for split, (path, raw) in sources.items():
        count = round(len(raw) * args.fraction)
        rng = random.Random(args.seed + (0 if split == "train" else 1))
        positions = sorted(rng.sample(range(len(raw)), count))
        rows = [convert(raw[i], split) for i in positions]
        if len({r["id"] for r in rows}) != len(rows):
            raise ValueError(f"Duplicate IDs in {split}")
        selected[split] = rows
        result[split] = {
            "source": str(path), "source_sha256": sha256(path),
            "source_rows": len(raw), "selected_rows": len(rows),
            "output": str(out / f"{split}.jsonl"),
            "selection_seed": args.seed + (0 if split == "train" else 1),
        }
    train_paths = {Path(r["video_path"]).name for r in selected["train"]}
    test_paths = {Path(r["video_path"]).name for r in selected["test"]}
    overlap = train_paths & test_paths
    if overlap:
        raise ValueError(f"Train/test video filename overlap: {sorted(overlap)[:5]}")
    for split, rows in selected.items():
        write_jsonl(out / f"{split}.jsonl", rows)
        result[split]["output_sha256"] = sha256(out / f"{split}.jsonl")
    result["fraction"] = args.fraction
    result["cross_split_video_filename_overlap"] = 0
    (out / "manifest.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({k: v["selected_rows"] for k, v in result.items() if k in ("train", "test")}))


if __name__ == "__main__":
    main()
