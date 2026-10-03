import json
from collections import Counter, defaultdict
from pathlib import Path

source = Path('/pfs/qcy/JeV-Data/video_multiple_choice_clean_48126.json')
old = Path('/pfs/hyx/videojev-rlcd/data/train.jsonl')
rows = json.loads(source.read_text())
old_rows = [json.loads(s) for s in old.read_text().splitlines()]
by_path = defaultdict(list)
for row in rows:
    path = (row.get('videos') or [row.get('path')])[0]
    by_path[path].append(row)
old_ids = {int(r['id'].split('-', 1)[1]) for r in old_rows}
old_paths = {r['video_path'] for r in old_rows}
expanded = sum(len(by_path[p]) for p in old_paths)
print(json.dumps({
    'source_rows': len(rows),
    'source_unique_ids': len({r['problem_id'] for r in rows}),
    'source_unique_video_paths': len(by_path),
    'video_group_size_counts': Counter(len(v) for v in by_path.values()).most_common(12),
    'old_rows': len(old_rows),
    'old_ids_in_source': sum(r['problem_id'] in old_ids for r in rows),
    'old_paths': len(old_paths),
    'old_paths_expanded_rows': expanded,
}, indent=2))
