"""Materialize the complete JeV pool and complete Holmes test in /pfs/hyx."""
import hashlib
import json
from pathlib import Path

from prepare_data import TRAIN_SOURCE, TEST_SOURCE, ROOT, convert, write_jsonl

OUT = ROOT / 'data/jev_full_holmes_v1'


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    sets = {}
    for name, source in [('jev_all', TRAIN_SOURCE), ('holmes_all', TEST_SOURCE)]:
        raw = json.loads(source.read_text(encoding='utf-8'))
        if name == 'jev_all':
            bad = [row for row in raw if str(row['problem_id']) == '9020007652']
            if len(bad) != 1 or bad[0]['options'][1] != 'B. ).':
                raise ValueError('Expected single known malformed JeV option')
            bad[0]['options'][1] = 'B. [missing option text]'
        rows = [convert(row, name) for row in raw]
        ids = {r['id'].split('-', 1)[1] for r in rows}
        if len(ids) != len(rows):
            raise ValueError(f'Duplicate {name} problem IDs')
        sets[name] = {'source': source, 'raw_count': len(raw), 'rows': rows,
                      'ids': ids, 'paths': {r['video_path'] for r in rows},
                      'names': {Path(r['video_path']).name for r in rows}}
    train, test = sets['jev_all'], sets['holmes_all']
    if train['ids'] & test['ids'] or train['paths'] & test['paths'] or train['names'] & test['names']:
        raise ValueError('JeV/Holmes question or video overlap')
    OUT.mkdir(parents=True)
    manifest = {'local_repair': {'problem_id': '9020007652',
                                 'field': 'options[1]', 'old': 'B. ).',
                                 'new': 'B. [missing option text]',
                                 'reason': 'The original distractor contains no usable text'},
                'cross_source_problem_id_overlap': 0, 'cross_source_video_path_overlap': 0,
                'cross_source_video_filename_overlap': 0, 'sets': {}}
    for name, data in sets.items():
        output = OUT / f'{name}.jsonl'
        write_jsonl(output, data['rows'])
        manifest['sets'][name] = {'source': str(data['source']),
                                  'source_sha256': sha256(data['source']),
                                  'rows': len(data['rows']),
                                  'video_paths': len(data['paths']),
                                  'video_filenames': len(data['names']),
                                  'output': str(output), 'output_sha256': sha256(output)}
    (OUT / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({name: data['raw_count'] for name, data in sets.items()}))


if __name__ == '__main__':
    main()
