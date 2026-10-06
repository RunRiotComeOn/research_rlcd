"""Free space from finished full-JeV runs while retaining adapter checkpoints."""
import argparse
import json
from pathlib import Path


ROOT = Path('/pfs/hyx/videojev-rlcd').resolve()
RUNS = {
    'jevfull_action_fold0_v1': 24063,
    'jevfull_action_fold1_v1': 24063,
    'jevfull_action_all_v1': 48126,
    'jevfull_confidence_oof_v1': 48126,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    targets = []
    for name, final_step in RUNS.items():
        run = ROOT / 'runs' / name
        assert (run / 'STATUS.txt').read_text().strip() == f'COMPLETED {final_step}'
        final = run / 'checkpoints' / f'examples_{final_step:05d}'
        assert (final / 'adapter_model.safetensors').is_file()
        assert (final / 'optimizer.pt').is_file()
        for checkpoint in (run / 'checkpoints').glob('examples_*'):
            if checkpoint == final:
                continue
            adapter = checkpoint / 'adapter_model.safetensors'
            optimizer = checkpoint / 'optimizer.pt'
            if not optimizer.exists():
                continue
            assert adapter.is_file() and not adapter.is_symlink()
            assert optimizer.is_file() and not optimizer.is_symlink()
            assert optimizer.resolve().is_relative_to(ROOT)
            targets.append(optimizer)
    result = {'files': len(targets), 'bytes': sum(p.stat().st_size for p in targets),
              'paths': [str(p) for p in targets], 'executed': args.execute}
    if args.execute:
        for path in targets:
            path.unlink()
        (ROOT / 'logs/f0conf_optimizer_cleanup.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'files': result['files'], 'bytes': result['bytes'],
                      'executed': args.execute}, indent=2))


if __name__ == '__main__':
    main()
