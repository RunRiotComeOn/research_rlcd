"""Evaluate all prespecified full-run checkpoints on JeV validation."""
import argparse
import subprocess
import sys
from pathlib import Path

ROOT=Path('/pfs/hyx/videojev-rlcd')
STEPS=(500,1000,2000,3000,4000,4813)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--mode',choices=('ce','ce_brier'),required=True)
    p.add_argument('--device',required=True)
    args=p.parse_args()
    for step in STEPS:
        run=f'jev10_decision_context_full_{args.mode}_v1'
        checkpoint=f'runs/{run}/checkpoints/step_{step:04d}'
        name=f'full_{args.mode}_step_{step:04d}_validation'
        subprocess.run([sys.executable,str(ROOT/'eval_jev10_action.py'),
                        '--split','validation','--name',name,
                        '--adapter',checkpoint,'--device',args.device],
                       cwd=ROOT,check=True)
    print(f'COMPLETED route {args.mode}',flush=True)


if __name__=='__main__': main()
