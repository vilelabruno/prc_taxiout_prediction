import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def parse(stages, default):
    ap = argparse.ArgumentParser()
    ap.add_argument('stages', nargs='*', default=default, choices=list(stages) + ['all'])
    ap.add_argument('--threads', type=int, default=8)
    ap.add_argument('--jobs', type=int, default=3)
    ap.add_argument('--nproc', type=int, default=6)
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--dry-run', action='store_true')
    a = ap.parse_args()
    if 'all' in a.stages:
        a.stages = list(stages)
    return a
