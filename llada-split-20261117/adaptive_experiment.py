#!/usr/bin/env python3
"""External lab experiment; no product cache changes or fixed pass quota."""
import argparse
import json
import signal
from canvas_experiment import Experiment, ROOT, selftest
from probe32_experiment import LocalExperiment

parser = argparse.ArgumentParser()
parser.add_argument('--split', action='store_true')
parser.add_argument('--thresholds', default='.95,.9,.8')
args = parser.parse_args()
thresholds = [float(t) for t in args.thresholds.split(',')]
assert all(0 <= t <= 1 for t in thresholds)
signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError('bounded adaptive experiment')))
signal.alarm(900)
selftest()
prompt = json.loads((ROOT/'incident_prompt.json').read_text())['prompt_token_ids']
name = 'split' if args.split else 'local'
e = (Experiment if args.split else LocalExperiment)(ROOT/f'canvas-adaptive-{name}-results.json')
e.codec = 'f16'
e.results.update(codec='f16' if args.split else 'f32', prompt_token_ids=prompt,
                 decoder='argmax, cfg0, block32, threshold acceptance with top1 fallback, no cache')
try:
    for threshold in thresholds:
        e.generate(f'incident-{name}-threshold{threshold}', prompt, 256, 256, args.split, 32, threshold)
finally:
    e.close()
