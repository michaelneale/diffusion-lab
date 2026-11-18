#!/usr/bin/env python3
"""Fair WAN test: d3LLM-LLaDA split head/tail, prefix stop, fixed settings.

Every (task, threshold) is reported, so a solve rate per setting can be read
directly; no per-task selection. The `--local` reference runs the same settings
on the unsplit full model to separate compute from transport.
"""
import argparse, json, os, sys, time
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / 'llada-split-20261117')]
import canvas_experiment as ce
ce.MODEL = 'models/d3LLM_LLaDA.Q4_K_M.gguf'
from wan_sweep import generate
from fair_tasks import TASKS, CANVAS, check

class Split(ce.Experiment):
    split = True
    def evaluate(self, tokens, prompt_len, split):
        return super().evaluate(tokens, prompt_len, self.split)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--thresholds', default='.8,.6,.4')
    ap.add_argument('--local-thresholds', default='.6')
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    e = Split(Path(a.out))
    e.codec = 'f16'
    e.results.update(model='d3LLM_LLaDA Q4_K_M sha256 eb450d47', codec='f16', stop='immutable-prefix',
                     planner_identity_note='canvas_worker planner labels the plan with the original LLaDA sha; d3 shares its architecture')
    try:
        for split, ths in [(True, a.thresholds), (False, a.local_thresholds)]:
            e.split = split
            for th in map(float, ths.split(',')):
                for task, (group, text) in TASKS.items():
                    size = CANVAS.get(task, 128)
                    prompt = e.tokenize(ce.template([('user', text)]))
                    try:
                        r = generate(e, f'{task}-{"split" if split else "local"}-{th}', prompt, size, min(128, size), th)
                    except Exception as err:  # a failed generation is a result, not a crash
                        r = {'name': task, 'error': repr(err)}; e.results['runs'].append(r)
                    r.update(task=task, group=group, split=split, threshold=th, correct=check(task, r.get('text', '')))
                    e.save()
                    print('FAIR', json.dumps({k: r.get(k) for k in ('task','group','split','threshold','steps','elapsed_s','visible_tokens','correct','error')}), flush=True)
    finally:
        e.close()

if __name__ == '__main__':
    main()
