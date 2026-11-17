#!/usr/bin/env python3
"""Local-only bounded decoder probes; reuses unchanged native worker and decoder."""
import argparse
import json
import signal
from canvas_experiment import Experiment, Worker, ROOT, template, selftest

TASKS = {
    'extract': 'Return only a JSON object with keys name, count, enabled. Facts: the name is cedar, the count is 17, and enabled is false. Do not add other keys or prose.',
    'reason': 'A shop starts with 23 apples, sells 7, receives 12, then packs all remaining apples equally into 4 boxes. How many apples are in each box? Show the arithmetic briefly and give the final answer.',
    'code': 'Write a Python function unique_in_order(xs) that returns a list with duplicate values removed while preserving first occurrence order. Inputs can include unhashable lists. Give the function and exactly two assert examples, including an example with nested lists. No prose.',
}

class LocalExperiment(Experiment):
    def __init__(self, output):
        self.workers = []
        self.output = output
        self.results = {'decoder': 'native Q4_K_M, argmax, cfg0, block32, no cache',
                        'transport': 'local full model only', 'runs': [], 'parity': [], 'tasks': TASKS}
        self.request_id = 0
        self.full = Worker('full')
        self.workers.append(self.full)


def prefix_completion(result):
    """Replay immutable commitments; end token and every preceding slot must be set."""
    end = result['visible_tokens_before_end']
    required = set(range(result['prompt_tokens'], result['prompt_tokens'] + min(end+1, result['canvas_tokens'])))
    elapsed = 0
    for n, record in enumerate(result['passes'], 1):
        required.difference_update(record['selected_positions'])
        elapsed += record['wall_s']
        if not required:
            return {'prefix_complete_pass': n, 'prefix_eval_wall_s': elapsed,
                    'prefix_tokens_per_cycle': end/n}
    raise AssertionError('completion invariant')


def checks(result):
    text = result['text'].strip()
    task = result['name'].split('-')[0]
    if task == 'extract':
        try:
            return {'exact_json': json.loads(text) == {'name': 'cedar', 'count': 17, 'enabled': False}}
        except ValueError:
            return {'exact_json': False}
    return {'manual_review_required': True}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('suite', choices=['main', 'incident', 'short', 'math-threshold'])
    args = parser.parse_args()
    selftest()
    fake = {'visible_tokens_before_end': 1, 'canvas_tokens': 4, 'prompt_tokens': 2,
            'passes': [{'selected_positions': [3], 'wall_s': 1}, {'selected_positions': [2], 'wall_s': 1}]}
    assert prefix_completion(fake)['prefix_complete_pass'] == 2
    e = LocalExperiment(ROOT/f'probe32-{args.suite}-results.json')
    try:
        if args.suite == 'incident':
            prompt = json.loads((ROOT/'incident_prompt.json').read_text())['prompt_token_ids']
            jobs = [('incident', prompt, 256, 256, .6), ('incident', prompt, 256, 256, .4)]
        elif args.suite == 'math-threshold':
            prompt = e.tokenize(template([('user', TASKS['reason'])]))
            jobs = [('reason', prompt, 256, 256, .4), ('reason', prompt, 256, 256, .2)]
        else:
            size = 256 if args.suite == 'main' else 64
            policies = [(256, None), (32, None), (256, .8), (256, .6)] if args.suite == 'main' else [(64, None), (32, None)]
            jobs = [(name, e.tokenize(template([('user', text)])), size, steps, threshold)
                    for name, text in TASKS.items() for steps, threshold in policies]
        for name, prompt, size, steps, threshold in jobs:
            label = f'{name}-canvas{size}-' + (f'threshold{threshold}' if threshold is not None else f'fixed{steps}')
            r = e.generate(label, prompt, size, steps, False, 32, threshold)
            r.update(prefix_completion(r))
            r['quality_checks'] = checks(r)
            e.save()
            print('SUMMARY', json.dumps({k:v for k,v in r.items() if k not in ('passes', 'generated_token_ids')}), flush=True)
    finally:
        e.close()

if __name__ == '__main__':
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError('bounded probe')))
    signal.alarm(900)
    main()
