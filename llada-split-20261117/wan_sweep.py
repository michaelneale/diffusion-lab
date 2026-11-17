#!/usr/bin/env python3
"""Pure diffusion local sweeps; frozen native worker, actual immutable-prefix stop."""
import argparse
import json
import signal
import time
from canvas_experiment import MASK, SPECIAL_END, ROOT, template, choose_updates, adaptive_count, selftest
from probe32_experiment import LocalExperiment, TASKS


def complete_prefix(xs):
    """Only committed end with every earlier slot committed permits a stop."""
    for i, token in enumerate(xs):
        if token == MASK:
            return None
        if token in SPECIAL_END:
            return i
    return len(xs) if MASK not in xs else None


def count_updates(tokens, p, confidence, lo, hi, threshold, policy):
    if policy == 'one':
        return 1
    if policy == 'threshold':
        return adaptive_count(tokens, p, confidence, lo, hi, threshold)
    # A bounded joint-error budget using marginal confidences (union bound).
    # Not calibrated probabilities or a correctness guarantee; experimental heuristic.
    ranked = sorted((confidence[i-p] for i in range(lo, hi) if tokens[i] == MASK), reverse=True)
    count, error = 0, 0.
    for prob in ranked:
        if error + 1-prob > threshold:
            break
        count += 1
        error += 1-prob
    return max(1, count)


def generate(e, name, prompt, size, block, threshold=.8, policy='threshold', stop=True):
    assert size % block == 0 and policy in ('threshold', 'one', 'error_budget')
    tokens = prompt + [MASK]*size
    records = []
    p = len(prompt)
    started = time.monotonic()
    end = None
    for offset in range(0, size, block):
        lo, hi = p+offset, p+offset+block
        while MASK in tokens[lo:hi]:
            response, stats = e.evaluate(tokens, p, False)
            n = count_updates(tokens, p, response['confidence'], lo, hi, threshold, policy)
            selected = choose_updates(tokens, p, response['ids'], response['confidence'], n, lo, hi)
            stats.update(selected_positions=selected, committed=n,
                         selected_tokens=[tokens[i] for i in selected],
                         selected_confidence=[response['confidence'][i-p] for i in selected])
            records.append(stats)
            end = complete_prefix(tokens[p:])
            if stop and end is not None:
                break
        if stop and end is not None:
            break
    elapsed = time.monotonic()-started
    assert tokens[:p] == prompt
    end = complete_prefix(tokens[p:])
    assert end is not None and MASK not in tokens[p:p+end]
    result = dict(name=name, prompt_tokens=p, canvas_tokens=size, block=block, threshold=threshold,
                  policy=policy, early_stop=stop, steps=len(records), elapsed_s=elapsed,
                  visible_tokens=end, text=e.detokenize(tokens[p:p+end]),
                  generated_token_ids=tokens[p:], passes=records,
                  unfilled_after_end=tokens[p+end:].count(MASK))
    result['tokens_per_cycle'] = end/len(records)
    e.results['runs'].append(result)
    e.save()
    print(json.dumps({k:v for k,v in result.items() if k not in ('passes', 'generated_token_ids')}), flush=True)
    return result


def checks():
    selftest()
    eos = min(SPECIAL_END)
    assert complete_prefix([MASK,eos]) is None
    assert complete_prefix([3,eos,MASK]) == 1
    assert complete_prefix([3,4]) == 2
    assert complete_prefix([3,MASK]) is None
    assert complete_prefix([eos,MASK]) == 0
    assert count_updates([1,MASK,MASK,MASK],1,[.99,.9,.5],1,4,.12,'error_budget') == 2
    assert count_updates([1,MASK,MASK],1,[.2,.3],1,3,.01,'error_budget') == 1
    assert count_updates([1,5,MASK],1,[1,.9],1,3,.2,'error_budget') == 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('suite', choices=['selftest','blocks','policy','long','controls'])
    parser.add_argument('--task', default='all')
    parser.add_argument('--model', choices=['original','d3'], default='original')
    args = parser.parse_args()
    if args.model == 'd3':
        import canvas_experiment
        canvas_experiment.MODEL = 'models/d3LLM_LLaDA.Q4_K_M.gguf'
    checks()
    if args.suite == 'selftest':
        print('all helper selfchecks passed')
        return
    suffix = '-d3' if args.model == 'd3' else ''
    e = LocalExperiment(ROOT/f'wan-{args.suite}-{args.task}{suffix}.json')
    e.results['model'] = args.model
    e.results['decoder'] = 'LLaDA architecture Q4_K_M; greedy cfg0; no cache; block/policy/actual-prefix-stop sweep'
    try:
        if args.suite == 'blocks':
            incident = json.loads((ROOT/'incident_prompt.json').read_text())['prompt_token_ids']
            prompts = {name:e.tokenize(template([('user', TASKS[name])])) for name in ('extract','reason')}
            prompts['incident'] = incident
            for task,prompt in prompts.items():
                if args.task not in ('all',task):
                    continue
                for block in (16,32,64,128,256):
                    generate(e,task,prompt,256,block)
        elif args.suite == 'policy':
            prompt = (json.loads((ROOT/'incident_prompt.json').read_text())['prompt_token_ids']
                      if args.task == 'incident' else e.tokenize(template([('user',TASKS[args.task])])))
            for block in (64,256):
                for policy,value in [('threshold',.6),('threshold',.9),('error_budget',.1),('error_budget',.5)]:
                    generate(e,args.task,prompt,256,block,value,policy)
        elif args.suite == 'controls':
            prompt = e.tokenize(template([('user', TASKS['extract'])]))
            for policy in ('one','threshold'):
                a=generate(e,'stop-off',prompt,256,32,.8,policy,False)
                b=generate(e,'stop-on',prompt,256,32,.8,policy,True)
                assert a['text'] == b['text']
                assert a['generated_token_ids'][:a['visible_tokens']+1] == b['generated_token_ids'][:b['visible_tokens']+1]
            # Baseline comparison uses the SAME early stopping criterion.
            for task in ('reason','incident'):
                prompt=(e.tokenize(template([('user',TASKS[task])])) if task=='reason' else
                        json.loads((ROOT/'incident_prompt.json').read_text())['prompt_token_ids'])
                generate(e,task,prompt,256,32,.8,'one')
        else:
            # Mechanistic easy-output test, not evidence of general reasoning ability.
            task = ('Write exactly 32 lines, no heading, no fences, no explanation. '
                    'For each integer 1 through 32, write the line "Item N: status pending review." '
                    'replacing N with that integer. Include all 32 items in order.')
            prompt=e.tokenize(template([('user',task)]))
            e.results['task_text']=task
            for block in (32,128,512):
                r=generate(e,'list32',prompt,512,block,.8)
                expected='\n'.join(f'Item {i}: status pending review.' for i in range(1,33))
                r['exact_line_check']=r['text'].strip()==expected
                e.save()
    finally:
        e.close()

if __name__ == '__main__':
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError('bounded sweep')))
    signal.alarm(900)
    main()
