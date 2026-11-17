#!/usr/bin/env python3
import json
import signal
import sys
import canvas_experiment as ce
from probe32_experiment import LocalExperiment
from wan_sweep import generate, checks
from wan_fresh import TASKS_NEW


def main():
    checks()
    model,suite=sys.argv[1:]
    assert model in ('original','d3') and suite in ('repair','list100','canvas','policy')
    if model=='d3':
        ce.MODEL='models/d3LLM_LLaDA.Q4_K_M.gguf'
    e=LocalExperiment(ce.ROOT/f'wan-follow-{suite}-{model}.json')
    e.results.update(model=model,decoder='greedy cfg0 no-cache; actual immutable-prefix stopping')
    try:
        if suite=='repair':
            for task in ('boxes','discount','travel','gcd'):
                prompt=e.tokenize(ce.template([('user',TASKS_NEW[task])]))
                policies=[('one',.8)] if model=='original' else [('one',.8),('threshold',.8),('threshold',.95)]
                for policy,threshold in policies:
                    generate(e,task,prompt,256,32,threshold,policy)
        elif suite=='list100':
            text=('Write exactly 100 lines, no heading, no fences, no explanation. '
                  'For each integer 1 through 100, write the line "Item N: status pending review." '
                  'replacing N with that integer. Include all 100 items in order.')
            e.results['task_text']=text
            prompt=e.tokenize(ce.template([('user',text)]))
            for block in (32,128):
                r=generate(e,'list100',prompt,1024,block,.8)
                expected='\n'.join(f'Item {i}: status pending review.' for i in range(1,101))
                r['exact_line_check']=r['text'].strip()==expected
                e.save()
        elif suite=='canvas':
            from probe32_experiment import TASKS
            prompt=e.tokenize(ce.template([('user',TASKS['reason'])]))
            for size in (128,256,512):
                generate(e,'reason',prompt,size,128,.8)
        else:
            prompt=e.tokenize(ce.template([('user',TASKS_NEW['discount'])]))
            for budget in (.1,.5):
                generate(e,'discount',prompt,256,128,budget,'error_budget')
    finally:
        e.close()

if __name__=='__main__':
    signal.signal(signal.SIGALRM,lambda *_: (_ for _ in ()).throw(TimeoutError('bounded follow-up')))
    signal.alarm(900)
    main()
