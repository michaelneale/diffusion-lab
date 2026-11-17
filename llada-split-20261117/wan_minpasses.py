#!/usr/bin/env python3
import json
import signal
import sys
import canvas_experiment as ce
from probe32_experiment import LocalExperiment, TASKS
from wan_fresh import TASKS_NEW
from wan_sweep import generate, checks


def main():
    checks()
    model=sys.argv[1]
    assert model in ('d3','original')
    if model=='d3':
        ce.MODEL='models/d3LLM_LLaDA.Q4_K_M.gguf'
    short=len(sys.argv)>2 and sys.argv[2]=='short'
    suffix='-short' if short else ''
    e=LocalExperiment(ce.ROOT/f'wan-minpasses-{model}{suffix}.json')
    e.results.update(model=model,decoder='greedy cfg0 no-cache; actual immutable-prefix stop; minimize passes subject to quality')
    try:
        jobs=[('reason',TASKS['reason']),('travel',TASKS_NEW['travel']),('brackets',TASKS_NEW['brackets'])]
        for task,text in jobs:
            prompt=e.tokenize(ce.template([('user',text)]))
            for size in ((64,128) if short else (128,256)):
                for threshold in ((.8,.9) if short else (.6,.4,.2)):
                    generate(e,task,prompt,size,min(128,size),threshold)
    finally:
        e.close()

if __name__=='__main__':
    signal.signal(signal.SIGALRM,lambda *_: (_ for _ in ()).throw(TimeoutError('bounded minpasses sweep')))
    signal.alarm(600)
    main()
