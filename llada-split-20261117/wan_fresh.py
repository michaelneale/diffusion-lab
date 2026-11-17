#!/usr/bin/env python3
"""Matched fresh prompts and metadata checks for original vs distilled weights."""
import hashlib
import json
import signal
import struct
import sys
from pathlib import Path
import canvas_experiment as ce
from probe32_experiment import LocalExperiment, TASKS
from wan_sweep import generate, checks

TASKS_NEW = {
 'boxes': 'A warehouse has 48 boxes. It ships three quarters of them, then receives 18 more boxes. It divides all boxes now present equally between 5 trucks. How many boxes go on each truck? Show the arithmetic and final answer in at most 60 words.',
 'discount': 'A customer buys 3 notebooks at $17 each and 2 pens at $9 each. A 20% discount applies to the whole purchase. They pay with $100. What change should they receive? Show the arithmetic in at most 60 words.',
 'travel': 'A train travels 120 km at 60 km/h, stops for 15 minutes, then travels 45 km at 90 km/h. What is the total elapsed time in minutes? Show the arithmetic in at most 60 words.',
 'dedup': TASKS['code'],
 'brackets': 'Write Python function balanced(s) that returns True exactly when round parentheses in s are balanced. Ignore all other characters. A closing parenthesis before its matching opener is invalid. Return only the function, with no examples or prose.',
 'gcd': 'Write Python function gcd(a, b) using the Euclidean algorithm. Handle negative inputs and return 0 for gcd(0, 0). No imports. Return only the function, no examples or prose.',
}


def metadata(path):
    with open(path,'rb') as f:
        def number(fmt):
            size=struct.calcsize('<'+fmt)
            data=f.read(size)
            if len(data)!=size:
                raise ValueError('truncated GGUF')
            return struct.unpack('<'+fmt,data)[0]
        def string():
            size=number('Q')
            assert size<16*1024*1024
            return f.read(size).decode('utf-8')
        def value(t):
            if t==8:
                return string()
            if t==9:
                subtype,n=number('I'),number('Q')
                assert n<2000000
                return [value(subtype) for _ in range(n)]
            return number({0:'B',1:'b',2:'H',3:'h',4:'I',5:'i',6:'f',7:'?',10:'Q',11:'q',12:'d'}[t])
        assert f.read(4)==b'GGUF' and number('I')==3
        tensors,nkv=number('Q'),number('Q')
        assert nkv<10000
        result={}
        for _ in range(nkv):
            key=string()
            result[key]=value(number('I'))
        result['_tensor_count']=tensors
        return result


def validate_metadata():
    original=metadata(ce.ROOT/'models/LLaDA-8B-Instruct.Q4_K_M.gguf')
    distilled=metadata(ce.ROOT/'models/d3LLM_LLaDA.Q4_K_M.gguf')
    keys=[k for k in original if k.startswith(('tokenizer.', 'llada.'))]
    same={k: original[k]==distilled.get(k) for k in keys}
    assert original['tokenizer.ggml.tokens']==distilled['tokenizer.ggml.tokens']
    tokens=distilled['tokenizer.ggml.tokens']
    for i,expected in [(126080,'<|startoftext|>'),(126081,'<|endoftext|>'),(126336,'<|mdm_mask|>'),
                       (126346,'<|start_header_id|>'),(126347,'<|end_header_id|>'),(126348,'<|eot_id|>')]:
        assert tokens[i]==expected,(i,tokens[i],expected)
    out={'matching_metadata':same,
         'architectures':[x['general.architecture'] for x in (original,distilled)],
         'token_counts':[len(x['tokenizer.ggml.tokens']) for x in (original,distilled)],
         'token_array_sha256':hashlib.sha256(json.dumps(tokens).encode()).hexdigest(),
         'tensor_counts':[x['_tensor_count'] for x in (original,distilled)]}
    (ce.ROOT/'wan-metadata.json').write_text(json.dumps(out,indent=2))
    print(json.dumps(out),flush=True)


def main():
    checks()
    if sys.argv[1]=='metadata':
        validate_metadata()
        return
    model=sys.argv[1]
    assert model in ('original','d3')
    if model=='d3':
        ce.MODEL='models/d3LLM_LLaDA.Q4_K_M.gguf'
    e=LocalExperiment(ce.ROOT/f'wan-fresh-{model}.json')
    e.results.update(model=model,task_text=TASKS_NEW,prompt_token_ids={},decoder='greedy cfg0 no-cache block128 threshold.8 immutable-prefix stop')
    try:
        for name,text in TASKS_NEW.items():
            prompt=e.tokenize(ce.template([('user',text)]))
            e.results['prompt_token_ids'][name]=prompt
            if model=='d3':
                baseline=json.loads((ce.ROOT/'wan-fresh-original.json').read_text())
                assert baseline['prompt_token_ids'][name]==prompt
            generate(e,name,prompt,256,128,.8)
    finally:
        e.close()

if __name__=='__main__':
    signal.signal(signal.SIGALRM,lambda *_: (_ for _ in ()).throw(TimeoutError('bounded fresh suite')))
    signal.alarm(600)
    main()
