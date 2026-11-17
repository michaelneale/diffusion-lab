# Only run on manually reviewed inputs; AST checks are NOT a security sandbox.
"""Offline evidence checks. Generated functions limited to the prior reviewed AST subset."""
import json
import pathlib
import statistics
import sys
import signal

ROOT=pathlib.Path(__file__).resolve().parent
from function_checks import function_check

signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError('function validation limit')))
rows=[]
for path in sorted(ROOT.glob('*.jsonl')):
    for line_no,line in enumerate(path.read_text().splitlines(),1):
        r=json.loads(line)
        if r['kind']!='trial': continue
        calls=r['calls']
        assert len(calls)==r['all_forward_calls']
        assert sum(c['rows'] for c in calls)==r['all_rows']
        assert r['prompt_tokens']==len(r['prompt_token_ids'])==calls[0]['rows']
        assert r['visible_tokens']==len(r['token_ids'])-int(r['stopped'])
        assert r['stopped']==(r['token_ids'][-1]==11)
        if r['mode']=='diffusion':
            assert len(calls)==1+int(r['stats']['diffusion_total_nfe'])
            assert sum(not c['causal'] for c in calls)==int(r['stats']['diffusion_denoise_nfe'])
            assert sum(c['cache_write'] for c in calls)==int(r['stats']['diffusion_blocks'])
            assert r['stats']['diffusion_mixed_ar_forwards']==0
            # Each block's denoise runs are separated by causal cache/anchor calls.
            blocks=[]
            for c in calls:
                if c['causal']: blocks.append(0)
                else: blocks[-1]+=1
            capped_blocks=sum(x==min(r['steps_cap_per_block'],r['block']) for x in blocks)
        else:
            assert all(c['causal'] and c['cache_write'] for c in calls)
            assert len(calls)==len(r['token_ids'])+(not r['stopped'])
            capped_blocks=0
        checks={}
        if r['task'] in ('brackets','gcd','dedup'):
            signal.alarm(2)
            try: checks=function_check(r['text'],r['task'])
            except TimeoutError:checks={'functional':False,'timeout':True}
            finally:signal.alarm(0)
            checks['raw_function_only']=r['text'].strip().startswith('def ') and '```' not in r['text']
        if r['task']=='extract':
            try:checks={'exact_json':json.loads(r['text'])==dict(name='cedar',count=17,enabled=False)}
            except ValueError:checks={'exact_json':False}
        if r['task'].startswith('list'):
            n=int(r['task'][4:]); want='\n'.join(f'Item {i}: status pending review.' for i in range(1,n+1))
            checks={'exact_lines':r['text'].strip()==want,'lines_ignoring_trailing_spaces':'\n'.join(x.rstrip() for x in r['text'].strip().splitlines())==want}
        if r['task']=='prose':
            words=len(r['text'].split())
            checks={'whitespace_words':words,'requested_word_range':180<=words<=240,'content_requires_manual_review':True}
        if r['task'] in ['travel','boxes','discount']:
            checks={'whitespace_words':len(r['text'].split()),'at_most_60_words':len(r['text'].split())<=60,'math_requires_manual_review':True}
        out={k:r[k] for k in ['task','mode','block','threshold','steps_cap_per_block','synthetic_context','repeat','all_forward_calls','all_rows','visible_tokens','stopped','wall_s','metal_peak_bytes']}
        out.update(file=path.name,line=line_no,checks=checks,capped_blocks=capped_blocks)
        rows.append(out)
assert rows, 'No trial records found; supply reviewed experiment results first'
(ROOT/'validation.json').write_text(json.dumps({'total':len(rows),'runs':rows},indent=2))
print('Validated',len(rows),'generations')
for host in ['m5','mini']:
    for suite in ['machines','selected']:
        print(host,suite)
        xs=[r for r in rows if r['file']==f'{host}-{suite}.jsonl' and r['task']!='extract']
        for task in sorted(set(r['task'] for r in xs)):
            for mode in ['diffusion','ar']:
                ys=[r for r in xs if r['task']==task and r['mode']==mode]
                print(task,mode,'median_s',round(statistics.median(y['wall_s'] for y in ys),4),'calls',[y['all_forward_calls'] for y in ys],'peak_GB',round(max(y['metal_peak_bytes'] for y in ys)/1e9,3),'check',ys[0]['checks'])
print('Successful code:')
for r in rows:
    if r['checks'].get('functional') and r['file'] not in ['m5-machines.jsonl','mini-machines.jsonl','m5-selected.jsonl','mini-selected.jsonl']:
        print(r['file'],r['task'],r['mode'],r['block'],r['threshold'],r['all_forward_calls'],r['checks'])
