#!/usr/bin/env python3
# Only run on manually reviewed inputs; AST checks are NOT a security sandbox.
"""Offline checks: reviewed generated functions only; bounded exhaustive tiny inputs."""
import ast
import itertools
import json
import math
import signal
from canvas_experiment import ROOT
from wan_sweep import complete_prefix, checks


def function_check(text,task):
    text=text.strip()
    if text.startswith('```python\n') and text.endswith('\n```'):
        text=text[10:-4]
    try:
        tree=ast.parse(text)
    except SyntaxError:
        return {'syntax':False,'functional':False}
    functions=[n for n in tree.body if isinstance(n,ast.FunctionDef)]
    expected={'brackets':'balanced','gcd':'gcd','dedup':'unique_in_order'}[task]
    if len(functions)!=1 or functions[0].name!=expected:
        return {'syntax':True,'functional':False,'reason':'missing/extra function'}
    tree=ast.Module(body=functions,type_ignores=[])
    # Reviewed code uses only arithmetic, iteration, local containers and these calls.
    for node in ast.walk(tree):
        if isinstance(node,(ast.Import,ast.ImportFrom,ast.Global,ast.Nonlocal,ast.Lambda,ast.ClassDef)):
            return {'functional':False,'reason':'outside reviewed code subset'}
        if isinstance(node,ast.Attribute) and node.attr not in ('append','pop','add'):
            return {'functional':False,'reason':'outside reviewed attribute subset'}
        if isinstance(node,ast.Call):
            if isinstance(node.func,ast.Name) and node.func.id not in ('set','list','abs','len'):
                return {'functional':False,'reason':'outside reviewed call subset'}
            if not isinstance(node.func,(ast.Name,ast.Attribute)):
                return {'functional':False,'reason':'indirect call'}
        if isinstance(node,ast.Name) and node.id.startswith('__'):
            return {'functional':False,'reason':'dunder access'}
    namespace={'__builtins__':{'set':set,'list':list,'abs':abs,'len':len}}
    exec(compile(tree,'<reviewed-function>','exec'),namespace)
    fn=namespace[expected]
    if task=='brackets':
        def oracle(s):
            balance=0
            for ch in s:
                balance+=(ch=='(')-(ch==')')
                if balance<0:return False
            return balance==0
        cases=[(''.join(xs),) for n in range(7) for xs in itertools.product('()x',repeat=n)]
        expected_values=[oracle(c[0]) for c in cases]
    elif task=='gcd':
        cases=list(itertools.product(range(-15,16),repeat=2))
        expected_values=[math.gcd(*c) for c in cases]
    else:
        cases=[([],),([2,1,2],),([[1],[2],[1]],),([{'x':1},{'x':1},{'x':2}],)]
        expected_values=[[],[2,1],[[1],[2]],[{'x':1},{'x':2}]]
    for i,(case,want) in enumerate(zip(cases,expected_values)):
        try:got=fn(*case)
        except Exception as exc:
            return {'syntax':True,'functional':False,'passed':i,'total':len(cases),'error':type(exc).__name__,'input':case}
        if got!=want or (task=='brackets' and type(got) is not bool):
            return {'syntax':True,'functional':False,'passed':i,'total':len(cases),'input':case,'got':got,'expected':want}
    return {'syntax':True,'functional':True,'passed':len(cases),'total':len(cases)}


def main():
    checks()
    assert function_check('def balanced(s):\n return True','brackets')['functional'] is False
    assert function_check('def gcd(a,b):\n return abs(a) if b==0 else 0','gcd')['functional'] is False
    assert function_check('def unique_in_order(xs):\n out=[]\n for x in xs:\n  if x not in out: out.append(x)\n return out','dedup')['functional']
    outputs=[]
    for path in sorted(ROOT.glob('wan-*.json')):
        if path.name == 'wan-validation.json':
            continue
        blob=json.loads(path.read_text())
        for r in blob.get('runs',[]):
            generated=r['generated_token_ids']
            assert complete_prefix(generated)==r['visible_tokens']
            assert len(r['passes'])==r['steps']
            seen=set()
            for record in r['passes']:
                positions=record['selected_positions']
                assert not seen.intersection(positions)
                assert len(positions)==record['committed']
                assert all(r['prompt_tokens']<=p<r['prompt_tokens']+r['canvas_tokens'] for p in positions)
                seen.update(positions)
            check={}
            if r['name'] in ('brackets','gcd','dedup'):
                check=function_check(r['text'],r['name'])
            if r['name'].startswith('list'):
                n=int(r['name'][4:])
                expected='\n'.join(f'Item {i}: status pending review.' for i in range(1,n+1))
                check={'exact_lines':r['text'].strip()==expected}
                if not check['exact_lines']:
                    actual=r['text'].strip().splitlines()
                    check['mismatched_lines']=[{'line':i+1,'got':a,'expected':b} for i,(a,b) in enumerate(itertools.zip_longest(actual,expected.splitlines())) if a!=b]
                assert r.get('exact_line_check')==check['exact_lines']
            outputs.append({'file':path.name,**{k:r[k] for k in ('name','canvas_tokens','block','policy','threshold','steps','visible_tokens','elapsed_s')},'checks':check})
    assert outputs, 'No trial records found; supply reviewed experiment results first'
    (ROOT/'wan-validation.json').write_text(json.dumps({'runs':outputs,'total_generations':len(outputs)},indent=2))
    print('generations',len(outputs))
    for r in outputs:
        if r['checks']:
            print(json.dumps(r))

if __name__=='__main__':
    signal.signal(signal.SIGALRM,lambda *_: (_ for _ in ()).throw(TimeoutError('bounded offline validation')))
    signal.alarm(45)
    main()
