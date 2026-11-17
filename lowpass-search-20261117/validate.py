# Only run on manually reviewed inputs; AST checks are NOT a security sandbox.
"""Offline task checks; protocol framing is not answer text. No inference required."""
import ast
import itertools
import json
import math
import pathlib
import signal

ROOT=pathlib.Path(__file__).resolve().parent

def content(r):
    raw=r.get('raw_token_text')
    if raw is None:return r['text'],False
    prefix='<|channel>thought\n<channel|>'
    removed=raw.startswith(prefix)
    if removed:raw=raw[len(prefix):]
    if r['stopped']:
        if r.get('family')=='llada20' and raw.endswith('<|role_end|><|endoftext|>'):
            assert r['token_ids'][-2:]==[156900,156892]
            return raw[:-len('<|role_end|><|endoftext|>')],removed
        for end in ['<turn|>','<eos>','<|role_end|>','<|endoftext|>']:
            if raw.endswith(end):raw=raw[:-len(end)];break
    return raw,removed

def function_check(text,task):
    raw=text.strip();fenced=raw.startswith('```python\n') and raw.endswith('\n```')
    if fenced:raw=raw[10:-4]
    try:tree=ast.parse(raw)
    except SyntaxError:return {'functional':False,'syntax':False,'raw_function_only':False}
    name={'dev_brackets':'balanced','heldout_prime':'is_prime','heldout_rle':'runs','heldout_dedup':'unique_in_order'}[task]
    if len(tree.body)!=1 or not isinstance(tree.body[0],ast.FunctionDef) or tree.body[0].name!=name:
        return {'functional':False,'reason':'not exactly the required function'}
    allowed={'range':range,'len':len,'abs':abs,'list':list,'set':set,'enumerate':enumerate,'min':min,'max':max,'int':int,'all':all,'any':any}
    for n in ast.walk(tree):
        if isinstance(n,(ast.Import,ast.ImportFrom,ast.Global,ast.Nonlocal,ast.Lambda,ast.ClassDef)):return {'functional':False,'reason':'outside reviewed subset'}
        if isinstance(n,ast.Attribute) and n.attr not in ['append','pop','add']:return {'functional':False,'reason':'attribute outside reviewed subset'}
        if isinstance(n,ast.Name) and n.id.startswith('__'):return {'functional':False,'reason':'dunder'}
        if isinstance(n,ast.Call) and not (isinstance(n.func,ast.Name) and n.func.id in allowed or isinstance(n.func,ast.Attribute)):
            return {'functional':False,'reason':'call outside reviewed subset'}
    if task=='dev_brackets':
        def oracle(xs):
            b=0
            for x in xs:
                b+=(x=='(')-(x==')')
                if b<0:return False
            return b==0
        cases=[''.join(x) for n in range(7) for x in itertools.product('()x',repeat=n)]
    elif task=='heldout_prime':
        def oracle(n):return n>=2 and all(n%d for d in range(2,math.isqrt(max(0,n))+1))
        cases=list(range(-100,2001))+[7919,7920,99991,100000,100003]
    elif task=='heldout_rle':
        def oracle(xs):return [(k,len(list(g))) for k,g in itertools.groupby(xs)]
        cases=[list(x) for n in range(7) for x in itertools.product([0,1,2],repeat=n)]+[[[1],[1],[2],[1]],[{'a':1},{'a':1},{'a':2}]]
    else:
        def oracle(xs):
            out=[]
            for x in xs:
                if x not in out:out.append(x)
            return out
        cases=[list(x) for n in range(7) for x in itertools.product([0,1,2],repeat=n)]+[[[1],[2],[1]],[{'a':1},{'a':2},{'a':1}]]
    signal.alarm(3)
    try:
        env={'__builtins__':allowed};exec(compile(tree,'<checked-generated-function>','exec'),env)
        for i,x in enumerate(cases):
            got=env[name](x);want=oracle(x)
            if got!=want or task in ['dev_brackets','heldout_prime'] and type(got) is not bool:
                return {'functional':False,'passed':i,'total':len(cases),'input':x,'got':got,'want':want,'raw_function_only':not fenced}
        return {'functional':True,'passed':len(cases),'total':len(cases),'raw_function_only':not fenced}
    except Exception as exc:return {'functional':False,'error':repr(exc)}
    finally:signal.alarm(0)

signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('bounded function validation')))
assert function_check('def balanced(s):\n return True','dev_brackets')['functional'] is False
assert function_check('def is_prime(n):\n return n>1','heldout_prime')['functional'] is False
assert function_check('def unique_in_order(xs):\n out=[]\n for x in xs:\n  if x not in out:out.append(x)\n return out','heldout_dedup')['functional']
assert content({'raw_token_text':'<|channel>thought\n<channel|>{}<eos>','stopped':True})==('{}',True)
assert content({'family':'llada20','raw_token_text':'{}<|role_end|><|endoftext|>','stopped':True,'token_ids':[156900,156892]})==('{}',False)
assert content({'family':'llada20','raw_token_text':'```python\nx=1\n```<|role_end|><|endoftext|>','stopped':True,'token_ids':[156900,156892]})==('```python\nx=1\n```',False)
rows=[]
for f in sorted(ROOT.glob('*.jsonl')):
    for line,l in enumerate(f.read_text().splitlines(),1):
        r=json.loads(l)
        if r['kind']!='trial':continue
        assert len(r['calls'])==r['all_forward_calls']
        assert len(r['prompt_token_ids'])==r['prompt_tokens']
        assert len(r['token_ids'])==r['visible_tokens']+r['stopped']
        text,protocol=content(r);task=r['task'];checks={}
        if protocol: assert r['token_ids'][:4]==[100,45518,107,101]
        llada_role_end=r['family']=='llada20' and r['stopped'] and r['token_ids'][-2:]==[156900,156892]
        answer_tokens=r['visible_tokens']-(4 if protocol else 0)-(1 if llada_role_end else 0)
        if task in ['dev_brackets','heldout_prime','heldout_rle','heldout_dedup']:checks=function_check(text,task)
        elif task in ['dev_extract','heldout_json']:
            want={'name':'cedar','count':17,'enabled':False} if task=='dev_extract' else {'project':'willow','retries':0,'active':True,'tags':['red','blue']}
            try:checks={'exact_json':json.loads(text)==want}
            except ValueError:checks={'exact_json':False}
        else:
            checks={'words_whitespace':len(text.split()),'manual_content_check_required':True}
            if task in ['dev_prose','heldout_grounded']:checks['requested_length']=180<=len(text.split())<=240
            else:checks['requested_length']=len(text.split())<=(60 if task=='dev_travel' else 70)
        if r['family']=='gemma':
            decoder=sum(c['phase']=='decoder' for c in r['calls']);encoder=sum(c['phase']=='encoder' for c in r['calls'])
            assert decoder==int(r['stats']['diffusion_denoising_steps'])
            assert encoder+decoder==r['all_forward_calls']
        else:encoder=None;decoder=None
        rows.append(dict(file=f.name,line=line,task=task,family=r['family'],settings=r['settings'],all_forward_calls=r['all_forward_calls'],encoder_calls=encoder,decoder_calls=decoder,visible_tokens=r['visible_tokens'],answer_tokens=answer_tokens,wall_s=r['wall_s'],stopped=r['stopped'],unresolved_masks=r['unresolved_masks'],empty_protocol_thought_removed=protocol,text=text,checks=checks))
assert rows, 'No trial records found; supply reviewed experiment results first'
(ROOT/'validation.json').write_text(json.dumps({'total':len(rows),'runs':rows},indent=2))
print('validated',len(rows))
for r in rows:
    if 'heldout' in r['file'] or r['file']=='gemma4-dev48.jsonl':print(r['file'],r['task'],r['all_forward_calls'],r['checks'])
