"""Finite offline generated-code tests. Restricted AST/builtins, time-bound; not a hostile-code sandbox."""
import ast,bisect,collections,copy,itertools,json,math,pathlib,signal
ROOT=pathlib.Path(__file__).resolve().parent
ALLOWED={k:v for k,v in vars(__import__('builtins')).items() if k in ['range','len','abs','list','tuple','set','dict','enumerate','min','max','int','all','any','sum','sorted','zip','bool','reversed']}
NAMES={'dev_brackets':'balanced','heldout_prime':'is_prime','heldout_rle':'runs','heldout_dedup':'unique_in_order','new_lower':'lower_bound','new_merge':'merge_intervals','new_chunks':'chunks','new_moving':'moving_sums'}

def content(r):
    raw=r['raw_token_text'];prefix='<|channel>thought\n<channel|>'
    if raw.startswith(prefix):
        assert r['token_ids'][:4]==[100,45518,107,101];raw=raw[len(prefix):]
    if r['stopped']:
        for suffix in ['<turn|>','<eos>']:
            if raw.endswith(suffix):raw=raw[:-len(suffix)];break
    return raw.strip()

def balanced(s):
    n=0
    for c in s:
        n+=(c=='(')-(c==')')
        if n<0:return False
    return n==0

def unique(xs):
    out=[]
    for x in xs:
        if x not in out:out.append(x)
    return out

def merge(xs):
    # Independent connected-components oracle on interval intersection/touching.
    components=[]
    for a,b in xs:
        again=True
        while again:
            again=False;remaining=[]
            for c,d in components:
                if a<=d and c<=b:a,b=min(a,c),max(b,d);again=True
                else:remaining.append((c,d))
            components=remaining
        components.append((a,b))
    return sorted(components)

small=[list(x) for n in range(7) for x in itertools.product([-1,0,1],repeat=n)]
cases={
 'dev_brackets':[((s,),balanced(s)) for n in range(7) for x in itertools.product('()x',repeat=n) for s in [''.join(x)]],
 'heldout_prime':[((n,),n>=2 and all(n%d for d in range(2,math.isqrt(max(0,n))+1))) for n in list(range(-100,2001))+[7919,7920,99991,100000,100003]],
 'heldout_rle':[((xs,),[(k,len(list(g))) for k,g in itertools.groupby(xs)]) for xs in small+[[[1],[1],[2],[1]],[{'a':1},{'a':1},{'a':2}]]],
 'heldout_dedup':[((xs,),unique(xs)) for xs in small+[[[1],[2],[1]],[{'a':1},{'a':2},{'a':1}]]],
 'new_lower':[((xs,t),bisect.bisect_left(xs,t)) for n in range(8) for tup in itertools.combinations_with_replacement([-2,-1,0,1,2],n) for xs in [list(tup)] for t in range(-3,4)],
 'new_chunks':[((xs,k),[xs[i:i+k] for i in range(0,len(xs),k)]) for xs in small for k in [1,2,3,5,8]],
 'new_moving':[((xs,k),[sum(xs[i:i+k]) for i in range(len(xs)-k+1)]) for xs in small for k in [1,2,3,5,8]],
}
intervals=[(a,b) for a in range(-2,3) for b in range(a,3)]
cases['new_merge']=[((list(xs),),merge(xs)) for n in range(4) for xs in itertools.product(intervals,repeat=n)]
cases['new_merge'] += [(([list(t) for t in xs],),merge(xs)) for xs in [[],[(1,4),(2,3)],[(4,5),(0,1),(1,4)],[(1,1),(1,2),(8,10),(9,12)]]]

class CountingList(list):
    def __init__(self,x):super().__init__(x);self.reads=0
    def __getitem__(self,key):
        result=super().__getitem__(key);self.reads+=len(result) if isinstance(key,slice) else 1;return result
    def __iter__(self):
        for i in range(len(self)):yield self[i]

def check(text,task):
    if task=='heldout_json':
        def no_duplicates(pairs):
            result={}
            for key,value in pairs:
                if key in result:raise ValueError('duplicate JSON key')
                result[key]=value
            return result
        try:
            obj=json.loads(text,object_pairs_hook=no_duplicates)
            ok=(type(obj) is dict and set(obj)=={'project','retries','active','tags'} and type(obj['project']) is str and obj['project']=='willow' and type(obj['retries']) is int and obj['retries']==0 and type(obj['active']) is bool and obj['active'] is True and type(obj['tags']) is list and obj['tags']==['red','blue'])
        except ValueError:ok=False
        return dict(functional=ok,strict=ok)
    raw=text.strip();fenced=raw.startswith('```python\n') and raw.endswith('\n```')
    if fenced:raw=raw[10:-4]
    try:tree=ast.parse(raw)
    except SyntaxError:return dict(functional=False,strict=False,reason='syntax')
    if len(tree.body)!=1 or not isinstance(tree.body[0],ast.FunctionDef) or tree.body[0].name!=NAMES[task]:return dict(functional=False,strict=False,reason='required function missing or extra top-level code')
    for n in ast.walk(tree):
        if isinstance(n,(ast.Import,ast.ImportFrom,ast.Global,ast.Nonlocal,ast.ClassDef,ast.AsyncFunctionDef)):return dict(functional=False,strict=False,reason='outside reviewed AST subset')
        if isinstance(n,ast.FunctionDef) and (n.decorator_list or n.args.defaults or n.args.kw_defaults):return dict(functional=False,strict=False,reason='decorator/default not supported')
        if isinstance(n,ast.Attribute) and n.attr not in ['append','pop','add','extend','sort','copy','count','index']:return dict(functional=False,strict=False,reason='attribute outside subset')
        if isinstance(n,ast.Name) and n.id.startswith('__'):return dict(functional=False,strict=False,reason='dunder')
        if isinstance(n,ast.Call) and not (isinstance(n.func,ast.Name) and n.func.id in ALLOWED or isinstance(n.func,ast.Attribute)):return dict(functional=False,strict=False,reason='call outside subset')
    signal.alarm(5)
    try:
        env={'__builtins__':ALLOWED};exec(compile(tree,'<generated>','exec'),env);fn=env[NAMES[task]]
        for i,(args,want) in enumerate(cases[task]):
            original=copy.deepcopy(args);actual=copy.deepcopy(args);got=fn(*actual)
            if got!=want or task in ['dev_brackets','heldout_prime'] and type(got) is not bool or task=='new_merge' and (type(got) is not list or not all(type(v) is tuple for v in got)):
                return dict(functional=False,strict=False,reason='wrong answer/type',passed=i,total=len(cases[task]),input=original,want=want,got=got)
            if task.startswith('new_') and actual!=original:return dict(functional=False,strict=False,reason='input mutated',input=original)
        reads=None
        if task=='new_lower':
            xs=CountingList(list(range(8192)));assert fn(xs,8191)==8191;reads=xs.reads
            if reads>64:return dict(functional=False,strict=False,reason='linear work in binary-search probe',reads=reads)
        if task=='new_moving':
            xs=CountingList(list(range(2048)));got=fn(xs,1024);reads=xs.reads
            assert got==[sum(range(i,i+1024)) for i in range(1025)]
            if reads>20480:return dict(functional=False,strict=False,reason='superlinear work in moving-window probe',reads=reads)
        return dict(functional=True,strict=not fenced,raw_function_only=not fenced,passed=len(cases[task]),reads=reads)
    except Exception as exc:return dict(functional=False,strict=False,reason=repr(exc))
    finally:signal.alarm(0)

signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('5 second code test limit')))
# Regression checks: known bad solutions must fail and representative correct functions pass.
assert not check('{"project":"willow","retries":false,"active":true,"tags":["red","blue"]}','heldout_json')['strict']
assert not check('{"project":"willow","retries":0,"active":1,"tags":["red","blue"]}','heldout_json')['strict']
assert not check('{"project":"wrong","project":"willow","retries":0,"active":true,"tags":["red","blue"]}','heldout_json')['strict']
assert not check('def is_prime(n):\n return n>1','heldout_prime')['functional']
assert not check('def unique_in_order(xs):\n return [x for i,x in enumerate(xs) if i==0 or x!=xs[i-1]]','heldout_dedup')['functional']
assert check('def lower_bound(xs, target):\n lo=0\n hi=len(xs)\n while lo<hi:\n  mid=(lo+hi)//2\n  if xs[mid]<target:lo=mid+1\n  else:hi=mid\n return lo','new_lower')['functional']
assert check('def merge_intervals(intervals):\n out=[]\n for a,b in sorted(intervals,key=lambda x:x[0]):\n  if out and a<=out[-1][1]:out[-1]=(out[-1][0],max(out[-1][1],b))\n  else:out.append((a,b))\n return out','new_merge')['functional']
assert check('def chunks(xs,k):\n return [xs[i:i+k] for i in range(0,len(xs),k)]','new_chunks')['functional']
assert not check('def moving_sums(xs,k):\n return [sum(xs[i:i+k]) for i in range(len(xs)-k+1)]','new_moving')['functional']
rows=[]
for file in ['results.jsonl']:
    for line in (ROOT/file).read_text().splitlines():
        r=json.loads(line)
        if r['kind']!='trial':continue
        r['answer']=content(r);r['checks']=check(r['answer'],r['task'])
        r['answer_tokens']=len(r['token_ids'])-int(r['stopped'])-(4 if r['token_ids'][:4]==[100,45518,107,101] else 0)
        assert r['all_forward_calls']==len(r['calls'])
        # One packet per forward cut; feedback packet traverses both links unchanged.
        total_bytes=sum(e['bytes']*(2 if e['site']=='feedback' else 1) for e in r['wire_events'])
        encoder_calls=sum(c['phase']=='encoder' for c in r['calls'])
        total_bytes+=2*32*encoder_calls # modeled ACK32 bytes/link, not a measured packet
        r['modeled_total_network_bytes']=total_bytes if r['mode']!='native' else 0
        r['modeled_network_s']={str(mbps):2*.05*len(r['calls'])+total_bytes/(mbps*1e6/8) for mbps in [100,1000]} if r['mode']!='native' else None
        # Same-width, equal visible answer length proxy, not actual AR execution.
        g=r['answer_tokens'];p=r['prompt_tokens'];ar_bytes_link=(p+2*g+1)*5632
        r['modeled_ar_network_s']={str(mbps):2*.05*(g+1)+2*ar_bytes_link/(mbps*1e6/8) for mbps in [100,1000]}
        rows.append(r)
by=collections.defaultdict(list)
for r in rows:by[r['mode']].append(r)
summary={m:dict(trials=len(rs),functional=sum(r['checks']['functional'] for r in rs),strict=sum(r['checks']['strict'] for r in rs),calls=sum(r['all_forward_calls'] for r in rs),bytes=sum(r['modeled_total_network_bytes'] for r in rs),by_task={t:dict(functional=sum(r['checks']['functional'] for r in rs if r['task']==t),strict=sum(r['checks']['strict'] for r in rs if r['task']==t),calls=[r['all_forward_calls'] for r in rs if r['task']==t]) for t in NAMES.keys()|{'heldout_json'}}) for m,rs in by.items()}
lookup={(r['mode'],r['seed'],r['task']):r for r in rows}
assert all(lookup['native',r['seed'],r['task']]['token_ids']==r['token_ids'] and lookup['native',r['seed'],r['task']]['calls']==r['calls'] for r in by['17'])
old=ROOT.parent/'lowpass-search-20261117';matched=[]
for r in by['native']:
    if r['task'].startswith('new_'):continue
    if r['task']=='dev_brackets' and r['seed']!=42:continue
    suffix='' if r['seed']==42 else '-seed'+str(r['seed'])
    file=old/('gemma4-dev48.jsonl' if r['task']=='dev_brackets' else 'gemma4-heldout48'+suffix+'.jsonl')
    previous=next(x for l in file.read_text().splitlines() if (x:=json.loads(l)).get('task')==r['task'])
    matched.append(dict(task=r['task'],seed=r['seed'],exact_tokens=previous['token_ids']==r['token_ids'],exact_calls=previous['all_forward_calls']==r['all_forward_calls'],exact_prompt=previous['prompt_token_ids']==r['prompt_token_ids']))
assert all(x['exact_tokens'] and x['exact_calls'] and x['exact_prompt'] for x in matched)
retry={}
for mode in ['17','16','8','4']:
    retry[mode]={}
    for gate in ['functional','strict']:
        outcomes=[]
        for task in NAMES.keys()|{'heldout_json'}:
            attempts=[]
            for seed in [42,0,1,2]:
                r=lookup[mode,seed,task];attempts.append(r)
                if r['checks'][gate]:break
            outcomes.append(dict(task=task,success=attempts[-1]['checks'][gate],attempts=len(attempts),calls=sum(r['all_forward_calls'] for r in attempts),network_s={str(mbps):sum(r['modeled_network_s'][str(mbps)] for r in attempts) for mbps in [100,1000]}))
        retry[mode][gate]=outcomes
(ROOT/'validation.json').write_text(json.dumps(dict(summary=summary,native_historical_matches=matched,retry=retry,runs=rows),indent=2))
print(json.dumps(summary,indent=2));print('historical matched',len(matched))
