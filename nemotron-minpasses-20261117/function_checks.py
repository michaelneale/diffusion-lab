# Only run on manually reviewed inputs; AST checks are NOT a security sandbox.
# Exact function_check snapshot from llada-split-20261117/wan_validate.py
import ast
import itertools
import math

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
