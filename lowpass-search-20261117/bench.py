"""Read-only model harness: count every sequential backbone evaluation, retain failures."""
import argparse
import gc
import hashlib
import importlib.metadata
import inspect
import json
import pathlib
import signal
import sys
import time

ROOT=pathlib.Path(__file__).resolve().parent
TASKS=json.loads((ROOT/'tasks.json').read_text())
p=argparse.ArgumentParser()
p.add_argument('--model',required=True)
p.add_argument('--family',choices=['llada22','llada20','gemma'],required=True)
p.add_argument('--suite',choices=['smoke','dev','heldout'],default='smoke')
p.add_argument('--steps',type=int,default=32)
p.add_argument('--block',type=int,default=32)
p.add_argument('--post',type=int,default=16)
p.add_argument('--threshold',type=float,default=.5)
p.add_argument('--entropy',type=float,default=.1)
p.add_argument('--maximum',type=int,default=512)
p.add_argument('--seed',type=int,default=42)
p.add_argument('--tasks',default='')
a=p.parse_args()
if a.family=='llada22':
    sys.path.insert(0,a.model)
    import llada22_mlx
import mlx.core as mx
from mlx_vlm import load

calls=[]
patches=[]

def emit(x): print(json.dumps(x,ensure_ascii=False,default=str),flush=True)

def instrument(target,phase):
    cls=type(target)
    original=cls.__call__
    def wrapped(self,inputs,*args,**kwargs):
        if self is target:
            if len(calls)>=512: raise RuntimeError('experiment safety bound:512 backbone calls')
            c=kwargs.get('cache')
            caller=inspect.currentframe().f_back.f_code.co_name
            calls.append(dict(phase=phase,rows=int(inputs.shape[1]),prefix=int(c[0].offset) if c and hasattr(c[0],'offset') else None,caller=caller))
        return original(self,inputs,*args,**kwargs)
    cls.__call__=wrapped
    patches.append((cls,original))
    return dict(class_name=cls.__name__,source=inspect.getsourcefile(original),source_sha256=hashlib.sha256(pathlib.Path(inspect.getsourcefile(original)).read_bytes()).hexdigest())

start=time.perf_counter()
model,processor=load(a.model,lazy=False,trust_remote_code=(a.family=='llada22'))
mx.synchronize()
tok=getattr(processor,'tokenizer',processor)
load_time=time.perf_counter()-start
if a.family=='gemma':
    sources=[instrument(model.model.encoder,'encoder'),instrument(model.model.decoder,'decoder')]
else:
    sources=[instrument(model.language_model.model,'backbone')]
if a.family=='gemma':
    # Use intended entropy sampler; change only its scalar budget for explicit sweeps.
    cfg=dict(model.config.generation_config or {})
    cfg['sampler_config']={'_cls_name':'EntropyBoundSamplerConfig','entropy_bound':a.entropy}
    model.config.generation_config=cfg
emit(dict(kind='environment',args=vars(a),load_s=load_time,device=mx.device_info(),packages={x:importlib.metadata.version(x) for x in ['mlx','mlx-metal','mlx-vlm','transformers']},sources=sources,task_file_sha256=hashlib.sha256((ROOT/'tasks.json').read_bytes()).hexdigest(),tokenizer_probe=tok.encode('This is a simple test.',add_special_tokens=False),generation_config=getattr(model.config,'generation_config',None)))

def trial(task):
    prompt=tok.apply_chat_template([{'role':'user','content':TASKS[task]}],tokenize=False,add_generation_prompt=True,enable_thinking=False)
    ids=tok.encode(prompt,add_special_tokens=False)
    inputs=mx.array([ids],dtype=mx.int32)
    mx.eval(inputs)
    gc.collect();mx.synchronize();mx.clear_cache();mx.reset_peak_memory();mx.random.seed(a.seed)
    calls.clear();stats={};events=[]
    start=time.perf_counter()
    def callback(r):
        if r.text and not r.is_draft: events.append(dict(s=time.perf_counter()-start,tokens=r.generation_tokens,text=r.text))
        return True
    kwargs=dict(temperature=0.,gen_length=a.maximum,steps=a.steps,tokenizer=tok,stats=stats,skip_special_tokens=True,on_result=callback,eos_early_stop=True)
    if a.family=='gemma':
        kwargs.update(processor=processor,diffusion_sampler='entropy-bound',diffusion_max_canvas_length=256)
    else:
        kwargs.update(block_length=a.block,threshold=a.threshold,max_post_steps=a.post,editing_threshold=0. if a.family=='llada22' else None,max_steps_per_block=128)
    signal.alarm(180)
    try:
        out=model.generate(inputs,**kwargs) if a.family=='gemma' else model.language_model.generate(inputs,**kwargs)
        mx.eval(out);mx.synchronize()
        elapsed=time.perf_counter()-start
        generated=out[0].tolist()
        eos=getattr(model.config,'eos_token_id',None)
        if eos is None and a.family=='gemma':eos=(model.config.generation_config or {}).get('eos_token_id',model.config.text_config.eos_token_id)
        eos=set(eos if isinstance(eos,list) else [eos])
        stopped=bool(generated and generated[-1] in eos)
        # Tokenizer may carry additional turn-ending IDs in the Gemma implementation.
        if a.family=='gemma' and generated:stopped=bool(tok.stopping_criteria(generated[-1]))
        visible=generated[:-1] if stopped else generated
        text=tok.decode(visible,skip_special_tokens=True)
        mask_id=getattr(model.config,'mask_token_id',None)
        row=dict(kind='trial',task=task,family=a.family,settings=vars(a),prompt_token_ids=ids,prompt_tokens=len(ids),token_ids=generated,visible_tokens=len(visible),stopped=stopped,unresolved_masks=sum(x==mask_id for x in visible),text=text,raw_token_text=tok.decode(generated,skip_special_tokens=False),wall_s=elapsed,all_forward_calls=len(calls),calls=list(calls),stats=stats,events=events,metal_peak_bytes=mx.get_peak_memory())
        if a.family=='gemma':
            assert sum(c['phase']=='decoder' for c in calls)==int(stats['diffusion_denoising_steps'])
        else:
            assert all(c['rows']==a.block for c in calls)
        emit(row)
    except Exception as exc:
        emit(dict(kind='error',task=task,settings=vars(a),error=repr(exc),calls=list(calls),wall_s=time.perf_counter()-start))
        raise
    finally:signal.alarm(0)

signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('180s trial budget')))
try:
    tasks=a.tasks.split(',') if a.tasks else (['dev_extract'] if a.suite=='smoke' else [x for x in TASKS if x.startswith('dev_' if a.suite=='dev' else 'heldout_')])
    for task in tasks:trial(task)
finally:
    for cls,original in reversed(patches):cls.__call__=original
