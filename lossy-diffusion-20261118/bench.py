"""Local serialized activation perturbation; this does not distribute model execution."""
import argparse,gc,hashlib,importlib.metadata,inspect,json,pathlib,signal,time
import numpy as np
import mlx.core as mx
import mlx_vlm
from mlx_vlm import load
from wire import encode,decode,self_test
ROOT=pathlib.Path(__file__).resolve().parent
OLD=ROOT.parent/'lowpass-search-20261117'
p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--smoke',action='store_true');a=p.parse_args()
MODEL=pathlib.Path(a.model)
def emit(x):print(json.dumps(x,default=str),flush=True)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
# Fail closed before loading or generating if runtime no longer matches earlier evidence.
package=pathlib.Path(mlx_vlm.__file__).resolve().parent
sources={}
expected_sources=json.loads((ROOT/'runtime_hashes.json').read_text())
for old,new in [('language.py','models/diffusion_gemma/language.py'),('wrapper.py','models/diffusion_gemma/diffusion_gemma.py'),('sampler.py','generate/diffusion.py')]:
    installed=package/new
    assert expected_sources[new]==sha(installed),('runtime source mismatch',installed)
    sources[new]=sha(installed)
self_test()
old_tasks=json.loads((OLD/'tasks.json').read_text())
tasks={k:old_tasks[k] for k in ['dev_brackets','heldout_prime','heldout_rle','heldout_dedup','heldout_json']}
tasks.update({
 'new_lower':'Return only Python function lower_bound(xs, target), with no Markdown, imports, examples or explanation. xs is a sorted list of integers, possibly empty or containing duplicates. Return the first index i with xs[i] >= target, or len(xs) if none. Use binary search in O(log n) time.',
 'new_merge':'Return only Python function merge_intervals(intervals), no Markdown, imports, examples or explanation. Each interval is a pair (start, end) of integers with start <= end. Input may be unsorted. Merge overlapping or touching intervals and return sorted (start, end) tuples. Empty input returns []. Do not modify the input.',
 'new_chunks':'Return only Python function chunks(xs, k), no Markdown, imports, examples or explanation. For list xs and positive integer k, return consecutive slices of at most k elements, as a list of lists. Keep the final shorter slice; empty xs returns []. Do not modify xs.',
 'new_moving':'Return only Python function moving_sums(xs, k), no Markdown, imports, examples or explanation. xs is a list of integers and k is positive. Return the sum of each consecutive window of exactly k elements, in order. If len(xs) < k return []. Use O(len(xs)) time and do not modify xs.'})
if (ROOT/'tasks.json').exists():assert json.loads((ROOT/'tasks.json').read_text())==tasks
else:(ROOT/'tasks.json').write_text(json.dumps(tasks,indent=2))
model,processor=load(str(MODEL),lazy=False);mx.synchronize()
tok=getattr(processor,'tokenizer',processor)
config=dict(model.config.generation_config or {});config['sampler_config']={'_cls_name':'EntropyBoundSamplerConfig','entropy_bound':.1};model.config.generation_config=config
dec=model.model.decoder;enc=model.model.encoder
assert len(dec.layers)==30 and model.config.text_config.hidden_size==2816
calls=[];events=[];phase=None;mode='native'
patches=[]

def transfer(x,site):
    event=dict(site=site,phase=phase,shape=list(x.shape),native_dtype=str(x.dtype),mode=mode)
    if mode=='native':
        event.update(bytes=0);events.append(event);return x
    # Force real host bytes, including nibble packing and transmitted scale rounding.
    mx.eval(x);host=np.array(x.astype(mx.float32));start=time.perf_counter()
    packet=encode(host,int(mode));recovered=decode(packet)
    event.update(bytes=len(packet),codec_cpu_s=time.perf_counter()-start,relative_rmse=float(np.sqrt(np.mean((recovered-host)**2))/(np.sqrt(np.mean(host**2))+1e-30)),max_abs_error=float(np.max(np.abs(recovered-host))))
    events.append(event)
    return mx.array(recovered).astype(x.dtype)

cls=type(dec.layers[0]);original_layer=cls.__call__;cuts={id(dec.layers[9]):'forward_10',id(dec.layers[19]):'forward_20'}
def layer_call(self,*args,**kwargs):
    out=original_layer(self,*args,**kwargs)
    return transfer(out,cuts[id(self)]) if id(self) in cuts else out
cls.__call__=layer_call;patches.append((cls,original_layer))
def instrument(target,label):
    cls=type(target);original=cls.__call__
    def call(self,inputs,*args,**kwargs):
        global phase
        if self is not target:return original(self,inputs,*args,**kwargs)
        assert len(calls)<256,'forward safety bound'
        previous=phase;phase=label;calls.append(dict(phase=label,rows=int(inputs.shape[1])))
        try:
            out=original(self,inputs,*args,**kwargs)
            return transfer(out,'feedback') if label=='decoder' else out
        finally:phase=previous
    cls.__call__=call;patches.append((cls,original))
instrument(enc,'encoder');instrument(dec,'decoder')
emit(dict(kind='environment',model=str(MODEL),sources=sources,script_sha256=sha(pathlib.Path(__file__)),codec_sha256=sha(ROOT/'wire.py'),tasks_sha256=sha(ROOT/'tasks.json'),plan_sha256=sha(ROOT/'PLAN.md'),packages={x:importlib.metadata.version(x) for x in ['mlx','mlx-metal','mlx-vlm','transformers','numpy']},device=mx.device_info(),generation_config=config))
signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('180 second trial bound')))
try:
    for seed in ([42] if a.smoke else [42,0,1,2]):
        for task,prompt in tasks.items():
            if a.smoke and task!='heldout_prime':continue
            text=tok.apply_chat_template([{'role':'user','content':prompt}],tokenize=False,add_generation_prompt=True,enable_thinking=False)
            ids=tok.encode(text,add_special_tokens=False);inputs=mx.array([ids],dtype=mx.int32);mx.eval(inputs)
            for mode in ['native','17','16','8','4']:
                gc.collect();mx.synchronize();mx.clear_cache();mx.reset_peak_memory();mx.random.seed(seed)
                calls.clear();events.clear();stats={};start=time.perf_counter();signal.alarm(180)
                try:
                    out=model.generate(inputs,temperature=0.,gen_length=512,steps=48,tokenizer=tok,processor=processor,stats=stats,skip_special_tokens=True,eos_early_stop=True,diffusion_sampler='entropy-bound',diffusion_max_canvas_length=256)
                    mx.eval(out);mx.synchronize();wall=time.perf_counter()-start
                    generated=out[0].tolist();stopped=bool(generated and tok.stopping_criteria(generated[-1]))
                    assert sum(c['phase']=='decoder' for c in calls)==int(stats['diffusion_denoising_steps'])
                    assert len(events)==2*len(calls)+sum(c['phase']=='decoder' for c in calls)
                    cursor=0
                    for c in calls:
                        expected=['forward_10','forward_20']+(['feedback'] if c['phase']=='decoder' else [])
                        for site in expected:
                            e=events[cursor];cursor+=1
                            assert e['site']==site and e['phase']==c['phase'] and e['shape']==[1,c['rows'],2816]
                    emit(dict(kind='trial',task=task,seed=seed,mode=mode,prompt_token_ids=ids,prompt_tokens=len(ids),token_ids=generated,stopped=stopped,raw_token_text=tok.decode(generated,skip_special_tokens=False),text=tok.decode(generated,skip_special_tokens=True),calls=list(calls),all_forward_calls=len(calls),wire_events=list(events),stats=stats,wall_s=wall,metal_peak_bytes=mx.get_peak_memory()))
                except Exception as exc:
                    emit(dict(kind='error',task=task,seed=seed,mode=mode,error=repr(exc),calls=list(calls),wire_events=list(events)));raise
                finally:signal.alarm(0)
finally:
    for cls,original in reversed(patches):cls.__call__=original
