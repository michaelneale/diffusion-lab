"""Standalone local Nemotron probe. No model code or persistent services modified."""
import argparse
import gc
import hashlib
import importlib.metadata
import inspect
import json
import math
import pathlib
import resource
import sys
import time

import mlx.core as mx
from mlx_vlm import load
from mlx_vlm.models.nemotron_labs_diffusion.language import NemotronLabsDiffusionEncoder

ROOT = pathlib.Path(__file__).resolve().parent
TASKS = json.loads((ROOT / 'tasks.json').read_text())
RECORDS = []
ORIGINAL = NemotronLabsDiffusionEncoder.__call__

def counted(self, inputs, *args, **kwargs):
    cache = kwargs.get('cache')
    RECORDS.append({'rows': int(inputs.shape[1]),
                    'causal': kwargs.get('use_causal_mask', False),
                    'cache_write': kwargs.get('use_cache', True),
                    'prefix': int(cache[0].offset) if cache and cache[0] is not None else 0})
    return ORIGINAL(self, inputs, *args, **kwargs)

NemotronLabsDiffusionEncoder.__call__ = counted

def emit(x):
    print(json.dumps(x, ensure_ascii=False, allow_nan=False), flush=True)

def clean_stats(x):
    return {k: None if isinstance(v, float) and not math.isfinite(v) else v for k,v in x.items()}

def trial(model, tok, task, mode='diffusion', block=32, threshold=.9, steps=32, repeat=0, context=0, parity=True):
    text = TASKS[task]
    if context:
        filler = tok.encode('Reference note: a reliable service measures latency and memory.\n', add_special_tokens=False)
        text = 'Reference notes:\n' + tok.decode((filler*(context//len(filler)+1))[:context]) + '\nNow answer:\n' + text
    prompt = tok.apply_chat_template([{'role':'user', 'content':text}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
    ids = tok.encode(prompt, add_special_tokens=False)
    inputs = mx.array([ids], dtype=mx.int32)
    maximum = 1536 if task == 'list100' else 512
    gc.collect()
    mx.synchronize()
    mx.clear_cache()
    mx.reset_peak_memory()
    mx.random.seed(42)
    RECORDS.clear()
    stats, events = {}, []
    start = time.perf_counter()
    def callback(r):
        if r.text:
            events.append({'s': time.perf_counter()-start, 'tokens':r.generation_tokens, 'text':r.text})
        return True
    if mode == 'ar':
        output, library_nfe = model.language_model.ar_generate(inputs, max_new_tokens=maximum, temperature=0., stats=stats)
        output = output[:,len(ids):]
    else:
        output = model.language_model.generate(inputs, gen_length=maximum, block_length=block, steps=steps,
            threshold=threshold, temperature=0., eos_early_stop=True, generation_mode='diffusion',
            sampler='native', ar_weight=0., transformers_parity=parity, stats=stats, tokenizer=tok, skip_special_tokens=True, on_result=callback)
        library_nfe = None
    mx.eval(output)
    mx.synchronize()
    elapsed = time.perf_counter()-start
    generated = output[0].tolist()
    eos = model.config.eos_token_id
    endings = set(eos if isinstance(eos, list) else [eos])
    stopped = bool(generated and generated[-1] in endings)
    visible = generated[:-1] if stopped else generated
    assert model.config.mask_token_id not in visible, 'Unfilled mask in visible output'
    assert not any(t in endings for t in visible), 'Unexpected internal end token'
    calls = list(RECORDS)
    assert calls and calls[0]['rows'] == len(ids) and calls[0]['causal'] and calls[0]['cache_write']
    if mode == 'diffusion':
        assert len(calls) == 1 + int(stats['diffusion_total_nfe'])
        assert sum(not c['causal'] for c in calls) == int(stats['diffusion_denoise_nfe'])
        assert sum(c['cache_write'] for c in calls) == 1 + int(stats['diffusion_post_block_nfe'])
        assert stats['diffusion_mixed_ar_forwards'] == 0
        assert stats['diffusion_transformers_parity'] == float(parity)
    elif stopped:
        assert len(calls) == len(generated) == library_nfe
    result = dict(kind='trial', task=task, mode=mode, block=block, threshold=threshold, steps_cap_per_block=steps,
        repeat=repeat, transformers_parity=parity if mode == 'diffusion' else False, synthetic_context=context, max_tokens=maximum, prompt_token_ids=ids,
        prompt_tokens=len(ids), token_ids=generated, visible_tokens=len(visible), stopped=stopped,
        calls=calls, all_forward_calls=len(calls), all_rows=sum(c['rows'] for c in calls),
        wall_s=elapsed, first_text_s=events[0]['s'] if events else None,
        metal_peak_bytes=mx.get_peak_memory(), metal_active_bytes=mx.get_active_memory(),
        process_lifetime_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        stats=clean_stats(stats), events=events, text=tok.decode(visible, skip_special_tokens=True))
    emit(result)
    return result


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--model', required=True)
    p.add_argument('--suite', choices=['smoke','baseline','sweep','machines','context'], default='smoke')
    args=p.parse_args()
    source=pathlib.Path(inspect.getsourcefile(ORIGINAL))
    emit(dict(kind='environment', device=mx.device_info(), python=sys.version, source=str(source),
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(), packages={n:importlib.metadata.version(n) for n in ['mlx','mlx-metal','mlx-vlm','transformers']}, model_path=args.model))
    start=time.perf_counter()
    model,processor=load(args.model, lazy=False, trust_remote_code=False)
    mx.synchronize()
    tok=getattr(processor,'tokenizer',processor)
    emit(dict(kind='load', wall_s=time.perf_counter()-start, active_bytes=mx.get_active_memory(), config=model.config.to_dict() if hasattr(model.config,'to_dict') else vars(model.config)))
    if args.suite == 'smoke':
        for mode in ['diffusion','ar']: trial(model,tok,'extract',mode)
    elif args.suite == 'baseline':
        for task in ['extract','reason','travel','boxes','discount','brackets','gcd','dedup','prose','list32']:
            for mode in ['diffusion','ar']: trial(model,tok,task,mode)
    elif args.suite == 'sweep':
        for block,threshold,steps in [(32,.8,32),(64,.8,64),(64,.6,64),(32,.9,4),(128,.8,128)]:
            for task in ['reason','travel','brackets','prose','list32']:
                trial(model,tok,task,block=block,threshold=threshold,steps=steps)
    elif args.suite == 'machines':
        trial(model,tok,'extract') # cold, excluded from repetitions
        for repeat in range(3):
            for task in ['travel','brackets','prose']:
                for mode in ['diffusion','ar']:
                    trial(model,tok,task,mode,repeat=repeat)
    else:
        for context in [4096,16384]:
            for mode in ['diffusion','ar']:
                trial(model,tok,'extract',mode,context=context)

if __name__ == '__main__': main()
