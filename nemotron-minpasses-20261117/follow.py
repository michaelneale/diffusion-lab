"""Bounded follow-up; native selection unchanged, optional efficient runtime kernels."""
import bench
import argparse
import time
import pathlib
import hashlib

p=argparse.ArgumentParser()
p.add_argument('--model',required=True)
p.add_argument('--suite',choices=['quality','efficient'],required=True)
a=p.parse_args()
start=time.perf_counter()
model,processor=bench.load(a.model,lazy=False,trust_remote_code=False)
bench.mx.synchronize()
tok=getattr(processor,'tokenizer',processor)
bench.emit(dict(kind='environment',device=bench.mx.device_info(),model_path=a.model,load_s=time.perf_counter()-start,
    source_sha256=hashlib.sha256(pathlib.Path(bench.inspect.getsourcefile(bench.ORIGINAL)).read_bytes()).hexdigest(),
    packages={n:bench.importlib.metadata.version(n) for n in ['mlx','mlx-metal','mlx-vlm','transformers']}))
try:
    if a.suite=='quality':
        for task in ['extract','reason','travel','brackets','gcd']:
            for block,threshold in [(32,.6),(32,.4),(16,.9),(32,.95)]:
                bench.trial(model,tok,task,block=block,threshold=threshold)
    else:
        for task in ['travel','brackets','prose']:
            bench.trial(model,tok,task,parity=False)
        for context in [4096,16384]:
            bench.trial(model,tok,'extract',context=context,parity=False)
        # AR at 16K runs independently even where reference diffusion failed allocation.
        bench.trial(model,tok,'extract',mode='ar',context=16384)
finally:
    bench.NemotronLabsDiffusionEncoder.__call__=bench.ORIGINAL
