"""Post-selected successful settings, repeated; not a held-out quality benchmark."""
import bench
import sys
import time
start=time.perf_counter()
model,processor=bench.load(sys.argv[1],lazy=False,trust_remote_code=False)
bench.mx.synchronize()
tok=getattr(processor,'tokenizer',processor)
bench.emit(dict(kind='environment', device=bench.mx.device_info(), load_s=time.perf_counter()-start, note='post-selected configurations'))
try:
    bench.trial(model,tok,'extract')
    for repeat in range(3):
        for task,block,threshold in [('travel',32,.6),('brackets',16,.9),('gcd',32,.9)]:
            for mode in ['diffusion','ar']:
                bench.trial(model,tok,task,mode,block=block,threshold=threshold,repeat=repeat)
finally:
    bench.NemotronLabsDiffusionEncoder.__call__=bench.ORIGINAL
