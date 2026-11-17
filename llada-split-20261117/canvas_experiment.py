#!/usr/bin/env python3
"""Bounded external research driver. Trusted same-build SSH pipes, not product wire."""
import argparse
import json
import math
import os
from pathlib import Path
import signal
import shlex
import subprocess
import time

ROOT = Path(__file__).resolve().parent
MODEL = 'models/LLaDA-8B-Instruct.Q4_K_M.gguf'
MASK = 126336
SPECIAL_END = {126081, 126348}


def exact(stream, count):
    result = bytearray()
    while len(result) < count:
        chunk = stream.read(count - len(result))
        if not chunk:
            raise RuntimeError('worker closed mid-payload')
        result.extend(chunk)
    return bytes(result)


def worker_command(mode, remote=False):
    if not remote:
        return [str(ROOT / 'canvas_worker'), MODEL, mode]
    host = os.environ.get('DIFFUSION_SSH_HOST')
    directory = os.environ.get('DIFFUSION_REMOTE_DIR')
    if not host or not directory:
        raise ValueError('Set DIFFUSION_SSH_HOST and DIFFUSION_REMOTE_DIR for split runs')
    command = (f'cd {shlex.quote(directory)} && mkdir -p home && '
               f'HOME="$PWD/home" exec ./canvas_worker {shlex.quote(MODEL)} tail')
    return ['ssh', '-T', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=8',
            '-o', 'ServerAliveInterval=10', '-o', 'ServerAliveCountMax=3',
            '--', host, command]


class Worker:
    def __init__(self, mode, remote=False):
        self.mode = mode
        command = worker_command(mode, remote)
        (ROOT / 'home').mkdir(exist_ok=True)
        self.log = open(ROOT / f'canvas-{mode}.log', 'ab')
        env = dict(os.environ, HOME=str(ROOT / 'home'))
        self.p = subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.PIPE,
                                  stdout=subprocess.PIPE, stderr=self.log)
        ready = self.p.stdout.readline()
        if not ready:
            self.close()
            raise RuntimeError(f'{mode} failed startup; see canvas-{mode}.log')
        self.ready = json.loads(ready)
        assert self.ready['ready'] and self.ready['protocol'] == 'trusted-lab-canvas-v1'

    def ask(self, req, payload=b''):
        req = dict(req, payload_bytes=len(payload))
        self.p.stdin.write((json.dumps(req, separators=(',', ':')) + '\n').encode())
        self.p.stdin.write(payload)
        self.p.stdin.flush()
        line = self.p.stdout.readline()
        if not line:
            raise RuntimeError(f'{self.mode} exited; see worker log')
        reply = json.loads(line)
        count = reply.get('payload_bytes', 0)
        if not 0 <= count <= 64 * 1024 * 1024:
            raise RuntimeError('reply payload outside bound')
        return reply, exact(self.p.stdout, count)

    def close(self):
        if hasattr(self, 'p'):
            try:
                if self.p.poll() is None:
                    self.p.stdin.write(b'{"op":"stop"}\n')
                    self.p.stdin.flush()
                self.p.wait(timeout=10)
            except (OSError, subprocess.TimeoutExpired):
                self.p.kill()
                self.p.wait(timeout=10)
            for stream in [self.p.stdin, self.p.stdout]:
                stream.close()
        self.log.close()


def template(messages):
    return '<|startoftext|>' + ''.join(
        f'<|start_header_id|>{role}<|end_header_id|>\n\n{text.strip()}<|eot_id|>'
        for role, text in messages) + '<|start_header_id|>assistant<|end_header_id|>\n\n'


def choose_updates(tokens, prompt_len, ids, confidence, count, block_start, block_end):
    assert len(ids) == len(confidence) == len(tokens) - prompt_len
    masked = [i for i in range(block_start, block_end) if tokens[i] == MASK]
    assert 0 < count <= len(masked)
    assert all(math.isfinite(p) and 0 <= p <= 1 for p in confidence)
    ranked = sorted(masked, key=lambda i: (-confidence[i - prompt_len], i))[:count]
    for i in ranked:
        if ids[i - prompt_len] == MASK:
            raise RuntimeError('model selected mask; refusing false progress')
        tokens[i] = ids[i - prompt_len]
    return ranked


def adaptive_count(tokens, prompt_len, confidence, lo, hi, threshold):
    assert 0 <= threshold <= 1
    masked = [i for i in range(lo, hi) if tokens[i] == MASK]
    assert masked
    return max(1, sum(confidence[i - prompt_len] >= threshold for i in masked))


def parse_tool(text):
    text = text.strip()
    if text.startswith('```json\n') and text.endswith('\n```'):
        text = text[8:-4]
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def selftest():
    assert parse_tool('```json\n{"tool":"lab_status","arguments":{}}\n```') == {'tool':'lab_status','arguments':{}}
    assert parse_tool('run this {"tool":"lab_status"}') is None
    tokens = [11, MASK, MASK, MASK, MASK]
    chosen = choose_updates(tokens, 1, [20, 21, 22, 23], [.1, .9, .9, 1], 2, 1, 4)
    assert chosen == [2, 3] and tokens == [11, MASK, 21, 22, MASK]
    assert adaptive_count([11, MASK, MASK, MASK, MASK], 1, [.1, .9, .9, 1], 1, 4, .9) == 2
    assert adaptive_count([11, MASK, MASK, MASK, MASK], 1, [.1, .9, .9, 1], 1, 4, .95) == 1
    assert adaptive_count([11, 20, MASK, MASK, MASK], 1, [1, .9, .8, 1], 1, 4, .9) == 1
    for steps in [8, 16, 32, 64]:
        schedule = [256 // steps + (i < 256 % steps) for i in range(steps)]
        assert sum(schedule) == 256
    assert template([('user', ' hi ')]) == '<|startoftext|><|start_header_id|>user<|end_header_id|>\n\nhi<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n'


class Experiment:
    def __init__(self, output):
        self.workers = []
        self.output = output
        self.results = {'decoder': 'greedy, cfg=0, confidence-ranked fixed transfers, no prompt KV cache',
                        'transport': 'persistent SSH stdio, raw native activation descriptor + F32 payload',
                        'runs': [], 'parity': []}
        try:
            for mode, remote in [('full', False), ('head', False), ('tail', True)]:
                self.workers.append(Worker(mode, remote))
        except BaseException:
            self.close()
            raise
        self.full, self.head, self.tail = self.workers
        self.request_id = 0

    def save(self):
        self.output.write_text(json.dumps(self.results, indent=2))

    def close(self):
        for worker in reversed(self.workers):
            worker.close()

    def tokenize(self, text):
        return self.full.ask({'op': 'tokenize', 'text': text})[0]['tokens']

    def detokenize(self, tokens):
        return self.full.ask({'op': 'detokenize', 'tokens': tokens})[0]['text']

    def evaluate(self, tokens, prompt_len, split):
        self.request_id += 1
        req = {'op': 'eval', 'id': self.request_id, 'tokens': tokens, 'read_start': prompt_len,
               'codec': getattr(self, 'codec', 'f32') if split else 'f32'}
        started = time.monotonic()
        if split:
            a, payload = self.head.ask(req)
            after_head = time.monotonic()
            b, empty = self.tail.ask(req, payload)
            assert not empty and a['id'] == b['id'] == self.request_id
            stats = {'head_compute_s': a['compute_s'], 'tail_compute_s': b['compute_s'],
                     'activation_bytes': a['activation_bytes'], 'framed_bytes': len(payload),
                     'head_wall_s': after_head - started,
                     'tail_roundtrip_s': time.monotonic() - after_head,
                     'readout_s': b['readout_s']}
        else:
            b, empty = self.full.ask(req)
            assert not empty and b['id'] == self.request_id
            stats = {'compute_s': b['compute_s'], 'readout_s': b['readout_s']}
        stats['wall_s'] = time.monotonic() - started
        assert len(b['ids']) == len(tokens) - prompt_len
        return b, stats

    def parity(self, prompt, size):
        canvas = prompt + [MASK] * size
        a, sa = self.evaluate(canvas, len(prompt), False)
        b, sb = self.evaluate(canvas, len(prompt), True)
        agreement = sum(x == y for x, y in zip(a['ids'], b['ids']))
        result = {'positions': len(canvas), 'readout_rows': size, 'top1_agreement': agreement,
                  'max_confidence_abs_difference': max(abs(x-y) for x,y in zip(a['confidence'], b['confidence'])),
                  'max_top_logit_abs_difference': max(abs(x-y) for x,y in zip(a['top_logits'], b['top_logits'])),
                  'full': sa, 'split': sb}
        self.results['parity'].append(result)
        self.save()
        print('PARITY', json.dumps(result), flush=True)
        return a, b

    def generate(self, name, prompt, size, steps, split, block=None, threshold=None):
        block = block or size
        assert size % block == 0 and steps % (size // block) == 0
        block_steps = block if threshold is not None else steps // (size // block)
        assert block_steps <= block
        tokens = prompt + [MASK] * size
        records = []
        started = time.monotonic()
        for block_index in range(size // block):
            lo = len(prompt) + block_index * block
            hi = lo + block
            for step in range(block_steps):
                if MASK not in tokens[lo:hi]:
                    break
                reply, stats = self.evaluate(tokens, len(prompt), split)
                count = (adaptive_count(tokens, len(prompt), reply['confidence'], lo, hi, threshold)
                         if threshold is not None else block // block_steps + (step < block % block_steps))
                chosen = choose_updates(tokens, len(prompt), reply['ids'], reply['confidence'], count, lo, hi)
                stats.update(step=len(records), committed=count, selected_positions=chosen,
                             mean_selected_confidence=sum(reply['confidence'][i-len(prompt)] for i in chosen)/count)
                records.append(stats)
                print(f'PASS {name} {len(records)}/{steps} {stats["wall_s"]:.3f}s', flush=True)
        elapsed = time.monotonic() - started
        assert tokens[:len(prompt)] == prompt and MASK not in tokens[len(prompt):]
        generated = tokens[len(prompt):]
        end = next((i for i,t in enumerate(generated) if t in SPECIAL_END),len(generated))
        text = self.detokenize(generated[:end])
        result = {'name': name, 'split': split, 'prompt_tokens': len(prompt), 'canvas_tokens': size,
                  'visible_tokens_before_end': end, 'steps': len(records), 'threshold': threshold, 'block_length': block,
                  'elapsed_s': elapsed, 'canvas_tokens_per_s': size/elapsed,
                  'visible_tokens_per_s': end/elapsed, 'activation_bytes': sum(r.get('activation_bytes',0) for r in records),
                  'text': text, 'generated_token_ids': generated, 'passes': records}
        self.results['runs'].append(result)
        self.save()
        print('RESULT', json.dumps({k:v for k,v in result.items() if k not in ('passes','generated_token_ids')}), flush=True)
        return result

    def long_prompt(self):
        # Synthetic long chat context with a concrete, checkable incident task.
        facts = ('Incident report: checkout HTTP 500 errors rose from 0.2% to 8% immediately after deploy v42 at 10:05 UTC. '
                 'Database latency stayed 12 ms. CPU stayed 40%. Error logs say missing TAX_REGION. '
                 'Canary rollback to v41 reduced errors to 0.3%. No payments were lost; failed requests returned before charge creation. ')
        filler = ('Archived operations note: compare the canary against the stable release; preserve logs, verify transaction counts, '
                  'and report observed measurements rather than inventing missing numbers. ')
        beginning = self.tokenize(template([('user', facts + '\nReference notes:\n')]))
        # Preserve the exact task and generation marker at the end; truncate only filler.
        prefix_text = '<|startoftext|><|start_header_id|>user<|end_header_id|>\n\n' + facts + '\nReference notes:\n'
        prefix = self.tokenize(prefix_text)
        suffix = self.tokenize('\nTask: Write a practical incident response in about 180 words. Identify the likely cause, the immediate safe action, '
                               'three checks before redeploy, and a short customer update. Use the supplied facts and do not invent measurements.'
                               '<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n')
        fill = self.tokenize(filler * 70)
        need = 2048 - len(prefix) - len(suffix)
        assert need > 0 and len(fill) >= need and beginning
        return prefix + fill[:need] + suffix


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('suite', choices=['smoke','short','long','agent'])
    parser.add_argument('--steps', default='16,32')
    parser.add_argument('--block', type=int)
    parser.add_argument('--codec', choices=['f32','f16'], default='f32')
    args = parser.parse_args()
    selftest()
    suffix = (f'-block{args.block}' if args.block else '') + (f'-{args.codec}' if args.codec!='f32' else '') + f'-steps{args.steps.replace(",", "_")}-selective'
    experiment = Experiment(ROOT / f'canvas-{args.suite}{suffix}-results.json')
    experiment.codec = args.codec
    experiment.results['codec'] = args.codec
    try:
        e = experiment
        if args.suite == 'smoke':
            prompt = e.tokenize(template([('user','What is 2 + 2? Answer in one sentence.')]))
            e.parity(prompt,16)
            first,_ = e.evaluate(prompt+[MASK]*16,len(prompt),True)
            changed = prompt+[MASK]*16;changed[-1]=19
            e.evaluate(changed,len(prompt),True)
            again,_=e.evaluate(prompt+[MASK]*16,len(prompt),True)
            assert first['ids']==again['ids'] and first['confidence']==again['confidence']
            e.results['replacement_canvas_aba_identical']=True
            e.generate('arithmetic-local',prompt,32,16,False)
            e.generate('arithmetic-split',prompt,32,16,True)
            e.save()
        elif args.suite == 'short':
            prompt=e.tokenize(template([('user','A checkout deployment caused errors to rise from 0.2% to 8%. '
                'Logs show missing TAX_REGION, while database latency and CPU are unchanged. A canary rollback restored normal service. '
                'Write an incident response plan in about 150 words: likely cause, safe immediate action, three checks before redeploy, and a customer update. Do not invent numbers.')]))
            e.parity(prompt,256)
            for steps in map(int,args.steps.split(',')):
                e.generate(f'incident-local-{steps}',prompt,256,steps,False,args.block)
                e.generate(f'incident-split-{steps}',prompt,256,steps,True,args.block)
        elif args.suite == 'long':
            prompt=e.long_prompt();assert len(prompt)==2048
            e.results['prompt_text']=e.detokenize(prompt)
            e.parity(prompt,256)
            for steps in map(int,args.steps.split(',')):
                e.generate(f'2k-incident-local-{steps}',prompt,256,steps,False,args.block)
                e.generate(f'2k-incident-split-{steps}',prompt,256,steps,True,args.block)
        else:
            task=('You are checking a two-computer inference experiment. You have one read-only tool: lab_status, '
                  'which takes an empty arguments object and returns measured facts. First call the tool; do not guess its results. '
                  'Reply ONLY with a JSON object in this format: {"tool":"lab_status","arguments":{}}')
            prompt=e.tokenize(template([('user',task)]))
            prior = ROOT / 'canvas-agent-results.json'
            if prior.exists():
                action=json.loads(prior.read_text())['runs'][0]
                e.results['runs'].append(action)
                e.results['tool_request_reused_from_saved_run']=True
            else:
                action=e.generate('agent-tool-request',prompt,64,32,True)
            call=parse_tool(action['text'])
            e.results['valid_tool_request']=call=={'tool':'lab_status','arguments':{}}
            if e.results['valid_tool_request']:
                result={'model':'LLaDA-8B-Instruct Q4_K_M','first_host':'M5 Max','second_host':'Mac Mini Apple M4',
                        'second_host_memory_gib':16,'layers_per_host':16,'smoke_canvas_positions':37,
                        'smoke_activation_bytes':606208,'last_position_argmax_matched':True,
                        'transport':'SSH','full_generation_speed_advantage':'not established'}
                e.results['executed_tool_result']=result
                messages=[('user',task),('assistant',action['text']),('user','Tool lab_status returned: '+json.dumps(result)+
                    '\nNow write about 150 words explaining what worked, what remains unproven, and the next useful measurement. '
                    'Use these facts, and do not claim a speedup or native tool-calling support.')]
                e.generate('agent-final-answer',e.tokenize(template(messages)),256,32,True)
            e.save()
    finally:
        experiment.close()


if __name__ == '__main__':
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError('bounded experiment timeout')))
    signal.alarm(1500)
    main()
