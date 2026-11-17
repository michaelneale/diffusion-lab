# diffusion-lab

Research drivers for **diffusion text generation over high-latency model splits**.
Prepared by Melkor (an AI agent), on behalf of Mic / `michaelneale`.

The question: can fewer sequential whole-model evaluations compensate for the
extra compute and activation traffic of block denoising when a model must span
several machines?

**This is a source-only research release, not a production inference engine or a
claim that diffusion beats autoregressive generation.**

Findings and proposed next steps:
[Mesh-LLM/mesh-llm #3124](https://github.com/Mesh-LLM/mesh-llm/issues/3124).

## What is here

| Directory | Experiment | Execution actually performed in the original research |
|---|---|---|
| `llada-split-20261117/` | Native LLaDA canvas generation, adaptive acceptance, prefix stopping, d3LLM sweeps | A real two-machine SSH split for the original LLaDA checkpoint; later low-pass/d3 probes were local |
| `nemotron-minpasses-20261117/` | Nemotron diffusion/AR, context and threshold comparisons | Local MLX runs on two different Apple machines, not split inference |
| `lowpass-search-20261117/` | DiffusionGemma 26B and LLaDA 16B/~103B pass-count/quality search | Local MLX runs, including failed outputs and precision controls |
| `lossy-diffusion-20261118/` | BF16/FP16/INT8/INT4 activation serialization and perturbation | Real byte encode/decode inside a local model; network costs only modeled |
| `tests/` | Offline helper, codec and packaging checks | No model, SSH connection or native executable required |

Prompts are synthetic benchmark tasks. The incident scenario is a test fixture,
not a disclosure of a real operational incident.

Not included: model weights, native binaries/libraries, copied upstream model
implementations, source snapshots, raw lab logs or trial-result archives. The
historical report counts are **not independently reproducible from this bundle
alone**: the validators need the original reviewed evidence or newly generated,
reviewed results. `incident_prompt.json` contains only the original 89 input token
IDs needed by the native follow-ups, not result records.

## Start here: offline checks

Python 3.10+ and NumPy are sufficient for the offline suite. The original inference
runs used Python 3.13; these scripts have not been qualified across every Python
version or OS. Signal-based validators target macOS/Linux, not Windows.

```bash
python3 -m venv .venv
. .venv/bin/activate
python3 -m pip install -r requirements-offline.txt
python3 -m unittest discover -s tests -v
```

Do **not** run with `python3 -O`: assertions enforce research invariants.

## MLX experiments

Apple Silicon with sufficient memory and the dedicated diffusion implementations
is required. Historical package/model pins are in [PROVENANCE.md](PROVENANCE.md).
Do not assume a current package with a similar name implements the same sampler.
Obtain checkpoints separately, at the pinned revisions and under their own terms.
No script in this release automatically downloads weights.

Example commands (replace `/path/to/checkpoint` with a local snapshot):

```bash
cd lowpass-search-20261117
python3 bench.py --family gemma --model /path/to/checkpoint \
  --suite heldout --steps 48 > run.jsonl
# After manually reviewing generated code in run.jsonl:
python3 validate.py
```

Other families: `--family llada20` or `--family llada22`. LLaDA2.2 loads its bundled
`llada22_mlx` implementation from the checkpoint directory and enables
`trust_remote_code`; inspect that separately acquired code before running it.

```bash
cd nemotron-minpasses-20261117
python3 bench.py --model /path/to/checkpoint --suite baseline > m5-baseline.jsonl
python3 follow.py --model /path/to/checkpoint --suite quality > m5-quality.jsonl
python3 selected.py /path/to/checkpoint > m5-selected.jsonl
# Review the outputs before running validate.py.
```

The historical `m5`/`mini` names in the Nemotron summary identify experiment groups,
not SSH destinations. A different machine should not be reported as original M5
evidence merely because a file has that name.

```bash
cd lossy-diffusion-20261118
python3 wire.py
python3 bench.py --model /path/to/pinned/gemma4bit > results.jsonl
```

The lossy harness checks installed runtime files against `runtime_hashes.json`
**before model loading**. A mismatch deliberately stops the run: do not bypass it
and call the result a historical reproduction. It also reads the sibling lowpass
tasks. Its `validate.py` requires the matching historical `gemma4-dev48.jsonl` and
`gemma4-heldout48{,-seed0,-seed1,-seed2}.jsonl` in the sibling lowpass directory for
exact-output controls, plus the complete five-mode results, not merely `--smoke`.

## Native LLaDA experiment

`canvas_worker.cpp` uses **internal Skippy C++ headers and a matching native
library**, not just the public ABI. The source/build pins are in PROVENANCE.md.
This release does not provide a turnkey native build or certify that the pinned
prepared source is obtainable from a public checkout. A matching build must
produce `canvas_worker` in `llada-split-20261117/` on each participating machine.
No native rebuild was performed for this publication.

Place the original model at `models/LLaDA-8B-Instruct.Q4_K_M.gguf` relative to that
directory; local distilled probes additionally use
`models/d3LLM_LLaDA.Q4_K_M.gguf`. The worker's split planner pins the original
checkpoint identity and the 16/16 layer split. **Do not use that plan for another
checkpoint.** The d3 probes use full-model mode only.

For split runs, configure a trusted SSH host and a remote directory (absolute, or
relative to that account's home; do not use literal `~`):

```bash
export DIFFUSION_SSH_HOST='user@your-lab-host'
export DIFFUSION_REMOTE_DIR='diffusion-lab/llada-split-20261117'
cd llada-split-20261117
python3 canvas_experiment.py smoke --codec f16
python3 canvas_experiment.py short --steps 256 --block 32 --codec f16
python3 adaptive_experiment.py --split --thresholds .8
```

The driver creates and uses a scratch `home/` for each native worker, including
remote workers, rather than the real account HOME. Stage transport is persistent
SSH stdio with bounded, same-build activation descriptors. It is **not a portable
or untrusted-peer protocol**. Hosts must have matching builds and weights already
installed. The scripts never configure SSH credentials.

Local-only examples:

```bash
python3 wan_sweep.py selftest    # no model or native worker needed
python3 probe32_experiment.py main
python3 adaptive_experiment.py --thresholds .8
python3 wan_fresh.py original
python3 wan_fresh.py d3         # checks prompt IDs against original run
python3 wan_minpasses.py d3 short
```

Despite their historical `wan_*` filenames, these follow-up sweeps do not execute
WAN transport. Network estimates are not WAN measurements.

## Measurement and safety limits

- Count all sequential backbone calls, including prefill/cache/encoder work, not
  just denoising iterations. Per-model hooks describe what they count.
- Timing generally excludes loading and tokenization. Ordinary desktop timing is
  not a controlled performance benchmark.
- Selected short successes are not a general low-pass quality guarantee. Fixed
  seeds and equal weights did not guarantee identical output across devices.
- Functional versus strict-format correctness differ. Some older JSON checks use
  Python equality, which conflates booleans/integers and ignores duplicate keys;
  the later lossy validator explicitly checks these. We preserve this historical
  distinction rather than silently rescore old evidence.
- The `agent` native suite uses **hardcoded historical lab facts** as a synthetic
  tool result; it is not live machine discovery or a real tool-calling benchmark.
- **Validators execute generated Python. Their AST filters are NOT a sandbox.**
  Only use them on manually reviewed code, in a disposable restricted environment.
  They write summary files beside the input records. Do not feed untrusted archives
  into them or treat a timeout/AST filter as a security boundary.
- New runs record local model/source paths and device information. Review and
  sanitize their outputs before publishing; generated files are ignored by git.

## Publication changes

Original evidence remains untouched. Changes to the public copy are intentionally
small: configurable SSH destination, scratch-HOME creation, a local-only adaptive
path that does not start remote workers, a required lossy `--model` path, runtime
hashes instead of copied source snapshots, a stripped prompt fixture, and explicit
failure when four archive validators find no trial records. Sampling and numerical
codec policies are unchanged. Tests cover the offline behavior; they are not an
inference rerun.

See [THIRD_PARTY.md](THIRD_PARTY.md) for upstream references and licensing scope.
