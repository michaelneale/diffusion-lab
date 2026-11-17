# Historical provenance

These pins describe the November 17–18, 2026 research, not a promise that every
artifact remains available upstream. The publication cleanup did not rerun
inference. Full findings are in [issue #3124](https://github.com/Mesh-LLM/mesh-llm/issues/3124).

## MLX runtime

Original packages: `mlx==0.32.4`, `mlx-metal==0.32.4`, `mlx-vlm==0.8.0`,
`transformers==5.23.0`; recorded Nemotron environment additionally has
`numpy==2.5.5`, `tokenizers==0.23.3`, `huggingface-hub==2.4.0`.
Python 3.13.14 on the M5; 3.13.15 on the Mini.

These are an environment record, not a portable dependency lock. Model-specific
APIs may change. The package-name/version pins alone do not establish official
BF16 numerical parity. The lossy harness also verifies three exact source hashes
from its sibling `runtime_hashes.json`, derived from archived runtime snapshots.
The Nemotron encoder source SHA256 was
`5ea1733b3d18527f36d3f0118ac09a9a24c3ecef1d98ece6e9699be8e82263d3`.

| Hugging Face repository | Revision |
|---|---|
| `mlx-community/diffusiongemma-26B-A4B-it-4bit` | `a7a81407613811e8ba63af92ac0d852b809e191f` |
| `mlx-community/diffusiongemma-26B-A4B-it-8bit` | `7b95e3887078ba56283c24f2578d6e5a06b9d7e8` |
| `m1rkocasu/LLaDA2.2-mini-MLX-5bit` | `52eb9687f397fb2c3ba170d69fc686b505faf781` |
| `mlx-community/LLaDA2.0-mini-4bit` | `5aa66babf7e8015fc7fffce1920c8328a58ac868` |
| `mlx-community/LLaDA2.0-flash-4bit` | `d0472eea7a5220199b8963921fd8c558136d892c` |
| `mlx-community/Nemotron-Labs-Diffusion-3B-4bit` | `5fd1ad4bcbd02a3fd489ec6e87059e072118548c` |

The Gemma 4-bit conversion is mixed precision, including 8-bit module overrides;
"4-bit" does not mean every tensor is four bits. LLaDA2.0-flash is approximately
103B stored parameters / 6.1B active, **not dense 100B compute**.

## Native LLaDA

- Mesh source: `1ea1ecf754533a6a4fcc0eff8d3f33ff5b73762f`.
- Prepared native source: `46d38e0020de8804be2b959243cd3243d9a7a242`.
- Recorded native upstream pin: `372dc5ca157276beb6c67b2e92477e8d662e964a`.
- Skippy ABI: `0.1.79`; the worker additionally requires internal C++ headers.
- Original checkpoint: `mradermacher/LLaDA-8B-Instruct-GGUF`, revision
  `7887b62da83a936f64d5fcb6e61064415a0f572c`, `LLaDA-8B-Instruct.Q4_K_M.gguf`.
- Original weight SHA256:
  `8d313697ff01d76e0477ec30fe0b117e816d4da63af091b4d6a2a3e2eba23355`.
- Distilled local-probe filename: `d3LLM_LLaDA.Q4_K_M.gguf`, SHA256
  `eb450d47abe05c1c1503d78751a8453c8834ef1a8d34802173914379b7cafa88`.

The original split was layers 0–15 on an M5 Max / 128 GiB and 16–31 on an M4
Mac Mini / 16 GiB. Native transport used same-build activation descriptors over
persistent SSH. No general-purpose model-distribution mechanism is supplied here.
The source hashes above are evidence identifiers, not complete build instructions.

## Source publication versus results

The published scripts were copied from preserved experiment sources, then cleaned
as described in README.md. Archive outputs were not copied into this repository.
The public issue includes successes, failures, measurement conventions and the
modeled-network caveats; it is the findings record, not this repository's test suite.
