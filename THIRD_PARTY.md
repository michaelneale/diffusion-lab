# Upstream references and licensing scope

The MIT license in this repository covers its original research harness code,
fixtures, tests and documentation. It does not license any separately obtained
model weights, model implementation, native runtime or other dependency.

No upstream reference implementation, model source snapshot, native library or
weights are bundled. Acquire those separately and comply with their own licenses
and model terms. In particular, do not infer a model's license from the license of
an inference package or this repository.

Algorithm/runtime references used during the research:

- [LLaDA](https://github.com/ML-GSAI/LLaDA): instruct formatting and masked diffusion generation.
- [Fast-dLLM](https://github.com/NVlabs/Fast-dLLM/tree/main/v1/llada): confidence-based unmasking/no-cache policy. The archived reference file carried NVIDIA's Apache-2.0 notice; it is not redistributed here. Our native experiment implements that policy separately, rather than running the official PyTorch implementation.
- [d3LLM](https://github.com/hao-ai-lab/d3LLM/tree/e5bfcaccb775ee75759c24ccc9a28218b8372b34): distilled diffusion checkpoint and generation reference. Local checkpoint-swap probes are not a reproduction of the official benchmark.
- [DFlash](https://github.com/z-lab/dflash): comparison/reference only; no DFlash implementation is included and no DFlash execution is claimed.
- [MLX](https://github.com/ml-explore/mlx) and [MLX-VLM](https://github.com/Blaizzy/mlx-vlm): runtime dependencies for the Python experiments. Historical versions and source fingerprints are in PROVENANCE.md.
- [mesh-llm](https://github.com/Mesh-LLM/mesh-llm): native Skippy runtime/header dependency for `canvas_worker.cpp`; not vendored here.

Model publisher names and exact revisions are recorded in PROVENANCE.md. Review
those publishers' model cards and licenses before downloading, running or
redistributing their artifacts. This source release makes no blanket clearance
claim for third-party redistribution.
