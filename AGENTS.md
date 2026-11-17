# Research harness conventions

Keep changes small and preserve measurement semantics. Do not claim WAN execution
from local perturbation or modeled network costs. Never commit model weights,
credentials, machine-specific addresses, copied runtime sources, or raw lab logs.

Run the complete offline suite with `python3 -m unittest discover -s tests -v`.
It needs NumPy, not MLX or a model. Inference and native rebuilds are separate
checks; report explicitly when they have not been rerun.

Do not run generated-code validators on unreviewed inputs: their AST filters are
not a security sandbox. Do not use Python's `-O` (assertions are research guards).
