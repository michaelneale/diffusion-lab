# Running the LLaDA split across two sites

The head (driver plus the `full` and `head` workers) runs on one machine and the
`tail` worker runs on another, possibly at another site behind NAT. The driver
reaches the tail over a [dumbpipe](https://github.com/n0-computer/dumbpipe)
tunnel: iroh QUIC with hole punching and relay fallback.

```
head:  canvas_experiment.py ──stdio── dumbpipe connect <ticket>
                                         │ iroh QUIC (direct or relayed)
tail:  dumbpipe listen ──stdio── canvas_worker tail   (relisten after each connection)
```

Both sides share a per-run `IROH_SECRET`, so the head derives the ticket with
`dumbpipe generate-ticket` and no addresses need exchanging; unset
`IROH_SECRET` before `connect`. Point the driver at the tunnel with
`DIFFUSION_REMOTE_CMD="dumbpipe connect <ticket>"`.

`fair_wan.py` runs the fair comparison from mesh-llm #3124: d3LLM_LLaDA Q4_K_M,
prefix stop, thresholds 0.8/0.6/0.4 split and 0.6 local, over the tasks and
checks in `fair_tasks.py`. Run the same tasks against a Skippy split for comparison.

`canvas_worker` is the original research build (mesh `1ea1ecf7`, Skippy ABI
0.1.79, macOS arm64), published as the `jy-wan-bundle-1` release asset
`canvas-worker-darwin-arm64-mesh1ea1ecf7.tar.gz`, SHA256
`92eb7394dd67f036f88e4a297063c45be209622d65a2fd76e4e8a1974e43c81f`.
It is a research binary, not a supported release.
