# Running the LLaDA split through JY

A JY formation places the head (driver, `full` and `head` workers) on one lab
machine and the `tail` worker on another, possibly in another site behind NAT.
Members have no SSH to each other, so the driver reaches the tail worker over a
[dumbpipe](https://github.com/n0-computer/dumbpipe) tunnel: iroh QUIC with hole
punching and relay fallback, the same connection stack mesh-llm uses.

```
head:  canvas_experiment.py ──stdio── dumbpipe connect <ticket>
                                         │ iroh QUIC (direct or relayed)
tail:  dumbpipe listen-tcp ──TCP 127.0.0.1── worker_server.py ──stdio── canvas_worker tail
```

Both members derive the ticket from a per-run `IROH_SECRET`, so no address
exchange is needed. The driver selects the tunnel with
`DIFFUSION_REMOTE_CMD="dumbpipe connect <ticket>"`.

`canvas_worker` is the original research build (mesh `1ea1ecf7`, Skippy ABI
0.1.79, macOS arm64), shipped as the `jy-wan-bundle-1` release asset
`canvas-worker-darwin-arm64-mesh1ea1ecf7.tar.gz`, SHA256
`92eb7394dd67f036f88e4a297063c45be209622d65a2fd76e4e8a1974e43c81f`.
It is a research binary, not a supported release.
