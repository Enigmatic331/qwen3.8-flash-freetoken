# Operations checklist

## Before starting

- Confirm no other model owns GPU0/GPU1 or port 1919.
- Confirm the FreeToken and model revisions match `freetoken.lock`.
- Confirm all 131 checkpoint shards are present.
- Run `scripts/preflight.sh`.
- After driver/platform changes, rerun the separate checked CUDA P2P validation.

## Acceptance gate

1. Health endpoint responds.
2. Greedy short probes are deterministic and free of NaNs, repetition, or token-0 loops.
3. Run a retrieval prompt longer than both 8K working-tile boundaries.
4. Run the known long autoregressive/code fixture and compare to its accepted TP1 output.
5. Only then accept prefill/decode measurements.
6. For capacity changes, prove prompt plus output reaches the intended KV ceiling.

Performance without these gates is not an accepted result.

## Runtime defaults

- Batch/concurrency: one
- KV/context: 262,144 tokens
- Scheduler chunk: 32,768 tokens
- GDN and route working tiles: 8,192 tokens
- Expert caches: 2,048/4,096 slots
- PLE: CPU RAM
- Route wire: exact BF16
- MTP: off
- RTX 4080: unused

For a TP1 fallback with full KV, use a 2,048-slot expert cache and a 16,384-token
scheduler ceiling. A 32K TP1 tile is not accepted on a 32 GiB 5090; it OOMed with only
about 91 MiB free when requesting another 160 MiB expert buffer.

## Service boundaries

Stopping this model must not be interpreted as stopping Open WebUI, its ingress, or
Cloudflare Tunnel. Those portal services have independent lifecycles. Before starting
another model, stop this service cleanly and wait until its GPU processes release
memory and port 1919 is free.

The service template is intentionally not enabled at boot. On the original lab host,
the pre-existing multimodal 27B service remains the safe boot fallback.

## Rollback

First set `EP2_PACKED_ASYNC_SEND=0` to disable only the latest communication overlap.
If needed, restore the earlier 16K scheduler/untiled geometry using the variables in
the README. For a complete rollback, stop the EP2 service and start the preserved 27B
or FreeToken TP1 unit. Do not change portal or tunnel services during a model rollback.
