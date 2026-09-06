# Operations checklist

## Before starting

- Confirm no other model owns GPU0/GPU1, vision GPU2, or port 8080.
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
- Scheduler chunk: 16,384 tokens (longer prompts are split; context remains 262,144)
- GDN and route working tiles: 8,192 tokens
- Expert caches: 2,048/4,096 slots
- PLE: CPU RAM
- Route wire: exact BF16
- MTP: off
- RTX 4080: native Qwen vision encoder/projection
- Prompt cache: hybrid-radix with cache accounting enabled
- MoE prefill: resident expert rows reused device-to-device (CUDA 13)

Long multimodal decoder prompts are chunked at the same 16,384-token boundary. Each
pass receives only the vision features for image placeholders in that token slice.
Cross-request multimodal prefix reuse remains disabled because placeholder token IDs
do not identify image content; intermediate chunks still retain their request-local KV
state. Do not set the scheduler chunk to the 262,144-token KV capacity.

Keep the same 16,384-token ceiling for EP2 production and TP1 fallback. A cold 25K EP2
continuation OOMed rank 0 in the GDN state workspace after production caches were warm;
the 16K ceiling passed repeated cold 25K/32K probes and a post-vision 25K probe. A 32K
TP1 tile is also not accepted on a 32 GiB 5090.

## Service boundaries

Stopping this model must not be interpreted as stopping Open WebUI, its ingress, or
Cloudflare Tunnel. Those portal services have independent lifecycles. Before starting
another model, stop this service cleanly and wait until its GPU processes release
memory and port 8080 is free.

The repository's user-service template is intentionally not enabled automatically.
On the original lab host, the production system service uses a persistent drop-in;
do not rely on a `/run/systemd/system` override because it disappears at reboot.

## Rollback

First set `EP2_PACKED_ASYNC_SEND=0` to disable only the latest communication overlap.
Set `EP2_MOE_PREFILL_HIT_D2D=0` to disable prefill expert-row reuse independently.
Keep the 16K scheduler ceiling during feature rollbacks; it is a capacity guard, not a
feature toggle. For a complete rollback, stop the EP2 service, remove or disable its
persistent systemd drop-in, run `systemctl daemon-reload`, and start the preserved 27B
or FreeToken TP1 unit. Do not change portal or tunnel services during a model rollback.
