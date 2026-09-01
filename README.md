# Qwen3.8 Flash Next on FreeToken

Reproducible production profile for the official Qwen3.8-Flash-Next FP8 checkpoint
on two RTX 5090s. The Qwen backbone stays whole on GPU0, while the 512 routed experts
are split 256/256 across GPU0 and GPU1. The 47.7 GiB PLE table and streamed expert
banks live in host RAM. GPU2 is intentionally unused.

This repository contains deployment configuration, operational checks, and accepted
benchmark evidence. It does **not** contain or automatically download model weights,
driver binaries, credentials, or Open WebUI data.

## Pinned stack

- Model: `Qwen/Qwen3.8-Flash-Next-FP8`
- Model revision: `236dfdf285828023ca3bcd3f37366c58a3469b13`
- FreeToken fork: <https://github.com/Enigmatic331/FreeToken/tree/qwen38-ep2>
- FreeToken revision: `15e3a7b9b626d739f73419af1d381275d964005e`
- P2P driver source: <https://github.com/Enigmatic331/open-gpu-kernel-modules/tree/610.43.02-p2p-qwen-lab>
- P2P driver revision: `64b8c7ed55ab9c5a34717380fd1f5048b1d7218d`

The custom driver is specific to this lab. Do not install it blindly; read
[`docs/architecture.md`](docs/architecture.md) and keep a tested TTY rollback path.

## Accepted topology

```text
GPU0 RTX 5090: backbone, shared experts, routed experts 0..255, 2,048 cache slots
GPU1 RTX 5090: routed experts 256..511, 4,096 cache slots
CPU RAM:       47.7 GiB PLE plus both streamed FP8 expert banks
GPU2 RTX 4080: unused by the model
Transport:     exact packed BF16 routes over direct 5090-to-5090 CUDA P2P
KV:            full 262,144-token pool; MTP off
```

The exact ordered route reducer is a correctness requirement. Earlier rank-local BF16
subtotals were faster-looking but changed long autoregressive generations. The current
path retains TP1's canonical top-k accumulation order.

## Measured performance

One warmup plus three warm 255-token measured generations, batch one:

| Prompt | Prefill tok/s | Decode tok/s | TTFT |
| ---: | ---: | ---: | ---: |
| 16,384 | 3,893.79 | 82.95 | 4.208 s |
| 32,768 | 4,760.89 | 81.87 | 6.883 s |
| 261,888 | 4,454.01 | 74.88 | 58.798 s |

The final row reaches the exact 262,144-token pool ceiling after generation. Peak
residency was 32,076 MiB on GPU0 and 29,860 MiB on GPU1, process-tree RSS was about
171.4 GiB, and measured disk reads were zero. The miss-heavy pagoda replay decodes at
about 45 tok/s; synthetic prompts are substantially more cache-friendly.

Correctness was accepted only after 27 targeted CUDA/model tests, 12/12 deterministic
semantic probes, three 10,762-token tiled retrievals, and a byte-identical 1,023-token
TP1 pagoda trace (`6c6ae9e08ecd1bc6714873b807d87edfd7176239`).

## Deploy

1. Clone the pinned FreeToken fork and check out the revision above.
2. Install FreeToken using its upstream instructions and keep the native extension
   from the validated build.
3. Place all 131 model shards locally. The launcher refuses incomplete checkpoints.
4. Copy `.env.example` to a host-only configuration file and edit its paths.
5. Run `scripts/preflight.sh`, then `scripts/run.sh`.
6. Run `scripts/smoke-test.sh` before exposing the endpoint to Open WebUI.

The launcher defaults to `172.17.0.1:1919` so a Dockerized Open WebUI can reach it
without exposing the model API to the LAN. Review firewall rules for your host.

For a user service, copy `systemd/qwen38-flash-freetoken.service` into
`~/.config/systemd/user/`, adjust its two paths, reload the user service manager, and
start it. It is intentionally not enabled automatically.

## Rollback controls

- Disable only the new send overlap: `EP2_PACKED_ASYNC_SEND=0`
- Restore the earlier 16K scheduler geometry:
  `EP2_MAX_PREFILL_LENGTH=16384 EP2_GDN_PREFILL_TILE_TOKENS=0 EP2_PREFILL_ROUTE_TILE_TOKENS=0`
- Stop this model before starting another service that owns GPU0 or port 1919.

See [`docs/operations.md`](docs/operations.md) for the production checklist and
[`results/accepted.csv`](results/accepted.csv) for raw accepted rows.
