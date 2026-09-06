# Qwen3.8 Flash Next on FreeToken

Reproducible production profile for the official Qwen3.8-Flash-Next FP8 checkpoint
on two RTX 5090s, with multimodal vision preprocessing on an RTX 4080. The Qwen
backbone stays whole on GPU0, while the 512 routed experts are split 256/256 across
GPU0 and GPU1. The 47.7 GiB PLE table and streamed expert banks live in host RAM.

This repository contains deployment configuration, operational checks, and accepted
benchmark evidence. It does **not** contain or automatically download model weights,
driver binaries, credentials, or Open WebUI data.

## Pinned stack

- Model: `Qwen/Qwen3.8-Flash-Next-FP8`
- Model revision: `236dfdf285828023ca3bcd3f37366c58a3469b13`
- FreeToken fork: <https://github.com/Enigmatic331/FreeToken/tree/qwen38-ep2>
- FreeToken revision: `0742097e8c2d96422beaea7d2f1fc44aa5fde38b`
- P2P driver source: <https://github.com/Enigmatic331/open-gpu-kernel-modules/tree/610.43.02-p2p-qwen-lab>
- P2P driver revision: `64b8c7ed55ab9c5a34717380fd1f5048b1d7218d`

The custom driver is specific to this lab. Do not install it blindly; read
[`docs/architecture.md`](docs/architecture.md) and keep a tested TTY rollback path.

## Accepted topology

```text
GPU0 RTX 5090: backbone, shared experts, routed experts 0..255, 2,048 cache slots
GPU1 RTX 5090: routed experts 256..511, 4,096 cache slots
CPU RAM:       47.7 GiB PLE plus both streamed FP8 expert banks
GPU2 RTX 4080: native Qwen vision encoder/projection only
Transport:     exact packed BF16 routes over direct 5090-to-5090 CUDA P2P
KV:            full 262,144-token hybrid-radix pool; MTP off
Prefill:       cache-resident expert rows reused device-to-device
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

These historical 32K-chunk performance rows were measured on revision `15e3a7b`.
Production now uses a 16,384-token scheduler ceiling: longer prompts retain the same
full context but are split into memory-safe scheduler passes. Fresh-prefix production
probes at the current revision completed 25,031 tokens at 6.66 s TTFT and 32,731 tokens
at 8.09–8.10 s TTFT without increasing steady-state VRAM. The final row reaches the exact
262,144-token pool ceiling after generation. Peak
residency was 32,076 MiB on GPU0 and 29,860 MiB on GPU1, process-tree RSS was about
171.4 GiB, and measured disk reads were zero. The miss-heavy pagoda replay decodes at
about 45 tok/s; synthetic prompts are substantially more cache-friendly.

Multimodal decoder prefills use the same chunking rule. An Open WebUI request with a
real image, its production tool schema, and 24,379 decoder tokens completed as 16,384
and 7,995-token scheduler passes with a correct image-grounded answer. The vision
features are sliced to the image placeholders in each pass; the scheduler does not
raise its physical prefill ceiling to the full KV capacity.

Correctness was accepted only after 27 targeted CUDA/model tests, 12/12 deterministic
semantic probes, three 10,762-token tiled retrievals, and a byte-identical 1,023-token
TP1 pagoda trace (`6c6ae9e08ecd1bc6714873b807d87edfd7176239`).

### Final one-GPU control

The same commit was also gated as true TP1 with a 2,048-slot cache and full KV. A 32K
single scheduler chunk OOMed on a 160 MiB expert allocation, so the accepted TP1
ceiling is 16K. It still reaches the full context by scheduling multiple chunks.

| Prompt | TP1 prefill | EP2 prefill | TP1 decode | EP2 decode |
| ---: | ---: | ---: | ---: | ---: |
| 16,384 | 3,728.29 | 3,893.79 | 73.90 | 82.95 |
| 32,768 | 3,742.94 | 4,760.89 | 67.56 | 81.87 |
| 261,888 | 3,720.85 | 4,454.01 | 60.24 | 74.88 |

EP2's advantage grows with prompt length: +27.2% prefill and +21.2% decode at 32K,
and +19.7%/+24.3% at the pool ceiling. On the miss-heavy pagoda replay, EP2 is about
89.7% faster (44.95 versus 23.70 tok/s). This is the practical value of the second
5090 beyond merely fitting the workload.

## Deploy

1. Clone the pinned FreeToken fork and check out the revision above.
2. Install FreeToken using its upstream instructions and keep the native extension
   from the validated build.
3. Place all 131 model shards locally. The launcher refuses incomplete checkpoints.
4. Copy `.env.example` to a host-only configuration file and edit its paths.
5. Run `scripts/preflight.sh`, then `scripts/run.sh`.
6. Run `scripts/smoke-test.sh` before exposing the endpoint to Open WebUI.

The launcher defaults to `172.17.0.1:8080` so a Dockerized Open WebUI can reach it
without exposing the model API to the LAN. Review firewall rules for your host.

For a user service, copy `systemd/qwen38-flash-freetoken.service` into
`~/.config/systemd/user/`, adjust its two paths, reload the user service manager, and
start it. It is intentionally not enabled automatically.

## Rollback controls

- Disable only the new send overlap: `EP2_PACKED_ASYNC_SEND=0`
- Disable prefill expert-row reuse: `EP2_MOE_PREFILL_HIT_D2D=0`
- Keep `EP2_MAX_PREFILL_LENGTH=16384` for the memory-safe production geometry. It does
  not reduce the 262,144-token context; it only splits longer prefills into more passes.
- Stop this model before starting another service that owns GPU0/GPU1, vision GPU2,
  or port 8080.

See [`docs/operations.md`](docs/operations.md) for the production checklist and
[`results/accepted.csv`](results/accepted.csv) for raw accepted rows.
