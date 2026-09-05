# Architecture and correctness constraints

FreeToken starts one process per RTX 5090. It presents an EP2 expert communicator,
but Qwen's dense backbone remains TP1 on rank 0. Rank 0 owns experts 0–255 and rank 1
owns 256–511. Both ranks stream their owned FP8 expert banks from host RAM into
independent GPU caches.

For each expert tile, rank 1 packs only its selected BF16 route outputs in canonical
token/top-k order and sends them directly to rank 0. Rank 0 inserts those routes into
the complete ten-route tensor and performs TP1's ordered FP32 accumulation followed by
one BF16 store. The route ordering and reduction boundary are mandatory: independently
rounded rank-local BF16 subtotals produced locally plausible math but diverged during
long autoregressive generation.

The worker send is queued on a dedicated CUDA stream. This overlaps part of tile N's
P2P transfer with tile N+1's expert computation. A stream dependency at layer exit
preserves communicator order before the next layer's broadcasts. The measured prefill
gain is 0.5–0.9%; this is useful but should not be described as eliminating route
communication.

Single-sequence GDN recurrence and the exact ten-route working tensor are bounded to
8K tiles while each expert layer stays materialized once. Even with those inner tiles,
a 25K outer scheduler chunk can exhaust rank 0 after production caches and vision are
warm. The production scheduler ceiling is therefore 16K. Longer prompts use multiple
scheduler passes; this does not reduce the 262,144-token KV pool. PLE also avoids a
redundant single-request identity gather.

Direct P2P requires the separately maintained source-patched NVIDIA open driver. The
accepted build enabled byte-correct 5090-to-5090 copies at about 28.2 GB/s one-way and
55.7 GB/s bidirectional. NCCL all-to-all at 256 MiB improved from 5.41 GB/s host-staged
to 47.67 GB/s. The RTX 4080 pairings remain CUDA-ineligible for peer access and are
outside the EP2 transport topology. Multimodal serving deliberately assigns the
native Qwen vision encoder and projection path to that 4080; it does not participate
in expert routing.

After any driver, firmware, BIOS, GPU-slot, or kernel change, revalidate peer access
and byte correctness before serving the model. Keep a TTY-tested rollback procedure;
do not enable global static BAR1 options as part of this deployment.
