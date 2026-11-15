# Kernel work: PyTorch baseline (15 Nov 2026)

**Relation to `exp01_repro`.** This folder was built on 15 Nov without reference to the graph-captured certified step in `exp01_repro` (`l4_graph.py`, 14 Nov). That step also uses one round per stage and device-side compaction. The two differ in these ways:
- **Rules:** here greedy and Gumbel; there greedy only.
- **Batching:** here each state has its own row list; there the batch reads the union of all rows.
- **First pass:** here it runs on tensor cores and needs a wider slack (it assumes tensor-core adds may truncate); there, see `exp01_repro/README.md`.
- **Sarvam-30B:** not covered here, so there is no K2 correction and no overflow fallback.

Measured on different days and containers, batch-1 graphed steps on the L4 are:

| Model | This folder | exp01_repro |
|---|---|---|
| Qwen3-8B | 1.78 ms | 2.04 ms |
| Gemma 4 E4B | 1.84 ms | 2.19 ms |

That is 0.35 and 0.34 of the BF16 head here, against 0.41 there. Neither changes E1's finding: on an A100 the certified head was not faster than the stock head, and the end-to-end step was unchanged. Whether the tensor-core first pass helps on the A100 is untested.

| File | What |
|---|---|
| `cert_head.py` | Three-pass certified head in plain PyTorch: dithered INT4 over every row, INT8 on the rows the rule still needs, BF16 on the rest. Weights stored as a kernel would read them (INT4 packed two per byte, FP16 scales). Greedy and Gumbel rules. |
| `test_cert_head.py` | CPU test: weights equal `exp01_repro/cert_core.py` bit for bit; certified tokens equal the FP32 reference (batch 1, 8, 64; with and without soft-cap). |
| `baseline_modal.py` | Times it against the plain BF16 head on 512 saved decode states per rule, on a Modal L4, and checks every token against the FP32 reference. |
| `results/baseline_<model>_L4.json`, `logs/` | Results and logs. |
| `int4_pass1.py` | Triton kernels for pass 1 (INT4 product and bound): `tc` (tensor cores, any batch up to 64) and `fp32` (batch of one, CUDA cores). |
| `fused.py`, `fused_modal.py` | All three passes on the GPU (four kernels, per-state row lists, no host syncs) and a CUDA-graph wrapper; checked and timed. `results/fused_<model>_L4.json`. |
| `pass1_modal.py`, `common.py` | Checks the kernels against the PyTorch pass 1, runs the certified head with the kernel on 512 states per rule, and times it. `results/pass1_<model>_L4.json`. |

Method: torch 2.15.0, eager mode, TF32 off. Each stage is timed with a device sync around it; the median over 128 batches (batch 1), 32 (batch 8) or 16 (batch 32) after 3 warm-up batches. States are shuffled, so batches mix languages. Within a batch the INT8 and BF16 passes read the union of the rows any state needs. Gumbel noise is generated once and left out of both timings (see below).

## Results, L4, milliseconds (median per batch)

| Model | Rule | Batch | BF16 head | Certified, as is | Pass 1 (INT4) | Bound | INT8 pass | BF16 pass | INT8 rows | BF16 rows | Pass 1 at cuBLAS bandwidth | Certified, projected |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Gemma 4 E4B | Gumbel T=1 | 1 | 5.36 | 113.7 | 109.8 | 0.50 | 1.80 | 1.35 | 16 | 1 | 1.39 | **5.05** |
| Gemma 4 E4B | Gumbel T=1 | 8 | 5.74 | 117.8 | 112.0 | 0.54 | 3.86 | 1.34 | 12,706 | 12 | 1.44 | 7.18 |
| Gemma 4 E4B | Gumbel T=1 | 32 | 7.38 | 135.4 | 112.2 | 2.29 | 13.30 | 7.38 | 26,634 | 51 | 1.58 | 24.55 |
| Qwen3-8B | Gumbel T=1 | 1 | 4.91 | 103.0 | 101.1 | 0.21 | 0.89 | 0.73 | 3 | 1 | 1.29 | **3.12** |
| Qwen3-8B | Gumbel T=1 | 8 | 5.16 | 105.3 | 103.2 | 0.28 | 0.92 | 0.73 | 44 | 13 | 1.31 | 3.24 |
| Qwen3-8B | Gumbel T=1 | 32 | 5.81 | 109.8 | 103.7 | 0.82 | 2.69 | 2.50 | 306 | 46 | 1.44 | 7.45 |

Greedy is within a few percent of Gumbel T=1 everywhere (`results/*.json`). "Pass 1 at cuBLAS bandwidth" is the INT4 pass's bytes (0.26 of the BF16 head) divided by the bandwidth the BF16 product reached (about 250 GB/s at batch 1, 220 at batch 32; the L4's peak is 300). "Certified, projected" swaps that number in for the measured pass 1 and keeps everything else as measured.

## What it says

1. **Exactness holds in the timed path.** Zero certified mismatches on every model, rule and batch size. The plain BF16 head disagrees with the FP32 reference on 1 to 3 states in each cell, because its logits are rounded to BF16. "Exact" in the paper means exact against the FP32-accumulated reference, not against a stock BF16 head.
2. **Pass 1 needs a kernel.** In PyTorch it takes 100 to 112 ms, 20 times the BF16 head, because it unpacks and dequantises to FP32 in memory. Expected.
3. **Pass 1 alone is not enough.** At batch 1, even with pass 1 at cuBLAS bandwidth, Gemma 4 only goes from 5.36 to 5.05 ms. The INT8 and BF16 passes read 16 and 1 rows (about 40 KB), yet take 3.2 ms: kernel launches, syncs on data-dependent row lists, and the rule's elementwise passes over all 262K logits. Qwen3-8B fares better (4.91 to 3.12 ms, 1.6 times faster) because its later passes take 1.6 ms.
4. **The union of rows breaks batching on Gemma 4.** One Tamil state needs about 5,000 INT8 rows (median at batch 1), so a batch of 8 reads 12,706 rows and a batch of 32 reads 26,634, and the certified path loses to BF16 from batch 8 on. Each state needs its own shortlist. Qwen3-8B's shortlists stay small (306 rows at batch 32).
5. **Noise generation is slow.** Counter-based Gumbel noise in PyTorch takes 0.7 ms at batch 1 and 23 ms at batch 32 on Gemma 4. It is left out of both sides here, but a real sampler must generate it inside the kernel, only for the rows it needs.

## Targets for the kernel

- **Pass 1:** the INT4 product, the bound and the rule's first test (the running maximum of lower ends) in one pass over the packed weights, at close to cuBLAS bandwidth: about 1.4 ms on Gemma 4 and 1.3 ms on Qwen3-8B at batch 1.
- **Passes 2 and 3:** per-state row lists, gathered and computed in one or two launches without host syncs; target well under 0.5 ms at batch 1.
- If both are met, batch 1 on the L4 would go from 5.4 to about 2 ms on Gemma 4 and from 4.9 to about 1.8 ms on Qwen3-8B. That is the head alone, not tokens per second for the whole model.

Not covered: top-k and min-p, other GPUs (T4, RTX 4090), CUDA graphs or `torch.compile` for the later passes, and Sarvam-30B.

# Pass-1 kernel (15 Nov 2026)

`int4_pass1.py`, Triton 3.9.0, L4. Three versions were tried; the logs keep the first two (`logs/int4_pass1_v1.py`, `_v2.py`; `results/pass1_v1_*`, `pass1_v2_*`).

- **v1:** FP32 products on CUDA cores; `tl.dot` with `input_precision="ieee"` for batches. Batch 1: 1.88 ms on Qwen3-8B. Batches were very slow: 15.7 ms at 8 and 30 ms at 32.
- **v2:** batch tiles with a 3-D broadcast. Worse everywhere: 2.4 ms at batch 1, 54 ms at batch 32.
- **v3 (current):** two kernels.
  - **`tc`:** for each 128-weight group, a BF16 tensor-core dot product of the integer codes q with h. Both are exact in BF16, so every product is exact. Then subtract u_g·h_g and scale by s_g in FP32.
  - **`fp32`:** v1's batch-1 kernel, kept for comparison.

## Correctness

- **Logits.** Both kernels' logits differ from the PyTorch FP32 pass 1 by at most 0.04% of cert_core's FP32 slack, on real states at batch 1, 8 and 32.
- **Tokens.** With the `tc` kernel as pass 1, the certified head gave the FP32 reference token on every state: 896 states per model, greedy and Gumbel T = 1, batch 1, 8 and 32.
- **Slack for `tc`.** The kernel does its arithmetic in a different order, so its slack must also cover tensor-core addition, which may truncate (Fasi et al., 2021). It uses unit roundoff 2u over 2d operations and adds a term for the dither: `gamma2 (||h|| ||w~|| + 2 sum_g s_g ||u_g|| ||h_g||)`.
  - This makes the bound about twice cert_core's slack larger. For most rows that is negligible. On rows whose bound is mostly slack, it is up to 1.9% (Gemma 4) and 4.3% (Qwen3-8B) larger.
  - The soundness of this rests on that model of tensor-core rounding. The paper must state it, or we test it with planted cases (C6-f style) on each GPU.
- **`fp32` kernel bound.** It differs from PyTorch's by about 3e-7 relative, sometimes below it. That is the rounding in the bound's own computation, the gap already noted in exp01_repro.

## Timing, L4, milliseconds (median)

| Model | Batch | BF16 head (cuBLAS) | Pass 1 `tc` kernel | GB/s | Certified head, kernel + PyTorch passes 2–3 (Gumbel T=1) |
|---|---|---|---|---|---|
| Qwen3-8B | 1 | 4.94 | 2.06 | 156 | 4.84 (2.19 / 1.35 / 1.05) |
| Qwen3-8B | 8 | 5.10 | 2.16 | 149 | 5.24 |
| Qwen3-8B | 32 | 5.57 | 2.39 | 135 | 9.00 (2.58 / 3.13 / 2.74) |
| Gemma 4 E4B | 1 | 5.33 | 2.16 | 161 | 5.40 (2.27 / 1.56 / 1.31) |
| Gemma 4 E4B | 8 | 5.52 | 2.26 | 154 | 8.08 (2.49 / 3.95 / 1.37) |
| Gemma 4 E4B | 32 | 6.08 | 2.61 | 133 | 24.72 (2.79 / 13.64 / 7.29) |

Batch-1 `fp32` kernel: 1.89 ms (Qwen3-8B), 1.98 ms (Gemma 4).

What it says:
1. **Pass 1 is now 2.3–2.6 times faster than the BF16 head at every batch size**, but at 130–160 GB/s against cuBLAS's 220–250. Reaching cuBLAS bandwidth would take it to about 1.3–1.4 ms.
2. **The certified head as a whole is still no faster.** It roughly breaks even at batch 1, because PyTorch's INT8 and BF16 passes cost 2.4–2.9 ms, which is more than pass 1 saves.
3. **Gemma 4 batches are dominated by the union of rows** (13.6 ms for the INT8 pass at batch 32), as the baseline predicted.

## Next

- **Passes 2 and 3 as kernels:** per-state row lists, no host syncs, and the rule test fused into pass 1 (running maximum of lower ends). This is now the larger cost.
- **Pass-1 bandwidth:** from about 155 to above 220 GB/s. Options include wider loads per row, a split-K for d = 2560, and the scale loads.

# Fused passes 2–3 (15 Nov 2026)

`fused.py`. Greedy and Gumbel only. The noise is an input, generated outside the timed region for both sides.

**How it works.** A row is needed while its upper score can reach the largest lower score of any row. That maximum only grows as intervals tighten, so one INT8 round and one exact round already reach the stable state that cert_core's loop reaches. The head therefore runs as a fixed sequence:

1. **K1 (pass 1):** the INT4 tensor-core product and bound for every row, plus the largest lower score per row block.
2. **K2 (select):** rows whose upper score reaches that maximum are appended to each state's own list, using an atomic counter.
3. **K3 (INT8):** listed rows only. Computes the INT8 product and bound, intersects the intervals, and records the new largest lower score.
4. **K4 (exact):** listed rows still needed. Computes the BF16 row × h in FP32, then the best (score, row) per program.
5. **Torch reductions:** three small ones pick the token. Ties go to the lowest row index, as with `torch.argmax`.

Nothing waits on the CPU, so the whole sequence is captured as a CUDA graph.

## Correctness

- **Synthetic head:** zero mismatches against cert_core's reference. This covers soft-cap on and off; greedy, Gumbel T = 1 and T = 0.6; batch 1, 3, 17 and 64; eager and graphed.
- **Real heads:** zero mismatches on 896 states per rule and model, eager and graphed. The plain BF16 head disagrees with the FP32 reference on 1–3 states per cell, as before.
- **Soft-cap:** the kernels use libdevice `tanh`, not torch's. The tokens still matched everywhere, but a near-exact tie between two rows could in principle resolve differently. In the same spirit, Triton contracts multiply-adds into FMAs by default (`TRITON_DEFAULT_FP_FUSION`). FMAs only reduce rounding error, so the slack still covers them.

## Timing, L4, milliseconds (median per batch, CUDA events)

| Model | Rule | Batch | BF16 head, graphed | Fused certified head, graphed | Speed-up | Fused, eager | INT8 rows per state, median / p99 |
|---|---|---|---|---|---|---|---|
| Qwen3-8B | Gumbel T=1 | 1 | 5.05 | **1.78** | 2.8× | 2.50 | 3 / 548 |
| Qwen3-8B | Gumbel T=1 | 8 | 5.20 | **1.98** | 2.6× | 2.68 | 3 / 541 |
| Qwen3-8B | Gumbel T=1 | 32 | 6.02 | **2.52** | 2.4× | 3.15 | 4 / 419 |
| Gemma 4 E4B | Gumbel T=1 | 1 | 5.40 | **1.84** | 2.9× | 2.58 | 17 / 17,201 |
| Gemma 4 E4B | Gumbel T=1 | 8 | 5.73 | **2.44** | 2.3× | 3.19 | 25 / 21,780 |
| Gemma 4 E4B | Gumbel T=1 | 32 | 7.55 | **5.24** | 1.4× | 5.74 | 34 / 23,188 |

Greedy is within about 0.1 ms of Gumbel at every size (`results/fused_*.json`).

What it says:
1. **The certified head is now faster than the BF16 head at every batch size tested,** on both models, with exact tokens. At batch 1 it is about 2.8 times faster. These are times for the head alone, not for whole tokens.
2. **Per-state lists fixed most of the Gemma 4 batching problem:** 24.7 ms at batch 32 before, 5.2 ms now. What remains is imbalance. At batch 32 each state gets 8 INT8 programs, so a state with 23,000 rows (an Indian-language state, p99) holds up the batch.
3. **CUDA graphs save about 0.6–0.7 ms per call.** That is the launch and Python overhead that remained after the syncs were removed.

## Next

- **Balance the INT8 work across states** with one shared queue of (state, row) work instead of a fixed number of programs per state. This is the Gemma 4 batch-32 gap.
- **Pass-1 bandwidth** (about 155 GB/s against cuBLAS's 220–250). It now dominates batch-1 time.
- **Gumbel noise inside the kernels**, generated only for the rows that need it. It is excluded from both sides here, so the timings above do not include it.
- **Top-k and min-p; T4 and RTX 4090; whole-model tokens per second.**
