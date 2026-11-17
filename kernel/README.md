# Kernel work: PyTorch baseline (15 Nov 2026)

**Relation to `exp01_repro`.** This folder was built on 15 Nov without reference to the graph-captured certified step in `exp01_repro` (`l4_graph.py`, 14 Nov). That step also uses one round per stage and device-side compaction. The two differ in these ways:
- **Rules:** here greedy and Gumbel; there greedy only.
- **Batching:** here each state has its own row list; there the batch reads the union of all rows.
- **First pass:** here it runs on tensor cores and needs a wider slack (it assumes tensor-core adds may truncate); there, see `exp01_repro/README.md`.
- **Bound:** here the Hoeffding bound with no mean-row shift; there the sub-Gaussian bound (0.577 of Hoeffding) at the B1 shifts. The certificate here is the wider one, so its shortlists are longer than they need to be.
- **Sarvam-30B:** not covered here, so there is no K2 correction and no overflow fallback.

Measured on different days and containers, and with the differences above (so not like for like), batch-1 graphed steps on the L4 are:

| Model | This folder | exp01_repro |
|---|---|---|
| Qwen3-8B | 1.78 ms | 2.04 ms |
| Gemma 4 E4B | 1.84 ms | 2.19 ms |

That is 0.35 and 0.34 of the BF16 head here, against 0.41 there. Neither changes E1's finding: on an A100 the certified head was not faster than the stock head, and the end-to-end step was unchanged. Whether the tensor-core first pass helps on the A100 is untested.

## Review corrections (15 Nov 2026, after the first commit of this folder)

A review of this folder found the following. The sections below are corrected where they said otherwise; the results files are unchanged.

1. **Every time here is a median, and the tail is unmeasured.** The runs kept only the median time per batch. Shortlist lengths have a long tail: on Gemma 4, greedy, batch 1, the median state needs 29 INT8 rows, the p99 state 26,209 and the worst 46,689 (Qwen3-8B: 3, 93 and 1,364). So "faster than the BF16 head" is shown for the median batch only. A rerun that saves p99 and mean times is owed.
2. **Exactness was checked on 512 distinct states per rule and model, not 896.** Batch 1 uses the first 128 of them, batch 8 the first 256 and batch 32 all 512, so 896 counts some states two or three times.
3. **The Gumbel timings assume the noise is already there.** Both sides get the noise as an input. Generating it in PyTorch took 0.73 ms at batch 1 and 23.1 ms at batch 32 on Gemma 4 (0.43 and 11.9 ms on Qwen3-8B; `results/baseline_*.json`). With those times added to both sides, the batch-1 speed-up would be 2.4 times on Gemma 4 and 2.5 on Qwen3-8B, and at batch 32 it would be 1.08 and 1.24. Two further points:
   - The first pass and the select step test every row with that row's own noise. So noise is needed for all V rows of every state, and "only for the rows that need it" is not possible in this design.
   - The comparison is against a Gumbel-max BF16 head, which token equality requires. A stock sampler (softmax and one uniform draw) needs no per-row noise at all.
4. **The first-pass rate of about 155 GB/s is an underestimate.** It was timed eagerly, with device syncs, and includes preparing the state and allocating the outputs. The whole graphed step takes 1.78 ms (Qwen3-8B) and 1.84 ms (Gemma 4), so the first pass inside it reads its 322 and 348 MB at 180 GB/s or more. At cuBLAS's 250 GB/s it would take 1.29 and 1.39 ms, so at most about 0.5 ms is left to gain at batch 1.
5. **The exact pass sums in its own order.** K4 computes BF16 row × h in FP32 in a different order from the reference's matrix product. If two rows both reach the exact pass and their reference scores differ by less than FP32 rounding, the kernel could pick the other one. No such case occurred. The same holds for the row-gather step in `exp01_repro`.
6. **No protocol was hashed for this folder,** and it arrived as one commit, so the targets below cannot be shown to predate the results. Treat them as notes, not predictions.

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
5. **Noise generation is slow.** Counter-based Gumbel noise in PyTorch takes 0.7 ms at batch 1 and 23 ms at batch 32 on Gemma 4. It is left out of both sides here, but a real sampler must generate it inside the kernel. (Corrected: this said "only for the rows it needs", which the rule does not allow; see review correction 3.)

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
- **Tokens.** With the `tc` kernel as pass 1, the certified head gave the FP32 reference token on every state: 512 distinct states per rule and model (greedy and Gumbel T = 1), checked at batch 1, 8 and 32.
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
1. **Pass 1 is now 2.3–2.6 times faster than the BF16 head at every batch size**, but at 130–160 GB/s against cuBLAS's 220–250. Reaching cuBLAS bandwidth would take it to about 1.3–1.4 ms. (These rates include eager overhead; the kernel itself is faster. See review correction 4.)
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
- **Real heads:** zero mismatches on 512 distinct states per rule and model, at batch 1, 8 and 32, eager and graphed. The plain BF16 head disagrees with the FP32 reference on 1–3 states per cell, as before.
- **Soft-cap:** the kernels use libdevice `tanh`, not torch's. The tokens still matched everywhere, but a near-exact tie between two rows could in principle resolve differently. In the same spirit, Triton contracts multiply-adds into FMAs by default (`TRITON_DEFAULT_FP_FUSION`). FMAs only reduce rounding error, so the slack still covers them. The exact pass also sums in its own order (review correction 5).

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
1. **For the median batch, the certified head is now faster than the BF16 head at every batch size tested,** on both models, with exact tokens. At batch 1 it is about 2.8 times faster. These are times for the head alone, not for whole tokens, with the Gumbel noise supplied, and the tail was not measured (review corrections 1 and 3).
2. **Per-state lists fixed most of the Gemma 4 batching problem:** 24.7 ms at batch 32 before, 5.2 ms now. What remains is imbalance. At batch 32 each state gets 8 INT8 programs, so a state with 23,000 rows (an Indian-language state, p99) holds up the batch.
3. **CUDA graphs save about 0.6–0.7 ms per call.** That is the launch and Python overhead that remained after the syncs were removed.

## Next

- **Balance the INT8 work across states** with one shared queue of (state, row) work instead of a fixed number of programs per state. This is the Gemma 4 batch-32 gap.
- **Tail times:** save p99 and mean per batch, not only the median. The harness is written and its protocol hashed (`tail_protocol.json`, `fused_tail_modal.py`, `tail_summarize.py`), but it has not run: two launches (15 and 16 Nov) were stopped before any container produced a result (`logs/fused_tail_killed_*.txt`).
- **The sub-Gaussian bound and the B1 shifts,** as in `exp01_repro`. They shorten the shortlists, which is what holds up Gemma 4 at batch 32.
- **Pass-1 bandwidth** (at least 180 GB/s inside the graphed step, against cuBLAS's 220–250). It dominates batch-1 time, with at most about 0.5 ms left to gain.
- **Gumbel noise inside the kernels.** Every row's test needs its own noise, so this means V draws per state inside the first pass. It is excluded from both sides here, so the timings above do not include it.
- **The first pass on an A100,** where E1 found the earlier kernel limited by its own arithmetic.
- **Top-k and min-p; T4 and RTX 4090; whole-model tokens per second.**

# The fused step on an A100-80GB (17 Nov 2026)

Protocol `a100_protocol.json` (sha256 `bf58f92b…`, logged before any kernel of this folder ran on an A100), an addendum to `tail_protocol.json`. Harness `fused_tail_modal.py --gpu A100-80GB`, unchanged kernels; Triton's autotuner picks the block sizes on the A100. Results `results/fused_tail_<model>_A100-80GB.json`, table `logs/fused_tail_summary_a100.txt`. Every batch is timed three times and the median of the three is used; batch 1 covers all 512 states, batch 8 has 64 batches, batch 32 has 16.

Controls pass: no certified token differs from the FP32 reference on any repeat, eager or graphed, and the row counts are identical across repeats.

Graph-captured, milliseconds. "p99" with 16 batches is the second largest.

| Model | Rule | Batch | BF16 head, median | Fused: median, p99, largest | Fused / BF16, median | Batches slower than the BF16 median |
|---|---|---|---|---|---|---|
| Qwen3-8B | greedy | 1 | 0.75 | 0.52, 0.52, 0.53 | **0.69** | 0 of 512 |
| Qwen3-8B | greedy | 8 | 0.77 | 0.57, 0.58, 0.62 | 0.74 | 0 of 64 |
| Qwen3-8B | greedy | 32 | 0.86 | 0.82, 0.89, 1.05 | 0.95 | 3 of 16 |
| Qwen3-8B | Gumbel T=1 | 1 | 0.76 | 0.52, 0.54, 0.54 | **0.68** | 0 of 512 |
| Qwen3-8B | Gumbel T=1 | 8 | 0.79 | 0.58, 0.61, 0.63 | 0.73 | 0 of 64 |
| Qwen3-8B | Gumbel T=1 | 32 | 0.91 | 0.85, 0.96, 1.02 | 0.93 | 5 of 16 |
| Gemma 4 E4B | greedy | 1 | 0.82 | 0.54, 0.66, 0.76 | **0.66** | 0 of 512 |
| Gemma 4 E4B | greedy | 8 | 0.87 | 0.95, 1.87, 2.05 | 1.09 | 46 of 64 |
| Gemma 4 E4B | greedy | 32 | 1.09 | 3.39, 5.95, 6.57 | 3.10 | 16 of 16 |
| Gemma 4 E4B | Gumbel T=1 | 1 | 0.83 | 0.55, 0.67, 0.86 | **0.67** | 1 of 512 |
| Gemma 4 E4B | Gumbel T=1 | 8 | 0.89 | 0.97, 1.84, 2.66 | 1.09 | 36 of 64 |
| Gemma 4 E4B | Gumbel T=1 | 32 | 1.19 | 3.25, 5.39, 8.97 | 2.74 | 16 of 16 |

What it says:
1. **At batch 1 the certified head is faster than the stock head on an A100:** 0.66 to 0.69 of its time on both models, and slower on 1 state of 2,048. The step in `exp01_repro` took 1.06 and 1.11 ms on the same GPU type (E1); this one takes 0.52 and 0.54. The tensor-core first pass keeps up with fast memory where that one did not.
2. **The tail at batch 1 is short.** On Gemma 4 a state with a 46,689-row shortlist takes 0.76 ms against a median of 0.54. So the long shortlists cost little when one state has all the INT8 programs.
3. **The win shrinks with batch size and is gone by 32.** The BF16 head barely grows with the batch (0.75 to 0.86 ms on Qwen3-8B): it reads the weights once. The fused step grows from 0.52 to 0.82 ms, because its per-state work (writing and re-reading a logit and a bound for every row of every state, the select pass, the reductions) is no longer small next to the weight read on fast memory.
4. **Gemma 4 loses from batch 8 on,** and by a factor of three at batch 32. A state gets only 8 INT8 programs at batch 32, so one long shortlist holds up the batch: 5.75 ms for batches with a shortlist above 30,000 rows. These shortlists come from the Hoeffding bound with no shift; the sub-Gaussian bound and the B1 shift of `exp01_repro` would shorten them, and a shared work queue would balance them.

What it changes and what it does not:
- **Changes:** E1's "on an A100 the certified head is not faster than the stock head" holds for `exp01_repro`'s kernels, not for this one at batch 1.
- **Does not change:** E1's end-to-end finding. The head is 1% to 4% of a decode step in the Hugging Face loop, so saving 0.25 ms of a 62 ms step is invisible there.
- **Still not shown:** a win at batch 32, on Sarvam-30B (not covered by this folder), with the noise generated inside the timed region, or with the sub-Gaussian bound.

Against the predictions:
- **Batch 1: missed, in our favour.** Predicted 0.75 to 1.05 ms and a ratio of 0.85 to 1.20; measured 0.52 to 0.55 ms and 0.66 to 0.69.
- **Batch 32 on Qwen3-8B: missed.** Predicted a lower ratio than at batch 1; it is higher (0.93 to 0.95).
- **Gemma 4's slowest batch exceeds the BF16 head: held,** and so does every batch at 32.
- Controls: held.

Limits: one run on each of two A100 variants (Qwen3-8B on PCIe, Gemma 4 on SXM4). Head alone. Gumbel noise supplied. Hoeffding bound, no shift, no Sarvam-30B. The baseline applies the soft-cap and adds the noise, like the L4 baseline here; E1's stock head was the cuBLAS product with argmax (0.82 and 0.89 ms).

# The unified kernel: the byte runs' certificate in the fast kernels (17 Nov 2026)

`fused.py` wins at batch 1 on the A100 but uses the Hoeffding bound with no shift and has no K2 correction, so it does not cover Sarvam-30B and loses on Gemma 4 from batch 8. `fused2.py` puts the certificate of the byte tables into the same kernels:
- **First pass:** the sub-Gaussian bound, copies of the shifted head `W_t` with the exact add-back, and K2's correction on Sarvam-30B.
- **INT8 stage:** T1's dithered INT8 copy with the same bound, in place of the worst-case INT8 rows.
- **Select and exact kernels, tensor-core product and its slack:** unchanged from `fused.py`.

Protocol `unified_protocol.json` (sha256 `fb6ed5fa…`, logged before `fused2.py` ran on any GPU). Code `cert_head2.py`, `fused2.py`, `fused2_modal.py`, `test_cert_head2.py`; results `results/fused2_<model>_A100-80GB.json`, table `logs/fused2_summary_a100.txt`. Same states, batches, repeats and baseline as the A100 run above; shifts 0.5 (Qwen3-8B, Sarvam-30B) and 1.5 (Gemma 4).

Checks, all pass:
- **Copies:** `test_cert_head2.py` on the CPU: the stored copies equal `cert_core`'s and `mix_lib`'s quantizers bit for bit, and the kernel's formula in plain PyTorch gives the byte runs' intervals.
- **Synthetic head** with a shift and a direction: every token equals the reference, eager and graph-captured.
- **Intervals** on 64 real states per model and rule: no reference logit outside the first-pass interval or, for the listed rows, the INT8-stage interval. The kernel's first-pass logit equals the plain-PyTorch formula to within 0.01% of the bound. Its bound is 1.01 to 1.07 times that formula's on Qwen3-8B and Gemma 4 and up to 1.47 times on Sarvam-30B, because the tensor-core slack weighs more once K2 has shrunk the statistical term.
- **Tokens:** no certified token differs from the FP32 reference on any timed batch or repeat.

A100-80GB, graph-captured, milliseconds:

| Model | Rule | Batch | BF16 head, median | Unified: median, p99, largest | Ratio, median | Batches slower than the BF16 median |
|---|---|---|---|---|---|---|
| Qwen3-8B | greedy | 1 | 0.74 | 0.53, 0.54, 0.54 | 0.72 | 0 of 512 |
| Qwen3-8B | greedy | 8 | 0.77 | 0.59, 0.60, 0.60 | 0.77 | 0 of 64 |
| Qwen3-8B | greedy | 32 | 0.87 | 0.83, 0.84, 0.84 | 0.95 | 0 of 16 |
| Gemma 4 E4B | greedy | 1 | 0.82 | 0.56, 0.57, 0.57 | 0.68 | 0 of 512 |
| Gemma 4 E4B | greedy | 8 | 0.87 | 0.63, 0.65, 0.67 | 0.72 | 0 of 64 |
| Gemma 4 E4B | greedy | 32 | 1.09 | 0.97, 1.04, 1.14 | 0.88 | 1 of 16 |
| Sarvam-30B | greedy | 1 | 1.24 | 0.82, 0.83, 0.99 | 0.66 | 0 of 512 |
| Sarvam-30B | greedy | 8 | 1.34 | 0.90, 1.08, 2.02 | 0.67 | 1 of 64 |
| Sarvam-30B | greedy | 32 | 1.42 | 1.43, 2.06, 5.71 | 1.01 | 8 of 16 |

Gumbel T = 1 is within 0.02 of greedy's ratio in every row (`logs/fused2_summary_a100.txt`), with the noise supplied.

What it says:
1. **With the paper's certificate, the head is faster than the stock head on an A100 at batch 1 and 8 on all three models:** 0.66 to 0.77 of its time, and no batch-1 state is slower. Sarvam-30B takes 0.82 ms against 1.24.
2. **Gemma 4 no longer loses.** Its p99 shortlist is 376 rows, where `fused.py` had 26,209, and its ratio is 0.72 at batch 8 (was 1.09) and 0.88 at batch 32 (was 3.10).
3. **At batch 32 the gain is small or gone:** 0.95 on Qwen3-8B, 0.88 on Gemma 4, 1.01 on Sarvam-30B. On Sarvam-30B one state with a 17,083-row shortlist takes its batch to 5.7 ms.
4. **The tails are short at batch 1 and 8** except on Sarvam-30B, whose largest batch-8 time is 2.0 ms.
5. **Where the time goes** (eager, a device sync between kernels, so each part carries launch overhead): at batch 1 on Qwen3-8B the per-state preparation takes 0.26 ms, the first pass 0.50, select 0.05, the INT8 stage 0.08, the exact stage 0.13. At batch 32 only the first pass grows, to 0.74 ms. It is the part to work on for larger batches.

Against the predictions: the checks pass; Gemma 4's shortlist is below 1,000 rows (376) and its ratios are within the predicted limits; batch 1 is 0.68 to 0.73 on Qwen3-8B and Gemma 4 (predicted 0.66 to 0.82) and 0.66 on Sarvam-30B (predicted 0.55 to 0.80). The breakdown prediction missed narrowly: the first pass is 57% of the eager step at batch 32 on Qwen3-8B, not above 60%.

Limits: head alone, one run per model, two A100 variants, Gumbel noise supplied. Sarvam-30B's direction is K2's, fitted on half of the prompts the timed states come from. M2's group stage is not in this kernel. Not yet run on an L4. E1's end-to-end finding is unchanged by this.

# Memory tiers: the higher-precision copies in host memory (17 Nov 2026)

Keeping every copy on the GPU takes 1.77 times the BF16 head's memory (INT4 + INT8 + BF16). With T1's dithered INT8 stage the BF16 rows are needed for 1 to 3 rows per token, so they could live in host memory. This run prices that.

Protocol `tier_protocol.json` (sha256 `cc33ea29…`, logged before any tiered code ran on a GPU). Code `tiered.py`, `tier_modal.py`, `tier_summarize.py`; results `results/tier_<model>_A100-80GB.json`, table `logs/tier_summary_a100.txt`. A100-80GB, 512 states per rule, greedy and Gumbel T = 1, wall-clock per step with the token on the host.

- **Layout B:** INT4 and INT8 copies on the GPU, the BF16 head on the host. GPU bytes: 0.77 of the BF16 head.
- **Layout A:** only the INT4 copy on the GPU; INT8 and BF16 on the host. GPU bytes: 0.26.

The GPU part is graph-captured and ends in one small tensor that is read back once. The host finishes the step on the CPU from its own copies; no weight row is copied to the GPU.

Checks, all pass: every token of both layouts equals the FP32 reference, and the rows fetched equal the rows the all-GPU kernel reads, on every step.

Batch 1, greedy, milliseconds (median; ratio to the stock BF16 head in brackets):

| Model | Stock BF16 head | All copies on the GPU (1.77) | Layout B (0.77) | Layout A (0.26) | Layout A: p99, largest |
|---|---|---|---|---|---|
| Qwen3-8B | 0.75 | 0.54 (0.72) | 0.72 (0.97) | 0.87 (1.17) | 0.95, 1.21 |
| Gemma 4 E4B | 0.84 | 0.58 (0.69) | 0.81 (0.96) | 1.10 (1.31) | 1.75, 5.66 |
| Sarvam-30B | 1.24 | 0.83 (0.67) | 1.07 (0.86) | 1.23 (0.99) | 2.86, 41.9 |
| Qwen3-0.6B | 0.25 | 0.23 (0.92) | 0.42 (1.66) | 0.64 (2.53) | 0.83, 1.27 |

Rows fetched from the host per step at batch 1 (median, p99): BF16 1 and 2 to 5; INT8 in layout A 1 to 9 and 21 to 1,278.

What it says:
1. **With the BF16 head off the GPU, the step is about as fast as the stock head on the three large heads** (0.86 to 0.99 at batch 1). The GPU then holds 0.77 of the stock head's bytes, and the output is the reference token. The read-back and the host's finish cost 0.18 to 0.23 ms.
2. **With only the 4-bit copy on the GPU (0.26 of the bytes), the median step is 1.0 to 1.3 times the stock head, with a long tail.** A state with a 17,083-row shortlist took 42 ms on Sarvam-30B, because the host does the INT8 stage for those rows.
3. **On the small head both layouts are slower than the stock head** (1.7 and 2.5 times). The fetch costs a fixed 0.2 to 0.4 ms, and the whole stock head takes 0.25 ms there. Against E2's 4.5 ms decode step that is 4% to 9% more time per token.
4. **At batch 8 both layouts lose badly** (1.2 to 3.4 times for B, 3 to 12 times for A): the host finishes the states one after another.

So the footprint claim that holds is narrow: at batch 1, on an A100, the BF16 copy can leave the GPU at no cost in head time for heads of 1.2 GB and more. Taking the INT8 copy off as well costs time and has a bad tail. Neither is shown for batches.

Against the predictions: tokens exact, held. Layout B's extra time (0.10 to 0.30 ms predicted, 0.18 to 0.23 measured) and its ratios, held. Layout A's extra time held on Qwen3-8B and Sarvam-30B and was just over on Gemma 4 (0.52 ms against a predicted 0.50); its tail above the stock head on Sarvam-30B and Gemma 4, held. BF16 rows at most 4 at the p99: missed on Sarvam-30B (5).

How the implementation got here, since the protocol was hashed before any of it ran: the first version ran the kernels without graph capture and read back three times a step (1.2 ms on Qwen3-0.6B, where the all-GPU step takes 0.24); the second graph-captured the GPU part but copied the fetched rows to the GPU (0.84 ms); the one reported finishes on the host (0.42 ms). All three returned the reference token on every smoke state. The smoke logs of the first two are kept (`logs/tier_smoke_first_version.txt`, `logs/tier_smoke_second_version.txt`).

Limits: one GPU type, one run, batch 1 and 8 only, a host CPU shared with the Python driver. The GPU copies were not actually freed during timing; the byte figures are computed from the tensors each layout needs. The host part is plain PyTorch on the CPU and is not optimised. Top-50 is not covered by these kernels, and its INT8 shortlists are far too long for layout A.

# The runs that were waiting for an L4, on an A10G (17 Nov 2026)

No L4 could be started on Modal in four attempts over three days, so the waiting runs went to an A10G, the available GPU with slower memory than the A100. Protocol `a10g_protocol.json` (sha256 `a1caf90f…`, logged before any of these harnesses ran on an A10G): the tail, unified and tier protocols unchanged, with the GPU changed. Results `results/fused_tail_*_A10G.json`, `results/fused2_*_A10G.json`, `results/tier_*_A10G.json`; tables `logs/fused_tail_summary_a10g.txt`, `logs/fused2_summary_a10g.txt`, `logs/tier_summary_a10g.txt`.

Every check of the three protocols passes: no certified token differs from the FP32 reference on any timed batch or repeat, no reference logit lies outside an interval, and row counts are stable across repeats.

## The unified kernel (`fused2.py`), greedy, milliseconds

| Model | Batch | BF16 head, median | Unified: median, p99, largest | Ratio, median | Batches slower than the BF16 median |
|---|---|---|---|---|---|
| Qwen3-8B | 1 | 2.62 | 0.97, 0.99, 0.99 | 0.37 | 0 of 512 |
| Qwen3-8B | 8 | 2.65 | 1.05, 1.16, 1.16 | 0.40 | 0 of 64 |
| Qwen3-8B | 32 | 2.95 | 1.49, 1.51, 1.51 | 0.50 | 0 of 16 |
| Gemma 4 E4B | 1 | 2.83 | 1.00, 1.01, 1.03 | 0.35 | 0 of 512 |
| Gemma 4 E4B | 8 | 2.99 | 1.12, 1.15, 1.28 | 0.37 | 0 of 64 |
| Gemma 4 E4B | 32 | 3.74 | 1.67, 1.75, 1.80 | 0.45 | 0 of 16 |
| Sarvam-30B | 1 | 4.51 | 1.54, 1.56, 1.75 | 0.34 | 0 of 512 |
| Sarvam-30B | 8 | 4.60 | 1.67, 1.82, 2.54 | 0.36 | 0 of 64 |
| Sarvam-30B | 32 | 5.11 | 2.15, 2.62, 5.34 | 0.42 | 1 of 16 |

Gumbel T = 1 is within 0.02 of greedy's ratio in every row, with the noise supplied.

- **On the A10G the certified head takes about a third of the stock head's time at batch 1 and 8 on all three models, and under half at batch 32.** Of the 3,552 timed batches over both rules, one is slower than the stock head (Sarvam-30B, greedy, batch 32: the batch holding a 17,083-row shortlist).
- **The tails are short:** the p99 is within 11% of the median at batch 1 and 8.
- **The same kernel on the A100 gave 0.66 to 0.77 at batch 1 and 8 and about even at batch 32.** The stock head's time follows memory speed (2.6 ms here, 0.74 ms there on Qwen3-8B), so the slower the memory, the more the certified head saves.

## Tail timings of the first kernels (`fused.py`)

Batch 1: 0.36 of the stock head on Qwen3-8B (p99 0.36) and 0.35 on Gemma 4 (p99 0.42); no state is slower than the stock head. Qwen3-8B wins at every batch size (0.39 at 8, 0.51 at 32). Gemma 4 is 0.50 at batch 8 and loses on 9 of 16 batches at 32 (median 1.08), from the long shortlists of the Hoeffding bound that `fused2.py` does not have. This answers the first review correction of this README for the A10G: the batch-1 claim holds in the tail as well as at the median.

## Memory tiers (`tiered.py`), batch 1, greedy

Milliseconds at the median, with the ratio to the stock head in brackets.

| Model | Stock BF16 head | All copies on the GPU (1.77) | Layout B: BF16 on the host (0.77) | Layout A: BF16 and INT8 on the host (0.26) | Layout A: p99, largest |
|---|---|---|---|---|---|
| Qwen3-8B | 2.63 | 0.98 (0.37) | 1.36 (0.52) | 1.33 (0.50) | 1.44, 1.7 |
| Gemma 4 E4B | 2.84 | 1.01 (0.36) | 1.21 (0.43) | 1.56 (0.55) | 2.09, 3.1 |
| Sarvam-30B | 4.50 | 1.62 (0.36) | 2.11 (0.47) | 2.04 (0.45) | 3.38, 42.2 |
| Qwen3-0.6B | 0.71 | 0.30 (0.42) | 0.46 (0.65) | 0.62 (0.87) | 0.72, 1.2 |

- **On this GPU the head can hold a quarter of the stock head's bytes and take half its time.** With only the 4-bit copy on the GPU the median step is 0.45 to 0.55 of the stock head on the three large heads. On the A100 that layout was 1.0 to 1.3 times slower.
- **Both layouts beat the stock head on the small model too** (0.65 and 0.87).
- **The tail of layout A remains:** Sarvam-30B's p99 is 0.75 of the stock head, and the state with a 17,083-row shortlist took 42 ms.
- **At batch 8,** layout B is 0.61 to 0.75 on the large heads and 1.2 on the small one; layout A is slower than the stock head everywhere (1.1 to 3.6 times).

Against the predictions: all checks pass. Unified at batch 1, predicted 0.35 to 0.50: held on Qwen3-8B and Gemma 4, and Sarvam-30B is just below it (0.34). Unified at batch 32, predicted 0.50 to 0.85 on Qwen3-8B and Gemma 4: both are at or below the lower end (0.50 and 0.45). Tail run: held. Layout B, predicted 0.40 to 0.65: held (0.43 to 0.52). Layout A, predicted 0.50 to 0.90 with a long tail: held on Qwen3-8B and Gemma 4, Sarvam-30B a little better (0.45). Qwen3-0.6B layout B, predicted 0.65 to 1.0: held (0.65).

Limits: one run per configuration; head alone; Gumbel noise supplied; the host part of the tiers is unoptimised PyTorch on a shared CPU. Sarvam-30B's direction is K2's, fitted on half of the prompts the timed states come from. Still no L4.
