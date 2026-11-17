# Paper outline (one page, updated 17 Nov 2026)

Working title: **Read less, sample exactly: a certified low-bit output layer**

## 1. The problem

The output layer of a language model is read in full for every token: one BF16 row per vocabulary entry. A 4-bit copy is a quarter of the bytes, but used alone it emits a different token from the reference on 2% to 9% of greedy steps. We want the 4-bit read and the reference token, on every step.

## 2. The method

**Base certificate.** Read a dithered 4-bit copy of every row. The dither makes each row's error random with a known spread, so each logit gets an interval that holds with probability 1 − δ (δ = 1e-15 per row). Rows the sampling rule cannot yet decide are re-read from a dithered INT8 copy with the same kind of bound, then in BF16. The token equals the reference token (BF16 values, FP32 accumulation). Floor: 0.258 of the BF16 head's bytes. With the dithered INT8 stage the BF16 head is almost never read: 2 to 48 rows at the p99 on Qwen3-8B, Gemma 4 and Sarvam-30B, for every rule that passes.

**Part 1: remove a few state directions exactly.** Where the intervals are too wide, most of the state's energy lies along a few fixed directions. Store one number per row and direction, the row's error along it, and subtract that part of the error exactly. Only the rest of the state is bounded. Cost: 4 bytes per row and direction. No added probability. One direction on Sarvam-30B (the last block writes it, the final norm's gain doubles it); four on Sarvam-1, SmolLM3-3B and Phi-4-mini.

**Part 2: read one heavy coordinate group only where the rule asks.** Where one group of 128 coordinates carries much of the remaining bound, read that group in INT8 for the rows the rule still needs (about 200 bytes a row) before the full INT8 row. Free where Part 1 is at the floor; lowers top-50. Keeping the group in INT8 for every row was tested and rejected.

A mean-row shift of the head before quantizing (from prior work) is tuned per model.

## 3. Byte tables

Fraction of the BF16 head read, greedy (median, p99, mean).

| Model | Untuned | With the method |
|---|---|---|
| Qwen3-8B | 0.258, 0.258, 0.258 | same |
| Gemma 4 E4B | 0.258, 0.258, 0.258 | same |
| Qwen3-0.6B | 0.258, 0.259, 0.258 | same |
| Sarvam-30B | 0.258, 0.512, 0.279 | 0.258, 0.259, 0.258 |
| Sarvam-1 | 0.284, 0.759, 0.386 | 0.262, 0.268, 0.262 |
| Phi-4-mini | 0.258, 0.385, 0.275 | 0.261, 0.283, 0.262 |
| SmolLM3-3B | 0.409, 1.086, 0.482 | 0.264, 0.342, 0.270 |

Other rules: Gumbel sampling passes its gate on all seven. Min-p passes on five (not Sarvam-1, Phi-4-mini). Top-50 passes on Qwen3-8B, Gemma 4, Sarvam-30B and Qwen3-0.6B, at a p99 of 0.27 to 0.34.

Evidence: no token mismatch and no bound violation on any state, in over 700,000 state evaluations. The Sarvam-30B result holds on a second prompt set with nothing refitted (greedy p99 0.260 against 0.52 uncorrected) and across four dither draws.

## 4. Timing (head alone, greedy, exact tokens)

| GPU | Batch | Certified / stock BF16 head |
|---|---|---|
| L4 | 1 | 0.35 to 0.41 (first kernels; medians) |
| A10G | 1 and 8 | 0.34 to 0.40 on Qwen3-8B, Gemma 4, Sarvam-30B |
| A10G | 32 | 0.42 to 0.50 |
| A100-80GB | 1 and 8 | 0.66 to 0.77 on Qwen3-8B, Gemma 4, Sarvam-30B |
| A100-80GB | 32 | 0.88 to 1.01 |

End to end: in an eager Hugging Face loop the head is 1% to 4% of a step on 8B-and-larger models, so the step is unchanged. On Qwen3-0.6B with a static cache on an A10G the head is 12% of a step. With the certified head inside the loop the step is a measured 6% faster in each of two runs (5.18 and 5.19 ms against 5.50), every token equal to the reference. On an A100 that model shows no gain.

Memory: all copies on the GPU take 1.77 times the stock head's bytes. On an A10G at batch 1, with only the 4-bit copy on the GPU (0.26 of the bytes) and the rest in host memory, the step takes 0.45 to 0.55 of the stock head on the three large heads, with a long tail on Sarvam-30B. On an A100: With the BF16 copy in host memory (GPU: 0.77) the batch-1 step on an A100 is as fast as the stock head on the three large heads (0.86 to 0.99). With only the 4-bit copy on the GPU (0.26) the median is 1.0 to 1.3 times the stock head, with a long tail. Not shown for batches.

## 5. Stated limits

- **Top-p** is exact but cheap on Qwen3-8B only; on Gemma 4 and Sarvam-30B it reads the whole vocabulary in INT8 on over 1% of states.
- **Top-50** does not pass on Sarvam-1, SmolLM3-3B or Phi-4-mini.
- The guarantee is probabilistic (δ per row and per dithered read: up to 3δ), and it certifies the FP32-accumulated reference, which the stock BF16 head itself misses on 0.2% to 2.5% of steps through ties.
- Shift, directions and groups are chosen per model on half of one prompt set. Only Sarvam-30B has been checked on new prompts.
- Head timings are with Gumbel noise supplied. The one measured end-to-end gain is on one small model, at batch 1, on one mid-range GPU.

## Open before writing

1. End to end beyond one small model at batch 1: a larger share of the step, batches, and an L4.
2. The unified kernel on an L4, and tail timings of the first kernels there.
3. Batch 32: the first pass's per-state work.
4. What in Sarvam-30B's last block writes the shared vector.
