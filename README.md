# Read less, sample exactly: certified sampling from a low-bit LM head

Research code for certifying next-token decisions from a 4-bit copy of a language model's output head. A first pass reads the 4-bit head, bounds every logit's error, and re-reads in INT8 or BF16 only the rows that could still change the decision (greedy, Gumbel-max, top-k or min-p). Tokens match a fixed FP32-accumulated BF16 reference.

Models: Qwen3-8B (151,936 × 4,096 head), Gemma 4 E4B (262,144 × 2,560, tied, soft-cap 30), Sarvam-30B (262,144 × 4,096).

## Contents

- [`exp01_repro/`](exp01_repro/): an independent reproduction of Vizuara's Exp01 (how wide is a 4-bit certificate), then three ablations and first GPU timings. All but the shared-error probe were pre-registered: the rule file was hashed before the run (`exp01_repro/logs/a3_decision_rule.log`, `decision_rule.log`).
  - **A3**, mean-row shift before quantizing (after Kadav et al., arXiv:2609.31291)
  - a shared-error ("gap-to-leader") probe
  - **B1**, a sub-Gaussian bound for the dithered head
  - **L4 timing**, first kernels for the certified greedy step on an NVIDIA L4
  - **F1**, how often the same low-bit copies pick a different token when used without a certificate
  - **K1**, an exploratory probe of why Sarvam-30B's certificate is wide
  - **K2**, cancelling exactly the quantization error along the state's dominant directions
  - **K3**, a survey of fourteen models: Sarvam-30B is the only one whose states are dominated by one direction
  - **E1**, the head's share of a whole decode step on an A100-80GB

  Results and caveats are in [`exp01_repro/README.md`](exp01_repro/README.md).
- [`kernel/`](kernel/): a second, independently built set of L4 kernels for the certified step (15 Nov 2026): a PyTorch baseline, a tensor-core first pass, and a fused, graph-captured step with per-state row lists for greedy and Gumbel sampling. See [`kernel/README.md`](kernel/README.md), including how it relates to the graph-captured step in `exp01_repro`.
- [`notebooks/`](notebooks/): the vocabulary-mean-row statistics of the three heads.

## Status (14 Nov 2026)

With a dithered INT4 g128 head, the sub-Gaussian bound, an INT8 stage and the A3 shift:
- **Qwen3-8B** passes the gates for all five rules.
- **Gemma 4** passes for all rules except top-50.
- **Sarvam-30B** passes greedy, both Gumbel temperatures and min-p once the error along the state's dominant direction is cancelled exactly (K2, one stored number per row, with the direction fitted without the state's language): greedy bytes median, p99, mean 0.258, 0.260, 0.258, the same floor as the other two. Top-50 improves (mean 0.420 to 0.277) but does not pass. Without that correction it sits on the p99-bytes gate (< 0.5 of the BF16 head) for greedy and Gumbel: 0.512 (greedy), 0.514 (T = 0.6), 0.496 (T = 1). Only T = 1 passes, by 0.004, with one dither draw and no interval on the p99. Top-50 and min-p 0.05 do not pass (p99 0.694 and 0.561).

These are byte counts, and the guarantee is high-probability (δ = 1e-15 per row). Timings of the greedy step on an NVIDIA L4 are in [`exp01_repro/README.md`](exp01_repro/README.md). The 4-bit first pass alone takes 0.39 of the cuBLAS BF16 head's time. The graph-captured step takes 0.40 to 0.41 at the batch-1 median on all three heads (p99 0.42, and 0.68 on Sarvam) and 0.41 to 0.67 at batch 16, with exact tokens on every step. With the K2 correction Sarvam's step is 0.40 (p99 0.41) at batch 1 and 0.41 at batch 16, so all three heads sit at about 0.40 to 0.41 at both batch sizes.

That is an L4 result and does not carry over to a faster GPU or to a whole decode step (E1). On an A100-80GB the certified head takes 1.0 to 1.3 times as long as the stock BF16 head, because the first-pass kernel is bound by its own arithmetic, not by memory. In a Hugging Face greedy loop the head is 1% to 4% of a step, so the end-to-end step time is unchanged (1.000 to 1.004 of stock). The byte counts and the exactness are unaffected; a time saving in a memory-bound decode loop is not shown either way. These times are for the head alone, not end-to-end decoding.

Without the certificate, the same 4-bit first pass emits a different token from the reference on 2.0% (Qwen3-8B), 2.7% (Gemma 4) and 8.8% (Sarvam-30B) of greedy steps, and about four prompts in five depart from the reference within 128 tokens. On Qwen3-8B and Gemma 4 most of these are near-ties (median 0.3 logits); on Sarvam-30B nearly half are more than 1 logit off. The stock BF16 head itself differs from the reference on 0.41%, 0.23% and 2.54% of greedy steps, always through ties in its BF16 logits: the certificate reproduces the FP32-accumulated reference, not the stock BF16 run.

Sarvam-30B's tail is not a loose bound: its bound is no wider than Gemma 4's, but its whole vocabulary sits close under the leader (the median token is 1.3 interval-widths below the top logit, against 2.1 and 2.9), so a large part of it stays within reach. A shared offset supplies 77% of its top logit. That offset is in the state: 74% of the states' second moment lies along one direction, which widens every interval without separating any two tokens. Centring the head rows cannot remove it; cancelling its error exactly (K2) does, and takes Sarvam's greedy p99 from 0.512 to 0.260. Across fourteen models (K3) Sarvam-30B is the only one whose states are dominated by one direction, so in this set the correction is specific to it. The same survey found the uncorrected certificate at the 0.258 floor on seven of thirteen models and well off it on four (Phi-4-mini, Sarvam-30B, Sarvam-1, SmolLM3-3B).

## Not included

- Prompts and model outputs: the Indian-language prompts come from Flores-IN, cleared for evaluation but not redistribution.
- Per-state result arrays: about 170 MB here and more on Modal, regenerable with the scripts.
- Third-party papers and the private project brief and literature review.

`exp01_repro/sarvam_modal.py` uses the Sarvam lockfile from [VizuaraAI/certified-sampling](https://github.com/VizuaraAI/certified-sampling) (MIT), expected at `../vizuara_certified_sampling/` next to `exp01_repro/`.
