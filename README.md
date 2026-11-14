# Read less, sample exactly: certified sampling from a low-bit LM head

Research code for certifying next-token decisions from a 4-bit copy of a language model's output head. A first pass reads the 4-bit head, bounds every logit's error, and re-reads in INT8 or BF16 only the rows that could still change the decision (greedy, Gumbel-max, top-k or min-p). Tokens match a fixed FP32-accumulated BF16 reference.

Models: Qwen3-8B (151,936 × 4,096 head), Gemma 4 E4B (262,144 × 2,560, tied, soft-cap 30), Sarvam-30B (262,144 × 4,096).

## Contents

- [`exp01_repro/`](exp01_repro/): an independent reproduction of Vizuara's Exp01 (how wide is a 4-bit certificate), then three ablations. Each was pre-registered: its rule file was hashed before the run (`exp01_repro/logs/a3_decision_rule.log`, `decision_rule.log`).
  - **A3**, mean-row shift before quantizing (after Kadav et al., arXiv:2609.31291)
  - a shared-error ("gap-to-leader") probe
  - **B1**, a sub-Gaussian bound for the dithered head

  Results and caveats are in [`exp01_repro/README.md`](exp01_repro/README.md).
- [`notebooks/`](notebooks/): the vocabulary-mean-row statistics of the three heads.

## Status (14 Nov 2026)

With a dithered INT4 g128 head, the sub-Gaussian bound, an INT8 stage and the A3 shift:
- **Qwen3-8B** passes the gates for all five rules.
- **Gemma 4** passes for all rules except top-50.
- **Sarvam-30B** passes for Gumbel T = 1 and misses greedy on p99 bytes (0.512 of the BF16 head, gate < 0.5).

All of these are byte counts, not timings, and the guarantee is high-probability (δ = 1e-15 per row).

## Not included

- Prompts and model outputs: the Indian-language prompts come from Flores-IN, cleared for evaluation but not redistribution.
- Per-state result arrays: about 170 MB here and more on Modal, regenerable with the scripts.
- Third-party papers and the private project brief and literature review.

`exp01_repro/sarvam_modal.py` uses the Sarvam lockfile from [VizuaraAI/certified-sampling](https://github.com/VizuaraAI/certified-sampling) (MIT), expected at `../vizuara_certified_sampling/` next to `exp01_repro/`.
