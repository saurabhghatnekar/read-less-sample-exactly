# Exp01, independent reproduction

Rebuilt from the method in Vizuara's lab note (5 Nov 2026), without their code. Started 9 Nov 2026.

| File | What |
|---|---|
| `decision_rule.json` | What counts as "reproduced", written and hashed before any run (`logs/decision_rule.log`, sha256 `658306ba…`) |
| `cert_core.py` | Head variants (INT8 row, INT4 g128, dithered INT4, rotated INT4, NVFP4), worst-case / residual / Hoeffding bounds with FP32 slack, rule recompute sets, counter-based Gumbel noise, byte accounting |
| `test_local.py` | CPU self-tests: soundness, exact tokens for every rule and certificate, dither variance, top-k membership vs plain Python |
| `exp01_modal.py` | Modal: `--step prompts`, `--step dump`, `--step analyze` (A100-80GB) |
| `summarize.py` | Tables and decision-rule verdict from `results/<model>` |
| `controls_modal.py`, `controls_lib.py`, `c4_oracle_order.py`, `lab_status.py` | Controls C4, C6, C7, C8 and statuses under the lab's rule (added 10 Nov) |
| `logs/summary_qwen3_8b.txt` | Full Qwen3-8B result |

Prompts include Flores-IN text, cleared for evaluation with attribution but not for sharing. Keep `prompts.jsonl` and model outputs private.

## Qwen3-8B (b968826d), 57,811 states: REPRODUCED

Every pre-registered check passed. Bounds held on every row, every certified token matched the FP32 reference, and the dither's normalised squared error was 1.0007 over 8.8 billion rows.

Main table in the lab note's format (`python3 -I table.py results/qwen3_8b`). Each row is the dithered INT4 head with the INT8 stage. S is rows read in INT8 and R rows read in BF16, each as median, p99. Bytes are fractions of the BF16 head. The INT8 column is the INT8 certificate alone, with whichever bound (worst-case or residual) has the lower median.

| Model | Rule | Status | S | R | Bytes median, p99, mean | INT8 bytes median, p99 |
|---|---|---|---|---|---|---|
| Qwen3-8B | greedy | GATES_PASS | 3, 157 | 1, 9 | 0.258, 0.258, 0.258 | 0.500, 0.501 |
| Qwen3-8B | Gumbel T = 0.6 | GATES_PASS | 3, 170 | 1, 9 | 0.258, 0.258, 0.258 | 0.500, 0.501 |
| Qwen3-8B | Gumbel T = 1 | GATES_PASS | 3, 179 | 1, 9 | 0.258, 0.258, 0.258 | 0.500, 0.501 |
| Qwen3-8B | top-50 | UNDER_HALF | 6, 82,589 | 1, 10 | 0.258, 0.530, 0.315 | 0.500, 0.501 |
| Qwen3-8B | min-p 0.05 | GATES_PASS | 6, 629 | 1, 20 | 0.258, 0.260, 0.258 | 0.500, 0.501 |

Lab note, same rows: S 3,206 / 3,208 / 3,213 / 8,85,489 / 6,752; R 1,10 / 1,10 / 1,10 / 1,16 / 1,20; bytes 0.258,0.259,0.258 (×3), 0.258,0.540,0.320, 0.258,0.261,0.258; INT8 0.501, 0.501. Statuses match.

Status in these two tables is our own reading of the lab's labels, written before their rule file was public: GATES_PASS: median R ≤ 0.1% of the vocabulary, p99 R ≤ 1%, and p99 bytes under 0.5, over all states. DEAD: median bytes ≥ 0.5. UNDER_HALF: neither. The lab's rule is different (gates in every language group, median bytes ≤ 0.35, no p99-bytes gate); see "Review, 10 Nov 2026" for statuses under their rule.

| Readout (greedy medians) | ours | lab |
|---|---|---|
| mean worst-case / mean dithered bound | 5.14 | 4.87 |
| worst-case bound / true max error | 26.0 | 26.2 |
| dithered bound / true max error | 5.54 | 5.5 |
| true largest INT4 error (logits) | 2.20 | 2.19 |
| top-two margin | 8.53 | 8.16 |

Differences:
- Best single-stage deterministic 4-bit certificate at T=1: ours 0.652 (NVFP4, residual), lab 0.866. The cross-check below traces this to the NVFP4 quantiser, not the bound. Either way the conclusion is unchanged: no deterministic 4-bit bound beats INT8 (0.500).
- Prompts: Flores-IN instead of newspaper OCR. MT-Bench and HumanEval subsets chosen by our own rule (5 per category, first 40 ids).
- Software: torch 2.15.0, transformers 5.22.2. The lab used torch 2.13.0 and transformers 5.19.0.

## Gemma 4 E4B (4ecc2ba6), 59,215 states: PARTIAL (top-50 only)

Targets pre-registered in `decision_rule_gemma4.json` (sha256 `a0c6ed02…`) before any Gemma dump. Controls are clean: no bound violations, no token mismatches outside the dither certificate, dither normalised squared error 0.9992 over 15.5 billion rows, and C1 alignment 99.8%. Greedy, both Gumbel temperatures and min-p reproduce. Top-50 misses on two pre-registered checks: its median is 0.486 against 0.503, so its status is UNDER_HALF instead of DEAD.

| Model | Rule | Status | S | R | Bytes median, p99, mean | INT8 bytes median, p99 |
|---|---|---|---|---|---|---|
| Gemma 4 E4B | greedy | GATES_PASS | 24, 21,525 | 1, 10 | 0.258, 0.299, 0.261 | 0.501, 0.501 |
| Gemma 4 E4B | Gumbel T = 0.6 | GATES_PASS | 27, 22,807 | 1, 10 | 0.258, 0.301, 0.261 | 0.501, 0.501 |
| Gemma 4 E4B | Gumbel T = 1 | GATES_PASS | 31, 22,887 | 1, 12 | 0.258, 0.302, 0.261 | 0.501, 0.501 |
| Gemma 4 E4B | top-50 | UNDER_HALF | 118,718, 199,601 | 1, 7,831 | 0.486, 0.640, 0.435 | 0.501, 0.531 |
| Gemma 4 E4B | min-p 0.05 | GATES_PASS | 103, 37,525 | 1, 37 | 0.258, 0.330, 0.265 | 0.501, 0.501 |

Lab note: S 37,23,089 / 43,25,826 / 49,25,654 / 126,659,201,225 / 165,42,447; bytes p99 0.302 / 0.308 / 0.307 / 0.644 / 0.340; top-50 bytes 0.503, 0.644, 0.450 (DEAD).

| Readout (greedy medians) | ours | lab |
|---|---|---|
| mean worst-case / mean dithered bound | 3.65 | 3.35 |
| worst-case bound / true max error | 17.6 | 17.3 |
| dithered bound / true max error | 6.23 | 6.3 |
| true largest INT4 error (logits) | 4.91 | 4.94 |
| largest dithered bound (logits) | 34.1 | 34.1 |
| top-two margin (capped) | 7.14 | 5.60 |

| Median S on greedy, by group | English | code | Hindi | Bengali | Tamil | Telugu |
|---|---|---|---|---|---|---|
| ours | 2 | 1 | 14 | 292 | 458 | 290 |
| lab | 2 | 1 | 18 | 490 | 576 | 326 |

Top-50: ours and the lab's both put the dithered certificate at INT8 parity, not below it: a median of 0.49 against 0.50 for INT8, and a p99 of 0.64 against 0.53. Our median sits just under the 0.5 line that defines DEAD, and theirs just over. Their S median is also higher (126,659 against our 118,718). The cross-check below shows the read plans give identical sets, so the plan is not the cause. The conclusion is the same: top-k isn't claimable under this plan on a 262K head.

The language finding reproduces: Indian-language states need 150–300× more INT8 rows than English, but stay under every gate. The top-two margin is wider on our Flores prompts than on their newspaper OCR, which is consistent with cleaner source text.

## Against Vizuara's code (github.com/VizuaraAI/certified-sampling at 30bf9a6b, MIT)

Read on 9 Nov 2026 after both runs above were finished, so neither run used it. Cloned to `../vizuara_certified_sampling/`.

### Cross-check: their cert_core.py on our states (`crosscheck_modal.py`, `crosscheck_lib.py`)

Their `cert_core.py` is used unmodified on our saved T = 1 states (Gemma 4: 29,574; Qwen3-8B: 28,871), with our reference and our Gumbel noise. Each comparison swaps one component and holds the rest fixed. Bytes use their byte model throughout. Results: `results/crosscheck_<model>.json`, logs: `logs/crosscheck_<model>.log`.

Controls: the two codes' reference tokens agree on every state, and no cell has a single token mismatch.

| Dithered INT4 + INT8, top-50, bytes median / p99 | our plan | their plan |
|---|---|---|
| Gemma 4, our intervals | 0.486 / 0.641 | 0.486 / 0.641 |
| Gemma 4, their intervals | 0.491 / 0.643 | 0.491 / 0.643 |
| Qwen3-8B, our intervals | 0.258 / 0.530 | 0.258 / 0.530 |
| Qwen3-8B, their intervals | 0.258 / 0.538 | 0.258 / 0.538 |

| NVFP4 + residual bound, Gumbel T = 1, bytes median | our bound and plan | their bound and plan |
|---|---|---|
| Qwen3-8B, our quantiser | 0.652 | 0.665 |
| Qwen3-8B, their quantiser | 0.828 | 0.841 |
| Gemma 4, our quantiser | 1.093 | 1.100 |
| Gemma 4, their quantiser | 1.189 | 1.194 |

What this settles:
- **Top-50 on Gemma 4.** The read plan and the membership test make no difference: the sets are identical. Their slightly wider intervals (larger FP32 slack, 1e-4 padding, FP16 values rounded up) add 0.005, about a third of the 0.017 gap. The other 0.012 comes from the states themselves, most likely our Flores-IN prompts against their newspaper text; this is untested. Their full code on our states still gives 0.491, which is UNDER_HALF. The DEAD/UNDER_HALF difference is a median sitting on the 0.5 line, not a bug on either side. Gumbel T = 1 is unchanged (0.258) under every swap.
- **Best deterministic certificate on Qwen3-8B.** The NVFP4 quantiser explains it: their round-up block scales give 0.828 to 0.841 on our states, against 0.652 to 0.665 with ours. That matches their 0.866 to within the difference in states. Their bound and plan add only about 0.013.

What their code shows about the two differences above (the hypotheses written before the cross-check; the results above replace them):

- **Gemma 4 top-50 (ours 0.486 UNDER_HALF, theirs 0.503 DEAD).** Both use the same pre-registered plan: if any race candidate has uncertain membership, read every uncertain row. The gap is in S (median 118,718 against 126,659), and 7,941 rows × 0.5 / 262,144 ≈ 0.015 of the head, almost all of the 0.017 gap. Candidate causes: (1) their intervals are slightly wider: FP32 slack 3·d·u·‖h‖‖w‖ against our ≈2·d·u, plus 1e-4 on every interval end, plus FP16 scales and bounds rounded up; (2) our rank-count membership test excludes the row itself, while theirs compares against the k-th lower and (k+1)-th upper ends; (3) different prompts (Flores-IN against newspaper OCR). Their own post-hoc pass (lab note, "Top-k (post hoc)") finds that a stepwise plan brings Gemma 4 top-50 to a 0.258 median, so the DEAD label belongs to the plan, not the bound.
- **Best deterministic 4-bit certificate on Qwen3-8B at T = 1 (ours 0.652, theirs 0.866, both NVFP4 with the residual bound).** Both group the residual bound over 128 weights. The difference is in the NVFP4 emulation: they round each E4M3 block scale up (`searchsorted`), so no value clips but every step is coarser, while we cast to E4M3 (round to nearest) and let the few values above 6 clip. Our residuals are smaller, so our bound is tighter.
- **Other differences.** Their rotated INT4 uses a random permutation and Hadamard blocks of 512 (ours: 128, no permutation). Their noise is SplitMix64 (ours: fmix32). Their cascade reads INT8 once with the residual-norm interval; ours repeats the rule until nothing new is needed.

Their Sarvam-30B result, which we did not run: MIXED for greedy and Gumbel, DEAD for top-50 and min-p (median about 0.46 to 0.48 bytes, p99 near 0.77). Their whole run cost about 5.78 USD. They did not use Flores-IN (licence rule at the time); we did, under the project supervisor's later clearance.

## Review, 10 Nov 2026

### Statuses under the lab's published rule (`lab_status.py`)

Their `per_rule_status` (decision_rule.json in their repo), applied to our states, dithered INT4 + INT8 only. It differs from the status column above for top-50 on both models.

| Rule | Qwen3-8B | Gemma 4 E4B |
|---|---|---|
| greedy, Gumbel T = 0.6, Gumbel T = 1, min-p 0.05 | GATES_PASS | GATES_PASS |
| top-50 | GATES_PASS (largest group median 0.342, Bengali) | MIXED (English and code 0.258; Hindi 0.463, Bengali 0.549, Tamil 0.528, Telugu 0.570) |

### Controls C7 and C8 (`controls_modal.py`, `results/controls_<model>.json`)

- **C7 language: pass.** The median prompt is 99% to 100% in the target script in every Indian-language group, both models, both modes. Single prompts miss: Gemma 4 has one Bengali prompt at 0% (greedy) and one Telugu prompt at 9%.
- **C8 sampling: pass over all states.** NLL over entropy at T = 1 is 1.019 (Qwen3-8B) and 0.988 (Gemma 4), inside 0.9 to 1.1. The code group alone is outside it: 1.166 on Qwen3-8B and 0.826 on Gemma 4. The lab's rule gates the overall ratio only.

### Controls C4 and C6

- **C4 oracle order: pass** (`c4_oracle_order.py`, from the saved results, `results/controls_c4.json`). On every greedy state and all 11 single-stage certificates, the set under the true error is never larger than the certificate's set. The same holds for the other four rules (readout).
- **C6 planted faults: pass** (`controls_lib.py`, `controls_modal.py --step c6`, `results/controls_c6_<model>.json`; CPU test `test_controls.py`). First 1,024 states of each mode, both models:
  - 2% of the residual bound is violated on every head.
  - A head with rows shifted by one violates the worst-case and residual bounds on every head.
  - Taking the winner out of the recompute set gives a wrong token on every state, for every rule.
  - An exact head with a zero bound leaves exactly one row on every state, for every rule.
  - An all-zero head leaves every row and shows no violations.
  - FP32 against FP64 on 64 states: the largest difference is 0.45% of the slack, including the rotated head.
- The first Qwen3-8B run reported 1,023 of 1,024 on the winner-removed check at T = 1. The cause was in the check, not the certificate: when the winner was the only row read, the emptied set fell back to token 0, which was the reference token on that state. An empty set now returns no token; the rerun gives 1,024 (`logs/controls_c6_qwen3_8b_first_run.json` keeps the first result).

### Known gaps in `cert_core.py`

- FP16 scales are rounded to nearest, and the bound statistics stay in float32 although the byte model charges FP16 for them. The lab rounds up.
- The FP32 slack covers the logit product, not the rounding in the bound itself. The lab adds 1e-4 to each interval end. C6 shows the product rounding, including `h @ R` for the rotated head, uses under 0.5% of the slack.
- Top-k: an exact value that ties with an unread row's lower end can stay uncertain. No such case occurred (zero mismatches).
- One dither draw and one sampling seed. Package versions are not pinned.

## Ablation A3: mean-row shift (run 14 Nov 2026)

Question: if the INT8, INT4 and dithered INT4 copies are built from `W_t = W − t·1µᵀ` (Kadav et al., arXiv:2609.31291), do the bounds and the bytes read shrink? The reference stays the original BF16 head. Approximate logits get the exact term `t·(h·µ)` added back before any rule, which also makes Gemma 4's soft-cap path exact.

| File | What |
|---|---|
| `a3_decision_rule.json` | Grid, prompt split (even ids validation, odd test), selection by validation mean bytes, verdicts, controls. Hash in `logs/a3_decision_rule.log` (sha256 `1cb36415…`), logged before any run on real states |
| `a3_lib.py` | Shifted head copies, intervals in the original frame with extra FP32 slack, certificates at every t |
| `a3_modal.py` | Modal run on the existing exp01 states (`--smoke` for 512 states per mode). Refuses to run if the rule's hash is not logged |
| `a3_summarize.py` | Controls (C2, C3, C5, T0 = bitwise match to exp01 at t = 0, C6_shift), test-set tables with prompt-bootstrap intervals, verdict |
| `test_a3.py` | CPU self-tests on a synthetic head with a large mean row: bounds sound and tokens exact at every t, with and without the soft-cap; t = 0 bitwise equal to the exp01 path; dropping the add-back is caught |

```bash
python3 -I test_a3.py
modal run a3_modal.py --model qwen3_8b --smoke
modal run a3_modal.py --model qwen3_8b
modal volume get exp01-repro a3/qwen3_8b ./results/a3/
python3 -I a3_summarize.py results/a3/qwen3_8b results/qwen3_8b
```

Sarvam-30B: states from `sarvam_modal.py` under `a3_addendum_sarvam.json` (hash in `logs/a3_decision_rule.log`); results below.

### A3 results (full runs, 14 Nov 2026)

Summaries: `logs/a3_summary_qwen3_8b.txt`, `logs/a3_summary_gemma4_e4b.txt`. Controls pass on both under the rule's `invalid_if` (C2, C3, T0, C6_shift); no dither-certificate token mismatch at any t.

| Headline (dithered INT4 + INT8), test split | Qwen3-8B | Gemma 4 E4B |
|---|---|---|
| Verdict | NO_EFFECT (t* = 0.5) | HELPS (t* = 1.5) |
| Greedy mean bytes, t* − 0 [95%] | −0.0000 | −0.0015 [−0.0018, −0.0013] |
| Greedy p99 bytes, t = 0 → t* | 0.258 → 0.258 | 0.300 → 0.290 |
| Top-50 mean bytes, t* − 0 | −0.0049 | −0.0323 (median 0.485 → 0.437) |
| Hoeffding mean bound at t = 1 vs t = 0 | 0.960 | 0.927 |
| INT8 shortlist p99, greedy, t = 0 → 1 | 157 → 182 | 21,525 → 15,367 |
| Same at t = 4 (Kadav's Phi choice) | 2,835 | 227,566 |

Notes:
- The gain is real on Gemma but small for greedy and Gumbel, because the median already sits at the 0.258 floor. It is largest for top-50, where it moves Gemma 4 further below INT8 parity (0.501).
- t* is 0.5 to 1.5 everywhere; t = 4 is bad on both models.
- C5: Gemma 4's dither variance ratio leaves the 0.98–1.02 band at t = −1, 3 and 4 (up to 1.050) and stays in band at every selected t. Our guess, untested: with one dither pattern shared by all rows, a large shared component makes the rounding errors of different rows move together. The rule reports C5 but does not invalidate on it. a3_summarize.py first treated C5 as invalidating; it was changed to follow the rule's text.
- C6_shift passed but proved little: no state had |h·µ| above twice the largest bound, so nothing had to be caught (30 to 54 of 128 states were caught anyway).
- The smoke run's rise in RTN-cascade bytes at t = 1 (Qwen, English only) did not hold on the full run.
- One Qwen result file downloaded empty the first time and was fetched again.

### Sarvam-30B (071ae95e), 59,333 states, 14 Nov 2026

States: `sarvam_modal.py` (Vizuara's Sarvam lock and run-B prefill, our Flores prompts). Large result arrays stay on the Modal volume (`results/sarvam_30b`, `a3/sarvam_30b`) and are not copied into this Drive-synced folder; summaries are in `logs/summary_sarvam_30b_table.txt` and `logs/a3_summary_sarvam_30b.txt`.

Controls: C2 0, C3 0, T0 bitwise match, C1 alignment 98.8%. C7: every group's median target-script share is at least 0.915 except Bengali at T = 1 (0.573, flagged; it stays in the tables). C5 sits below the band (0.93 to 0.97) at several t including t = 0, meaning errors are smaller than the dither model predicts; no dither-certificate token mismatch. C6_shift caught 128 of 128 states in each mode (no state met the "must catch" threshold).

exp01 baseline (headline certificate, all states): greedy bytes median 0.291, p99 0.758, mean 0.390; Gumbel T = 1 0.326, 0.751, 0.404; top-50 0.757 (DEAD); min-p 0.470. Vizuara's run B on their prompts: greedy 0.316, 0.769, 0.406.

A3 verdict: HELPS, t* = 1 for every rule.

| Headline, test split | t = 0 (median, p99, mean) | t = 1 = t* | Δmean [95%] |
|---|---|---|---|
| greedy | 0.290, 0.758, 0.390 | 0.281, 0.677, 0.373 | −0.0170 [−0.0187, −0.0154] |
| Gumbel T = 1 | 0.322, 0.752, 0.403 | 0.306, 0.662, 0.384 | −0.0199 [−0.0219, −0.0179] |
| top-50 | 0.757, 0.861, 0.716 | 0.694, 0.833, 0.666 | −0.0505 [−0.0530, −0.0481] |
| min-p 0.05 | 0.459, 0.760, 0.471 | 0.421, 0.722, 0.443 | −0.0284 [−0.0314, −0.0256] |

Pre-registered predictions: bound at t = 1 is 0.883 of t = 0 (predicted 0.88–0.93, held); t* = 1 (held); HELPS with a larger reduction than Gemma 4 (held, 11 times larger); greedy median stays above 0.30 at t* (did not hold: 0.281, already 0.290 at t = 0 on these states); p99 above 0.5 (held, 0.677). t = 4 doubles the bounds and reads the whole head in INT8 at the median.

So the shift closes about a tenth of the gap to the 0.258 floor on mean bytes and trims the p99 by 0.08, but Sarvam's tail (p99 ≈ 0.68) still sits above the INT8 certificate (0.50).

## Shared-error ("gap-to-leader") probe, 14 Nov 2026 — exploratory, no rule

`shared_err_lib.py`, `shared_err_modal.py`, `test_shared_err.py`; raw numbers in `logs/shared_err_probe.json`. 1,024 random states per model (half greedy, half T = 1).

A bound on `z_v − z_leader` can only beat per-row bounds through error the two rows share. The probe removes that shared part exactly (the mean error row, or the top r = 1, 4, 16 directions of the error matrix, with r FP16 coefficients per row) and measures how much the residual bound narrows.

| Copy (t = 0) | Qwen3-8B | Gemma 4 E4B | Sarvam-30B |
|---|---|---|---|
| Error energy in the top 16 directions, INT4 / dithered INT4 | 0.8% | 3.5% | 2.7% |
| Mean residual bound after removing them, vs before | 0.991 | 0.984 | 0.975 |
| Hoeffding bound / residual bound (dithered INT4) | 0.24 | 0.31 | 0.24 |

- The quantization error is almost entirely row-specific, so removing it narrows bounds by at most 2.5%. The mean error row alone does nothing (≤0.1% of energy).
- Even after removal, the residual bound is about 4 times wider than the Hoeffding bound the headline certificate uses, so intersecting the two would not help.
- The largest bound over rows does shrink for INT8 at r = 16 (to 0.56–0.82), because a few outlier rows carry shared error, but the mean bound does not move.
- On Sarvam the mean shift (A3, t = 1) already removes most of the shared error (2.1% → 0.2% of energy at r = 1), consistent with A3's gain there.

Conclusion: not worth a full certificate run. Sarvam's tail does not come from shared error.

## B1: sub-Gaussian bound for the dithered head, 14 Nov 2026

Rule `subg_rule.json` (sha256 `5d8c3ac6…`, logged before any run). Code `subg_lib.py`, `subg_modal.py`, `subg_summarize.py`, `test_subg.py`. Summaries `logs/subg_summary_<model>.txt`; arrays on the Modal volume (`subg/<model>`).

The dithered error is exactly uniform, and U(−a, a) has E exp(λX) ≤ exp(λ²a²/6) (k!·6^k ≤ (2k+1)! term by term). So the logit error is sub-Gaussian with proxy Σ h²s²/12, which gives a bound 1/√3 = 0.577 of the Hoeffding term at the same δ = 1e-15 per row and under the same dither assumption. Bernstein's bound is never tighter than this one.

Controls: Hoeffding cascade identical to A3 on every state; no row exceeded the sub-Gaussian bound at any shift on any model (closest: 0.12 logits under the bound on Sarvam, 0.17 on Qwen, 0.58 on Gemma); no dithered-token mismatch; dither variance ratio 0.97–1.01.

Sub-Gaussian + INT8 at the validation-selected shift, test prompts (status over all states, table.py definitions):

| Model | Rule | t* | Status | Bytes median, p99, mean | Hoeffding + INT8, t = 0 |
|---|---|---|---|---|---|
| Sarvam-30B | greedy | 0.5 | UNDER_HALF (p99 0.512) | 0.258, 0.512, 0.279 | 0.290, 0.758, 0.390 |
| Sarvam-30B | Gumbel T = 1 | 0.5 | GATES_PASS | 0.258, 0.496, 0.281 | 0.322, 0.752, 0.403 |
| Sarvam-30B | top-50 | 1 | UNDER_HALF | 0.385, 0.694, 0.418 | 0.757, 0.861, 0.716 |
| Sarvam-30B | min-p 0.05 | 0.5 | UNDER_HALF | 0.260, 0.561, 0.301 | 0.459, 0.760, 0.471 |
| Gemma 4 E4B | greedy, Gumbel, min-p | 1.5 | GATES_PASS | 0.258, 0.258–0.260, 0.258 | p99 0.300–0.331 |
| Gemma 4 E4B | top-50 | 1 | UNDER_HALF (R p99 6,326 > 1% of V) | 0.258, 0.319, 0.271 | 0.485, 0.639, 0.435 |
| Qwen3-8B | all five rules, top-50 included | 0.5 | GATES_PASS | 0.258, ≤ 0.274, 0.258 | top-50 p99 0.532 |

Pre-registered prediction: Sarvam greedy p99 about 0.55 (got 0.512), greedy mean below 0.33 (got 0.279), status stays UNDER_HALF (held for greedy, did not hold for Gumbel T = 1, which passes). Sarvam greedy misses GATES_PASS only on p99 bytes (0.512 against < 0.5).

Caveat (added in review, 14 Nov): the Gumbel T = 1 pass is by 0.004 on the p99 (0.496), while T = 0.6 is at 0.514 and greedy at 0.512, so the order is not monotone in temperature. There is one dither draw, one sampling seed and no interval on the p99, and the T = 1 states include the Bengali group flagged by C7. Sarvam greedy and Gumbel are best read as on the gate, not on either side of it.

The shift and the bound combine: on Sarvam the sub-Gaussian cascade's best t is 0.5. At t = 1 the language groups' greedy means are 0.27–0.30 and p99s 0.48–0.52.

## L4 timing, first kernel version (14 Nov 2026, smoke and profiling only)

Protocol and prediction: `l4_protocol.json` (hash in `logs/a3_decision_rule.log`). Kernels: `l4_kernels.py` (fused dithered-INT4 first pass that also produces the sub-Gaussian variance term; row-gather GEMV for the INT8 and BF16 stages). Step: `l4_cert.py`. Bench: `l4_bench_modal.py`. Profiles: `l4_nsight_modal.py`, reports in `profiles/` (not published; `.nsys-rep` and `.ncu-rep` open in Nsight Systems and Nsight Compute).

Smoke, Qwen3-8B, 64 greedy states, batch 1 (`logs/l4_timing_qwen3_8b_smoke.json`); no token mismatches:

| | median ms | vs cuBLAS BF16 |
|---|---|---|
| cuBLAS BF16 head + argmax | 5.03 | 1 |
| INT4 first-pass kernel alone | 1.98 | 0.39 |
| full certified step | 4.85 | 0.96 |

Nsight Compute (`profiles/qwen3_8b_ncu.txt`, run with `--clock-control none` because Modal does not allow clock locking): the counters put the INT4 kernel at 92.7% of DRAM peak (278 GB/s, 1.79 ms, 56% achieved occupancy, 72 registers per thread). cuBLAS's BF16 GEMV takes 4.89 ms. It was not captured by Nsight Compute, so its rate of about 255 GB/s is bytes over time, not a counter.

Corrected in review, 14 Nov: an earlier version of this note concluded that the first pass is at the bandwidth roofline. That is not shown. Counted the same way as cuBLAS, the INT4 kernel reads 321 MB in 1.79 ms, which is 179 GB/s (155 to 173 GB/s in the full runs below): 0.258 of the bytes in 0.37 to 0.40 of the time, about two thirds of cuBLAS's rate. The counter's 278 GB/s over 1.79 ms is about 498 MB, against 321 MB read and 19 MB written. Either the kernel causes about 1.5 times the DRAM traffic it needs, or the counter over-reports; untested. At cuBLAS's rate the first pass would take about 1.28 ms, 0.26 of BF16. To settle it: the raw DRAM read and write byte counters for both kernels in one Nsight Compute run.

Nsight Systems (`profiles/qwen3_8b_nsys.txt`, 10 steps): the certified step's NVTX range has a median of 5.46 ms while its GPU kernels add up to about 2.1 ms. Each step makes about 100 kernel launches (about 9 µs of CPU each), 6 `cudaStreamSynchronize` calls and 10 small copies or memsets. The step is bound by the host: Python and launch overhead, plus a sync before each data-dependent row set.

Next: one graph-capturable step: bound and block maximum fused into the first-pass kernel, candidate rows compacted on the device into fixed-capacity buffers, one INT8 round and one BF16 round, and a single sync at the end that returns the token and an overflow flag (an overflow falls back to the current path).

### Full L4 timing, all three heads (14 Nov 2026)

`logs/l4_timing_<model>.json`; 512 greedy states per model at batch 1, 32 groups of 16 at batch 16; sub-Gaussian + INT8 at the B1 shifts. No token mismatch on any step (1,536 batch-1 steps and 1,536 states at batch 16).

| Model, batch | cuBLAS BF16 ms | certified ms (median, p99) | certified / BF16 (median, p99) | kernel / BF16 | bytes / BF16 (median, p99) |
|---|---|---|---|---|---|
| Qwen3-8B, 1 | 4.95 | 4.00, 4.56 | 0.81, 0.92 | 0.39 | 0.258, 0.258 |
| Gemma 4, 1 | 5.34 | 4.11, 5.19 | 0.77, 0.97 | 0.39 | 0.258, 0.260 |
| Sarvam-30B, 1 | 8.49 | 5.42, 7.46 | 0.64, 0.88 | 0.39 | 0.258, 0.53 |
| Qwen3-8B, 16 | 5.09 | 4.34, 4.94 | 0.85, 0.97 | 0.40 | 0.258, 0.258 |
| Gemma 4, 16 | 5.54 | 5.13, 5.41 | 0.92, 0.97 | 0.39 | 0.258, 0.259 |
| Sarvam-30B, 16 | 8.76 | 9.08, 10.13 | 1.04, 1.15 | 0.39 | 0.50, 0.55 |

- The first pass holds at 0.39 of BF16 on every head and batch size. The rest of the step costs about 2 ms of host time whatever the head, so the larger the head, the better the ratio (Sarvam 0.64 at batch 1).
- At batch 16 the rows are read once for the union of all 16 states. On Sarvam that union is about 125,000 INT8 rows at the median, half the vocabulary, and the step is slower than BF16. Batched certification needs per-state row sets or a smaller union.
- On Qwen3-8B and Gemma 4 the batch-1 mean (4.84 and 5.01 ms) is above the p99, so fewer than 1% of the steps are very slow. Likely cause, untested: Triton compiling a new specialization when the gather kernel's row count `N` first hits a new case (1, or a multiple of 16). Marking `N` as not specialized would test this.
- Against the prediction: the kernel at 0.30–0.40 held (0.39). The batch-1 step at 0.45–0.65 held only for Sarvam (0.64); Qwen and Gemma came in at 0.77–0.81, because of fixed host time. "Sarvam p99 exceeds BF16" did not hold (0.88). Batch 16 at 0.6–0.9 held for Qwen (0.85) and not for Gemma (0.92) or Sarvam (1.04).
- Run-to-run variation (added in review, 14 Nov). Two later runs of the same bench per model (`logs/l4_graph_timing_all.log`, then `logs/l4_graph_timing_sarvam_rerun.log` and `logs/l4_graph_timing_qg_rerun.log`) gave different certified-step medians, while the BF16 head stayed within 1% and the kernel within 3%:

  | Certified / BF16, median | this table | later runs |
  |---|---|---|
  | Qwen3-8B, 1 | 0.81 | 0.86, 0.85 |
  | Gemma 4, 1 | 0.77 | 0.66, 0.79 |
  | Sarvam-30B, 1 | 0.64 | 0.56, 0.66 |
  | Qwen3-8B, 16 | 0.85 | 0.91, 0.91 |
  | Gemma 4, 16 | 0.92 | 0.82, 0.95 |
  | Sarvam-30B, 16 | 1.04 | 0.99, 1.04 |

  The host-bound part of the step moves by up to 15% between Modal containers, so the certified ratios are good to about ±0.1. Against the predicted bands, Qwen and Gemma miss at batch 1 and Sarvam misses at batch 16 in every run. Sarvam at batch 1 (0.56 to 0.66) and Qwen and Gemma at batch 16 fall on both sides of the band across runs. Sarvam's batch-1 p99 stays under BF16 in all three (0.82 to 0.90).

### Single-sync, graph-captured step (14 Nov 2026)

`l4_graph.py`, timed by `l4_bench_modal.py` (`logs/l4_graph_timing_<model>.json`; same states and shifts as above). The step is one CUDA graph:
1. a fused first pass (logits, sub-Gaussian bound, `L`, `U`, and the per-state maximum of `L` by atomic max);
2. device-side compaction of candidate rows into fixed buffers;
3. one INT8 round and one BF16 round;
4. a single host sync for the token and the two row counts.

One round per stage is enough because the maximum of `L` only rises, so a row needed after a stage was already needed before it. If a buffer overflows, a flag sends the step to the eager path. The INT8 buffer holds the whole vocabulary and the BF16 buffer 1,024 rows; with a 32,768-row INT8 buffer, Sarvam overflowed on 14% of batch-1 steps and 29 of 32 batch-16 steps.

No token mismatch and no fallback on any model or batch size:

| Model, batch | cuBLAS BF16 ms | graph step ms (median, p99, mean) | graph / BF16 (median, p99, mean) | eager / BF16 (median) |
|---|---|---|---|---|
| Qwen3-8B, 1 | 4.96 | 2.04, 2.11, 2.05 | 0.41, 0.42, 0.41 | 0.85 |
| Gemma 4, 1 | 5.34 | 2.19, 2.22, 2.19 | 0.41, 0.42, 0.41 | 0.79 |
| Sarvam-30B, 1 | 8.50 | 3.38, 5.79, 3.61 | 0.40, 0.68, 0.42 | 0.66 |
| Qwen3-8B, 16 | 5.12 | 2.14, 2.20, 2.14 | 0.42, 0.43, 0.42 | 0.91 |
| Gemma 4, 16 | 5.55 | 2.29, 2.34, 2.29 | 0.41, 0.42, 0.41 | 0.95 |
| Sarvam-30B, 16 | 8.77 | 5.90, 6.33, 5.65 | 0.67, 0.72, 0.64 | 1.04 |

- On Qwen3-8B and Gemma 4 the step now costs about 0.1 ms more than the first-pass kernel alone (1.96–2.14 ms), at both batch sizes.
- Sarvam's batch-1 median matches the others. Its tail follows its INT8 rows (p99 144,000 rows, about 0.53 of the BF16 head's bytes).
- At batch 16 Sarvam reads the union of 16 candidate sets (median about 125,000 INT8 rows), so it stays at 0.67. Per-state row sets would be the next step for batched Sarvam.
- This is the head alone. End-to-end decode time depends on the head's share of each step (Sarvam about 31% of weight bytes with a BF16 body, 64% with a 4-bit one; see the lab note), and that has not been measured.

## F1: flip rate of the uncertified heads (14 Nov 2026)

Question: the certified step costs about 5% more time than the 4-bit first pass alone. What does that buy? F1 counts how often a low-bit copy, used with no certificate, emits a different token from the reference.

Rule `flip_rule.json` (sha256 `41c0a631…`, logged before any run on real states). Code `flip_lib.py`, `flip_modal.py`, `flip_summarize.py`, `test_flip.py`. Summaries `logs/flip_summary_<model>.txt`; per-state arrays in `results/flip/<model>` (not published). Every exp01 state, no subsampling. Copies: INT4 round-to-nearest g128, dithered INT4 g128 and INT8 per row, at t = 0 and at the model's B1 greedy shift. Nothing is selected on these states.

```bash
python3 -I test_flip.py
modal run flip_modal.py::main --model qwen3_8b --smoke
modal run flip_modal.py::main --model qwen3_8b
modal volume get exp01-repro flip/qwen3_8b ./results/flip/
python3 -I flip_summarize.py results/flip/qwen3_8b "Qwen3-8B"
```

Controls:
- **F0: pass.** The largest logit error of every state, copy and shift equals the value A3 stored, bit for bit (0 of 176,359 states differ), so the copies and states are A3's.
- **F1: did not match as written, because the rule misdescribed C1.** The rule expected the reference token to equal the emitted token as often as exp01's C1 (1.0000, 99.8%, 98.8%). It measured 99.59% (Qwen3-8B), 99.57% (Gemma 4) and 97.44% (Sarvam-30B). C1 compares the emitted token with the argmax of the BF16 product; F1 compares it with the reference, which accumulates in FP32. The gap is itself a result, measured under a hashed addendum below: the stock BF16 head does not reproduce the reference either.
- **F2: pass** (`test_flip.py`): an exact copy never flips, rolled rows flip at least 91% of states, and flips match a plain per-state loop.

Flip rate per state, with a 95% interval from resampling prompts. "First pass alone" is the dithered copy at the B1 shift, the one the certified step reads:

| Model | Rule | INT4 round-to-nearest, t = 0 | First pass alone (dithered, B1 shift) | INT8, t = 0 | Certified |
|---|---|---|---|---|---|
| Qwen3-8B | greedy | 1.89% [1.67, 2.13] | 2.04% [1.81, 2.27] | 0.14% [0.10, 0.19] | 0 |
| Qwen3-8B | Gumbel T = 1 | 2.07% [1.84, 2.31] | 2.31% [2.07, 2.56] | 0.14% [0.10, 0.19] | 0 |
| Gemma 4 E4B | greedy | 2.49% [2.24, 2.75] | 2.66% [2.40, 2.93] | 0.18% [0.12, 0.23] | 0 |
| Gemma 4 E4B | Gumbel T = 1 | 2.59% [2.33, 2.86] | 2.85% [2.58, 3.13] | 0.24% [0.18, 0.29] | 0 |
| Sarvam-30B | greedy | 22.87% [21.21, 24.63] | 8.84% [8.08, 9.67] | 1.18% [1.01, 1.37] | 0 |
| Sarvam-30B | Gumbel T = 1 | 27.59% [25.65, 29.58] | 11.52% [10.59, 12.52] | 1.46% [1.25, 1.68] | 0 |

Gumbel T = 0.6, top-50 and min-p 0.05 are within 0.5 points of Gumbel T = 1 in every column of this table (`logs/flip_summary_<model>.txt`).

| Greedy, of 240 prompts | Qwen3-8B | Gemma 4 E4B | Sarvam-30B |
|---|---|---|---|
| Prompts with at least one flip, first pass alone | 193 | 188 | 230 |
| Same, INT8 | 34 | 41 | 152 |
| Gap on flipped states, first pass alone: median, p90 (logits) | 0.27, 0.78 | 0.35, 1.03 | 0.92, 4.99 |
| Share of those flips with a gap above 1 logit | 3.2% | 11.2% | 47.6% |
| Median largest error over rows, dithered / round-to-nearest (logits, t = 0) | 2.37 / 2.20 | 5.39 / 4.91 | 5.09 / 10.00 |

- **What the certificate buys.** Without it, the first pass emits a different token on about 2% of steps on Qwen3-8B, 2.7% on Gemma 4 and 9% on Sarvam-30B, and about four prompts in five depart from the reference somewhere in 128 tokens (all but ten on Sarvam). States follow the reference trajectory, so this is the chance of a first departure, not a count of wrong tokens after it.
- **On Qwen3-8B and Gemma 4 most flips are near-ties**: the emitted token's reference score is a median 0.27 to 0.35 logits below the reference token's. On Sarvam they are not: the median is 0.92 logits and nearly half are above 1.
- **Round-to-nearest INT4 breaks on Sarvam**: 23% of greedy tokens flip, with a median gap of 12.4 logits, and the shift makes it worse (44% at t = 0.5, median largest error 10.0 → 17.9 logits). The dithered copy does not have this failure. Untested guess: with absmax scales most of a Sarvam row rounds to zero, and the shared mean row was acting as an accidental dither that the shift removes.
- **The shift does not reduce flips.** Dithered greedy flips go 2.26% → 2.04% on Qwen, 2.74% → 2.66% on Gemma and 8.00% → 8.84% on Sarvam. The Sarvam rise is unexplained: A3's mean bound shrinks there, but the median largest error does not (5.09 → 5.12 logits).
- **INT8 flips 0.14% to 0.18% of greedy steps on Qwen and Gemma, 1.2% on Sarvam.** On Qwen and Sarvam that is fewer than the stock BF16 head's flips against the same reference (addendum below).
- **Truncation rules.** The top-50 set differs on almost every state for both INT4 copies (median 6 to 26 tokens of symmetric difference), but the emitted token is outside the reference set on at most 0.01% of states on Qwen and Gemma (0.17% on Sarvam, dithered; 11% for round-to-nearest). For min-p 0.05 it is outside on 0.2% to 0.4% (3.4% on Sarvam).
- By language group (greedy, first pass alone): code flips least on Qwen and Gemma (0.2%, 0.04%); English is the worst group on Sarvam (18.8%) and on Qwen (3.7%); Bengali and Tamil are worst on Gemma (4.0%, 4.2%).

Against the predictions: the Qwen and Gemma ones held and every Sarvam one missed.

| Prediction | Qwen3-8B | Gemma 4 | Sarvam-30B |
|---|---|---|---|
| Greedy, round-to-nearest, t = 0: 0.5–3% (Qwen), 1–6% (others) | held (1.89%) | held (2.49%) | missed (22.87%) |
| Dithered within a factor of 1.5 of round-to-nearest | held (1.19) | held (1.10) | missed (0.35) |
| Gumbel T = 1 higher than greedy, in 2–15% | held for round-to-nearest; dithered equal (2.26%) | held | higher, but round-to-nearest out of range (27.6%) |
| INT8 greedy below 0.3% | held | held | missed (1.18%) |
| More than half the greedy prompts flip somewhere | held (178) | held (187) | held (240) |
| Median gap on flipped greedy states below 0.5 logits | held (0.27) | held (0.32) | missed (12.4) |

The language prediction (each Indian-language group flips more than English on Gemma 4) did not hold: Hindi is at 1.87% against English at 1.92%.

Limits: these are this project's simple quantizers (absmax scales, no calibration); a calibrated 4-bit scheme would flip less. One dither draw and one noise seed.

### F1 addendum: the stock BF16 head against the reference (14 Nov 2026)

`flip_addendum_bf16.json` (sha256 `8f2e0e91…`, logged after the F1 results were read and before this ran). `flip_modal.py::bf16`, `flip_lib.run_bf16_chunk`, `flip_bf16_summarize.py`; summaries `logs/flip_bf16_summary_<model>.txt`. The copy is the BF16 product exactly as exp01's C1 computes it, standing for the stock head. `flip_rule.json` is unchanged.

Controls: the corrected F1 passes. The BF16-product argmax equals the emitted token on exactly the number of greedy states exp01 stored, in every job: 28,940 of 28,940 (Qwen3-8B), 29,581 of 29,641 (Gemma 4), 29,474 of 29,826 (Sarvam-30B). Reference tokens equal the F1 run's on every state and rule.

| Greedy, against the FP32 reference | Qwen3-8B | Gemma 4 E4B | Sarvam-30B |
|---|---|---|---|
| Stock BF16 head, flips | 0.41% [0.33, 0.49] (118) | 0.23% [0.17, 0.29] (68) | 2.54% [2.26, 2.83] (758) |
| INT8, t = 0 | 0.14% | 0.18% | 1.18% |
| First pass alone (dithered INT4, B1 shift) | 2.04% | 2.66% | 8.84% |
| Certified | 0 | 0 | 0 |
| Flips that are exact ties in the BF16 logits | 118 of 118 | 68 of 68 | 758 of 758 |
| States with tied top BF16 logits | 0.82% | 0.40% | 4.90% |
| Prompts (of 240) with at least one flip | 89 | 54 | 189 |
| Gap on flipped states: median, max (logits) | 0.05, 0.23 | 0.02, 0.07 | 0.20, 1.68 |

- **The stock BF16 head does not emit the reference tokens.** It differs on 0.2% to 0.4% of greedy steps on Qwen and Gemma and 2.5% on Sarvam, and on Sarvam 189 of 240 prompts depart from the reference within 128 tokens.
- **Every one of these flips is a tie.** BF16 logits are coarse (0.25 apart near 37, Qwen's typical top logit; on Sarvam tokens up to 1.7 logits apart in the reference come out tied, and the logit scale there was not checked), the two largest come out equal, and argmax takes the lower token id. On Sarvam the top two BF16 logits are tied on 4.9% of greedy states, so the stock model's greedy output is decided by token order that often.
- **What "exact" means here.** The certified head reproduces the FP32-accumulated reference, not the stock BF16 run. The reference is the better-defined target, since it does not depend on tie order or on the BLAS path, but a claim of "the same tokens as the BF16 model" would be wrong on the shares above. Reproducing the stock run instead would mean certifying its rounding and tie order.
- INT8 with FP32 accumulation is closer to the reference than the stock head on Qwen and Sarvam; on Gemma the two are within each other's intervals (68 against 52 flips).
- Sampling rules: 0.25% to 0.32% (Qwen), 0.10% to 0.16% (Gemma) and 2.0% to 2.5% (Sarvam), with the same noise.

Against the addendum's predictions: Qwen exactly 118 flips held. Sarvam between 1.5% and 3.5% and above INT8 held. Gemma between 0.3% and 0.6% missed (0.23%). Median gap below 0.15 logits held on Qwen and Gemma and missed on Sarvam (0.20). "More than a third of flips are ties" held (all of them). "More flips than INT8 on every model" held in counts, but not outside the intervals on Gemma.

Limits: the BF16 product is the analysis image's, on an A100, 128 states at a time; another BLAS path or batch shape can round differently (Sarvam's generating run already differs from it on 1.2% of states). Gemma 4's soft-cap is applied in FP32 to the BF16 logits here, in BF16 in the stock model.

## K1: why Sarvam's certificate is wide (14 Nov 2026, exploratory)

Prompted by a note from the lab (14 Nov, not public) about Sarvam-30B's shared head offset. A3 above found that centring the head rows narrows the bound by only 12% and leaves Sarvam's tail, and the shared-error probe found the quantization error to be row-specific. K1 asks what does make the certificate wide.

Rule `crowd_rule.json` (hashed before any run as sha256 `2b036300…`; exploratory, predictions but no verdict rule). The published copy leaves the lab's figures out of its `context` field and differs from the hashed one in that field only; both hashes are in `logs/a3_decision_rule.log`, and the hashed copy will be published once the lab's note is. Code `crowd_lib.py`, `crowd_modal.py`, `crowd_summarize.py`, `test_crowd.py`. Summary `logs/crowd_summary.txt`; arrays in `results/crowd/<model>` (not published). All greedy states of the three models.

Control K0: pass. The shortlist recomputed here equals A3's (Hoeffding) and B1's (sub-Gaussian) stored value on every one of the 88,407 states.

| Median over greedy states | Qwen3-8B | Gemma 4 E4B | Sarvam-30B |
|---|---|---|---|
| Raw top logit | 37.7 | 35.5 | 112.6 |
| Offset `h · µ` (share of the raw top logit) | −3.0 (−0.08) | −23.3 (−0.66) | 85.4 (0.77) |
| Hoeffding bound, median row, t = 0 (logits) | 7.0 | 14.9 | 10.4 |
| Sub-Gaussian bound, median row, B1 shift | 4.0 | 8.2 | 5.5 |
| Top logit minus the 2nd, 1,000th largest | 8.5, 31.4 | 15.8, 39.5 (raw) | 5.5, 19.5 |
| Top logit minus the median logit | 40.7 | 58.4 (raw) | 28.8 |
| The same in units of 2 × the Hoeffding bound | 2.93 | 2.13 | 1.29 |
| 1,000th largest, in the same units | 2.27 | 1.44 | 0.84 |
| Tokens within 16, 32 logits of the top (capped for Gemma) | 7, 1,422 | 10, 4,041 | 101, 207,536 |
| Shortlist with the copy's logits as centres (Hoeffding, t = 0) | 3 | 24 | 17,262 |
| Shortlist with exact logits as centres, same widths | 3 | 23 | 15,948 |
| Entropy at T = 1 (nats), top probability | 0.003, 1.000 | 0.010, 0.999 | 0.053, 0.993 |

- **The bound is not unusually wide on Sarvam.** In logits it sits between Qwen's and Gemma's (10.4 against 7.0 and 14.9), and bound over true largest error is about 6 on all three (exp01, A3).
- **Estimation error is not the cost.** Replacing the copy's logits by the exact ones as interval centres leaves the shortlist unchanged (median ratio 1.00 on every model). The shortlist is set by the width of the intervals alone.
- **Sarvam's logits are packed under the leader.** The whole vocabulary lies closer to the top: the median token is 28.8 logits below it, against 40.7 (Qwen) and 58.4 (Gemma, raw). At the median state 79% of Sarvam's vocabulary is within 32 logits of the top, against 0.9% and 1.5%. In units of the interval's width, the median token is 1.29 widths away and the 1,000th token 0.84. So a slightly wider bound or a slightly lower top logit pulls in a large part of the vocabulary, which is the cliff in Sarvam's tail (p99 shortlist: every row).
- **This is not uncertainty about the next token.** Sarvam's top probability is 0.993 at the median and the runner-up is 5.5 logits down. The rows that cost bytes are tokens 15 to 30 logits below the top, with no chance of being sampled.
- **The shared offset is large on Sarvam**: 85.4 of 112.6, a share of 0.77. Gemma 4 has a large offset of the opposite sign (−23.3 against a top logit of 35.5).
- **The offset explains the stock head's ties** (F1 addendum). BF16 is coarser at larger magnitudes, and Sarvam's tie share rises with the raw top logit: 1.6% of states below 64, 4.7% from 64 to 128, 6.7% above 128. On Qwen and Gemma almost every state is below 64.
- By language on Sarvam: English has the largest offset (163), the largest top logit (194), the widest bound (13.7 against 9 to 10) and the largest shortlist (median 133,244 rows).

A lead, post hoc (tested in K2 below, where it holds): across Sarvam's states the bound follows the offset (Spearman 0.83) and the shortlist follows the bound (0.86), not the gap to the 1,000th token (−0.26). The bound grows with the length of `h` in each group, and the part of `h` along `µ` adds the same amount to every logit: it widens every interval without separating any two tokens. Centring the head rows does not remove that part of `h`, which would explain why centring gains so little in A3. If so, the fix is on the state side: store each row's quantization-error projection on `µ̂` (one number per row, exact, computed offline), subtract `(h · µ̂)` times it from the first-pass logit, and bound only the error against the part of `h` orthogonal to `µ`. The correlations cannot separate this from `h` simply being longer in those states (Qwen and Gemma show 0.61 and 0.68); the probe that would is the share of each group's `‖h‖²` that lies along `µ`.

Against the predictions: all seven held as written (offset share 0.77; Sarvam's bound 0.70 of Gemma's; shortlist ratio 1.00; headroom of the 1,000th token below 1 on Sarvam only; 101 tokens within 16 logits against 10 and 7; English worst; median raw top above 64 with more ties in the upper half, 6.4% against 3.4%). Two notes on them. The offset prediction said "below 0.5" for Gemma and did not anticipate a large negative offset. And the rule's headroom divides capped gaps by raw bounds, which is wrong for Gemma's soft-cap: as written Gemma is 1.02, and with gaps mapped back to raw logits it is 1.44, the figure in the table.

Limits: greedy states only; medians over states hide the tail; the lead above rests on correlations found after the fact.

## K2: cancelling the error along the state's dominant directions (14 Nov 2026)

K1's lead: the part of the state `h` that lies along a few fixed directions widens every interval without separating any two tokens. K2 removes it exactly.

For the dithered copy with error rows `e_v`, any fixed directions `P` (d × K) and `α = Pᵀh`:

`h · e_v = α · (Pᵀe_v) + (h − Pα) · e_v`

The first term is K stored numbers per row (`r_v = Pᵀe_v`, computed offline) times K numbers per state, so it is subtracted from the first-pass logit exactly. Only the second term is bounded, by B1's sub-Gaussian bound with `h` replaced by `h − Pα`. The dither argument is unchanged because `h − Pα` does not depend on the dither. Storing `r` in FP32 costs 2K/d of the BF16 head's bytes, which is charged to every state.

Rule `hdir_rule.json` (sha256 `76d198b2…`, logged before any run). Code `hdir_lib.py`, `hdir_modal.py`, `hdir_summarize.py`, `test_hdir.py`. Summaries `logs/hdir_summary_<model>.txt`; arrays in `results/hdir/<model>` (not published). Direction sets: `mu` (the unit mean head row, K = 1) and `pca1`, `pca4`, `pca16`, `pca64` (top eigenvectors of the uncentred second moment of the greedy states of the even prompt ids). Tables are on the odd prompt ids; status is over all states, as in B1. Primary sets named before the run: `mu` and `pca16`.

Controls, all three models: with no correction, bytes and S equal B1's on every one of the 176,359 states (H0). No row exceeded a corrected bound at any shift or direction set (H1; closest 0.11 logits under on Qwen, 0.17 on Sarvam, 0.46 on Gemma). No token mismatch (H2). CPU test passes, including a planted fault (H3).

| Calibration states | Qwen3-8B | Gemma 4 E4B | Sarvam-30B |
|---|---|---|---|
| Share of the second moment in the top 1, 4, 16 eigenvectors | 0.14, 0.32, 0.47 | 0.22, 0.52, 0.61 | 0.74, 0.84, 0.89 |
| Cosine of `µ` with the top eigenvector | +0.19 | −0.33 | +0.78 |
| Share of `‖h‖²` along `µ`, median greedy state | 0.003 | 0.034 | 0.378 |

Sarvam-30B, sub-Gaussian + INT8 at t = 0.5 (B1's configuration), test prompts:

| Rule | Directions | Share of `‖h‖²` removed | Bound vs none | S median, p99 | Status | Bytes median, p99, mean | Δmean vs none [95%] |
|---|---|---|---|---|---|---|---|
| greedy | none (B1) | 0 | 1 | 149, 133,642 | UNDER_HALF | 0.258, 0.512, 0.279 | |
| greedy | `mu`, K = 1 | 0.38 | 0.80 | 30, 23,954 | GATES_PASS | 0.258, 0.305, 0.260 | −0.0183 [−0.0211, −0.0157] |
| greedy | `pca1`, K = 1 | 0.64 | 0.62 | 3, 614 | GATES_PASS | 0.258, 0.260, 0.258 | −0.0203 [−0.0233, −0.0174] |
| greedy | `pca4` | 0.80 | 0.46 | 2, 167 | GATES_PASS | 0.260, 0.260, 0.260 | −0.0189 [−0.0219, −0.0161] |
| greedy | `pca16` | 0.86 | 0.40 | 2, 54 | GATES_PASS | 0.266, 0.266, 0.266 | −0.0131 [−0.0161, −0.0103] |
| Gumbel T = 1 | none (B1) | 0 | 1 | 326, 123,945 | GATES_PASS | 0.258, 0.496, 0.281 | |
| Gumbel T = 1 | `mu`, K = 1 | 0.42 | 0.77 | 65, 24,590 | GATES_PASS | 0.258, 0.306, 0.261 | −0.0198 [−0.0227, −0.0170] |
| Gumbel T = 1 | `pca1`, K = 1 | 0.70 | 0.57 | 5, 1,006 | GATES_PASS | 0.258, 0.260, 0.258 | −0.0222 [−0.0253, −0.0192] |
| Gumbel T = 1 | `pca16` | 0.87 | 0.38 | 2, 150 | GATES_PASS | 0.266, 0.266, 0.266 | −0.0150 [−0.0182, −0.0120] |

- **Sarvam's tail is gone with one stored number per row.** With the states' top eigenvector the greedy p99 falls from 0.512 to 0.260 of the BF16 head and the p99 shortlist from 133,642 rows to 614. Over all 29,826 greedy states the largest byte count is 0.332 (0.562 without) and one state is above 0.30 (3,650 without). Sarvam now sits at the 0.258 floor like the other two heads.
- **The mean head row alone is enough to pass the gates** (p99 0.305, no states needed to fit it), but it is not the best single direction: it removes 38% of `‖h‖²` against 64% for the top eigenvector, and only 10% to 16% on Tamil and Telugu.
- **This is why centring the head gained so little.** On Sarvam 74% of the states' second moment lies in one direction, which has cosine 0.78 with `µ`. Centring changes the rows; this component is in the state, and it enters the bound through `‖h‖`.
- **The shift is no longer needed on Sarvam.** At t = 0 with `pca1`: greedy 0.258, 0.260, 0.258, the same as at t = 0.5.
- **More directions buy little here and cost bytes.** `pca4` is flat at 0.260; `pca16` (0.266) and `pca64` (0.289) are above the floor because of their own storage, and `pca64`'s mean is worse than no correction.
- **Qwen3-8B and Gemma 4 gain nothing in bytes** (already at the floor) and pay the storage: +0.0005 and +0.0008 for K = 1. Their bounds do narrow with more directions (`pca4`: 0.87 and 0.73 of the uncorrected bound; Gemma's p99 shortlist 285 → 28), which matters for time and for batched steps, not for these byte counts. The addendum below shows that most of this narrowing does not carry over to a language the directions were not fitted on.
- By language on Sarvam (greedy, t = 0.5, all states): every group's p99 with `mu` is 0.27 to 0.32 (0.46 to 0.53 without), and 0.266 with `pca16`.

Against the predictions: share along `µ` on Sarvam 0.3–0.7, above Qwen's, English largest: held (0.378; 0.003; English 0.561). Top eigenvector above 0.4 of the second moment: held (0.74); its cosine with `µ` above 0.8: missed (0.78). Bound with `mu` 0.55–0.85 and with `pca16` below 0.6: held (0.80, 0.40). Sarvam greedy with `mu`: p99 below 0.5 over all states and mean below 0.275, `pca16` p99 below 0.40: held (0.304, 0.260, 0.266). Qwen and Gemma "medians stay at 0.258 and means change by less than 0.01 for `mu` and `pca16`": held for `mu`, missed for `pca16`, whose own storage is 0.008 (Qwen) and 0.0125 (Gemma). Controls: held. The falsifier was not triggered.

Limits:
- The eigenvectors were fitted on half of the same 240 prompts' states and tested on the other half. Directions fitted on unrelated text are untested. `mu` needs no fitting and already passes.
- Greedy and Gumbel T = 1 only. Gumbel T = 0.6, top-50 and min-p 0.05 were not rerun.
- Byte counts here; the corrected step is timed on an L4 in the last section of this file.
- `r` is kept in FP32. One dither draw and one noise seed.
- Five direction sets were run; `mu` and `pca16` were the pre-named primaries, and `pca1` is the best by bytes. Treat `pca1`'s margin over `pca4` as selected after the fact.

### K2 addendum: the other rules, and directions fitted without the state's language (14 Nov 2026)

`hdir_addendum_robust.json` (sha256 `44d0af47…`, logged after K2's results were read and before this ran). `hdir_modal.py::robust`, `hdir_lib.run_chunk_sets`, `hdir_robust_summarize.py`; summaries `logs/hdir_robust_summary_<model>.txt`. All five rules, at t = 0 and the B1 greedy shift. New sets `lolo1` and `lolo4`: for each of the six language groups, eigenvectors fitted on the other five groups' greedy states, so every state is corrected with directions that never saw its language.

Controls, all three models: with no correction every rule reproduces B1 on all 176,359 states; no bound violation (closest 0.19 logits under on Qwen, 0.23 on Sarvam, 0.66 on Gemma); no token mismatch.

Sarvam-30B at t = 0.5, test prompts, one direction fitted without the state's language (`lolo1`) against none:

| Rule | none: bytes median, p99, mean | `lolo1`: bytes median, p99, mean | Status none → `lolo1` | S p99 none → `lolo1` |
|---|---|---|---|---|
| greedy | 0.258, 0.512, 0.279 | 0.258, 0.260, 0.258 | UNDER_HALF → GATES_PASS | 133,642 → 673 |
| Gumbel T = 0.6 | 0.258, 0.514, 0.283 | 0.258, 0.260, 0.258 | UNDER_HALF → GATES_PASS | 133,949 → 1,209 |
| Gumbel T = 1 | 0.258, 0.496, 0.281 | 0.258, 0.260, 0.258 | GATES_PASS → GATES_PASS | 123,945 → 1,093 |
| min-p 0.05 | 0.260, 0.561, 0.301 | 0.258, 0.271, 0.259 | UNDER_HALF → GATES_PASS | 158,350 → 8,500 |
| top-50 | 0.386, 0.697, 0.420 | 0.270, 0.370, 0.277 | UNDER_HALF → UNDER_HALF | 182,431 → 55,243 |

- **The first direction transfers.** On Sarvam each left-out fit's top eigenvector has cosine 0.9987 to 0.9995 with K2's, and `lolo1` matches `pca1` on every rule (greedy p99 0.260 for both; per group, p99 0.258 to 0.260 either way). The mean head row, which needs no states at all, also passes greedy, both Gumbel temperatures and min-p (p99 0.305, 0.316, 0.306, 0.411).
- **Directions two to four do not transfer.** `lolo4` is no better than `lolo1` (bound 0.61 against 0.61 of the uncorrected one; in-sample `pca4` reaches 0.46). They are language-specific, so K2's in-sample gain from `pca4` over `pca1` should not be expected on new text. The robust recipe is K = 1.
- **Sarvam now passes four of the five rules** with one out-of-language direction. Top-50 improves from a mean of 0.420 to 0.277 but stays UNDER_HALF: its R p99 is 10,145 rows, above 1% of the vocabulary, the same miss Gemma 4 has.
- **Gemma 4 and Qwen3-8B: the narrowing was mostly in-sample.** With left-out directions the bound is 0.96 (Gemma, `lolo4`) and 0.92 (Qwen) of the uncorrected one, against 0.73 and 0.87 in-sample. On Gemma the left-out top eigenvector's cosine with K2's falls to 0.82 (Telugu) and 0.89 (Bengali). Gemma's top-50 passes with in-sample `pca1` and `pca4` and stays UNDER_HALF with `mu`, `lolo1` and `lolo4`. Their states have no single dominant direction (14% and 22% of the second moment in the first eigenvector, against Sarvam's 74%), so there is nothing shared to cancel.

Against the addendum's predictions: transfer on Sarvam (cosine above 0.99, `lolo1` p99 within 0.01 of `pca1`): held. Gumbel T = 0.6 passes with p99 below 0.30: held (0.260). Min-p passes with p99 below 0.35: held (0.270 with `pca1`, 0.271 with `lolo1`). Top-50 with `pca4` median below 0.30 and mean below 0.33: held (0.263, 0.268). Gemma top-50 stays UNDER_HALF with every set: missed for in-sample `pca1` and `pca4`, which pass; the stated reason (R is set by the INT8 stage alone) was wrong, since R p99 falls from 4,833 to 824 with `pca4`. Controls: held.

Limits: leaving out a language keeps the same model, chat template and kind of prompt; it tests transfer across language and domain, not across tasks or lengths. The uncorrected top-50 row is at t = 0.5, not B1's selected t = 1 (0.385, 0.694, 0.418). Byte counts only.

## L4 timing with the K2 correction (14 Nov 2026)

Protocol and predictions: `l4_corr_protocol.json` (sha256 `bf4757b6…`, logged before any corrected kernel ran). Code `l4_corr.py`: `CorrHead` adds `r` and an eager corrected step to `l4_cert.CertHead`; `GraphStepCorr` is `l4_graph.GraphStep` with a first-pass kernel that subtracts `α r_v` and bounds only the remainder. The selection, INT8 and BF16 kernels are `l4_graph`'s, unchanged, and the code timed earlier is not modified. Bench `l4_corr_bench_modal.py`, results `logs/l4_corr_timing_<model>.json`. Same L4, states, baseline and timing method as the sections above; each step times the uncorrected graph step, the two corrected ones and the BF16 head on the same state.

Checks, on every step of every variant (3 heads × 2 batch sizes × 3 variants; 1,536 batch-1 steps and 1,536 states at batch 16 per variant): no token mismatch against the FP32 reference; no row of any state outside the interval the step left in its buffers; the graph step's INT8 and BF16 row counts equal the eager loop's; no fallback.

Sarvam-30B, graph-captured step (K = 1):

| Batch | Correction | Step ms (median, p99, mean) | Step / BF16 (median, p99, mean) | INT8 rows (median, p99) | Bytes / BF16 (median, p99) |
|---|---|---|---|---|---|
| 1 | none | 3.39, 5.80, 3.62 | 0.40, 0.68, 0.43 | 258, 144,351 | 0.258, 0.534 |
| 1 | `mu` | 3.41, 4.02, 3.44 | 0.40, 0.47, 0.40 | 58, 34,848 | 0.258, 0.325 |
| 1 | `pca1` | 3.41, 3.46, 3.41 | 0.40, 0.41, 0.40 | 5, 889 | 0.258, 0.260 |
| 16 | none | 5.87, 6.28, 5.60 | 0.66, 0.71, 0.63 | 124,972, 150,752 | 0.497, 0.547 |
| 16 | `mu` | 3.99, 5.41, 4.09 | 0.45, 0.61, 0.46 | 18,292, 92,437 | 0.294, 0.436 |
| 16 | `pca1` | 3.62, 3.89, 3.63 | 0.41, 0.44, 0.41 | 534, 11,646 | 0.260, 0.281 |

The cuBLAS BF16 head takes 8.51 ms at batch 1 and 8.88 ms at batch 16.

- **Sarvam's tail in time is gone.** At batch 1 the p99 step falls from 0.68 to 0.41 of the BF16 head and the slowest of the 512 steps from 5.88 ms to 3.71 ms.
- **Batched Sarvam is fixed.** The union of 16 shortlists falls from 124,972 rows (half the vocabulary) to 534, and the step from 0.66 to 0.41 of BF16. With the mean head row alone it is 0.45.
- **All three heads now sit at about 0.40 to 0.41 of the BF16 head at both batch sizes**, taking the corrected step for Sarvam and the uncorrected one for Qwen3-8B and Gemma 4, which do not need it.

The cost of the correction, on the heads that gain nothing from it (median step, uncorrected → `pca1`):

| | Qwen3-8B | Gemma 4 E4B |
|---|---|---|
| Batch 1 | 2.02 → 2.06 ms (+2%) | 2.13 → 2.23 ms (+5%) |
| Batch 16 | 2.06 → 2.21 ms (+7%) | 2.29 → 2.49 ms (+9%) |

The overhead is 0.04 to 0.20 ms. It has not been profiled; the corrected step computes `α` and `h − αp` in FP64 on the device with several small launches before the first pass, which is the first place to look.

Against the predictions: controls held. Sarvam batch 1: median unchanged (+0.6%), `pca1` p99 below 3.7 ms and mean below 3.45 ms (3.46, 3.41), `mu` p99 in between (4.02): held. Sarvam batch 16 with `pca1` below 3.9 ms and 0.45: held (3.62, 0.41); with `mu` between 0.45 and 0.62: just outside (0.449). Overhead within 3% on Qwen and Gemma at both batch sizes: missed except Qwen at batch 1 (+2%, +5%, +7%, +9%).

Limits: greedy only. The 512 timed states are a random sample of all greedy states, so about half come from prompts `pca1` was fitted on; the K2 addendum found the fit without the state's language to be the same direction on Sarvam (cosine above 0.998), and `mu` uses no states. Head alone, not end-to-end decoding. One run per head; the earlier sections show the graph step's medians move by about 2% between runs.

## K3: is Sarvam-30B special? (14 Nov 2026, exploratory survey)

K2's correction works on Sarvam-30B because one direction holds 74% of its states' second moment. K3 measures that share on eleven more open-weight models, chosen before any was run, to vary family, size, lab, and dense against mixture-of-experts.

Rule `sweep_rule.json` (sha256 `2a7bee49…`, logged before any run). Code `sweep_modal.py`, `sweep_summarize.py`; per-model summaries `logs/sweep_<model>.json`, table `logs/sweep_summary.txt`. New models: this reproduction's 240 prompts, the model's own chat template, greedy, 64 new tokens, final state captured at the head's input. The three existing models use their saved states, steps 0 to 63. Directions are fitted on the even prompt ids and everything else is measured on the odd ones. The certificate is K2's greedy one (dithered INT4, sub-Gaussian bound, INT8 stage, t = 0), about 7,500 test states per model.

Controls: all 14 models loaded and ran. A1, the BF16-product argmax equals the emitted token on 98.7% (Sarvam-30B) to 100% of states, so the captured state and head are the model's. A2, the three existing models reproduce K2's top-1 shares (0.14, 0.22, 0.74 against 0.14, 0.22, 0.74). A3, no bound violation and no token mismatch on the 13 models with a certificate (gpt-oss-20b's hidden size, 2,880, is not a multiple of 128).

| Model | V | d | Second moment in the top 1, 4 directions | Offset share of the top logit | Bound with `pca1` vs none | Bytes, none: median, p99, mean | Bytes, `pca1`: median, p99, mean |
|---|---|---|---|---|---|---|---|
| Sarvam-30B (MoE) | 262,144 | 4,096 | 0.74, 0.84 | +0.77 | 0.60 | 0.258, 0.530, 0.282 | 0.258, 0.260, 0.258 |
| Sarvam-1 | 68,096 | 2,048 | 0.43, 0.86 | +0.17 | 0.71 | 0.293, 0.759, 0.393 | 0.260, 0.760, 0.331 |
| Gemma 4 E2B | 262,144 | 1,536 | 0.33, 0.60 | −2.61 | 0.87 | 0.258, 0.288, 0.259 | 0.259, 0.265, 0.259 |
| gpt-oss-20b (MoE) | 201,088 | 2,880 | 0.30, 0.49 | −0.19 | no certificate | | |
| SmolLM3-3B | 128,256 | 2,048 | 0.28, 0.48 | +0.12 | 0.90 | 0.406, 1.069, 0.476 | 0.337, 0.837, 0.379 |
| Mistral-Nemo | 131,072 | 5,120 | 0.23, 0.47 | +0.10 | 1.00 | 0.258, 0.287, 0.259 | 0.258, 0.280, 0.259 |
| Gemma 4 E4B | 262,144 | 2,560 | 0.22, 0.52 | −0.67 | 0.96 | 0.258, 0.259, 0.258 | 0.259, 0.259, 0.259 |
| Phi-4-mini | 200,064 | 3,072 | 0.16, 0.37 | −0.32 | 0.99 | 0.258, 0.396, 0.278 | 0.259, 0.360, 0.268 |
| Qwen3-8B | 151,936 | 4,096 | 0.14, 0.33 | −0.08 | 0.96 | 0.258, 0.258, 0.258 | 0.258, 0.258, 0.258 |
| Qwen3-4B | 151,936 | 2,560 | 0.13, 0.29 | −0.11 | 0.94 | 0.258, 0.258, 0.258 | 0.259, 0.259, 0.259 |
| Qwen3-14B | 151,936 | 5,120 | 0.13, 0.28 | −0.07 | 0.95 | 0.258, 0.258, 0.258 | 0.258, 0.258, 0.258 |
| Qwen3-30B-A3B (MoE) | 151,936 | 2,048 | 0.12, 0.32 | −0.06 | 0.99 | 0.258, 0.258, 0.258 | 0.259, 0.259, 0.259 |
| Sarvam-M | 131,072 | 5,120 | 0.09, 0.26 | +0.04 | 0.98 | 0.258, 0.258, 0.258 | 0.258, 0.258, 0.258 |
| Granite 3.3 8B | 49,159 | 4,096 | 0.07, 0.19 | −0.03 | 0.97 | 0.258, 0.258, 0.258 | 0.258, 0.259, 0.258 |

- **Sarvam-30B is the outlier.** No other model has a top-1 share above 0.5. The next is Sarvam-1 at 0.43; nine of the eleven new models are at 0.30 or below.
- **It is not the family, the lab or the architecture.** Qwen3 is flat across sizes (0.12 to 0.14), mixture-of-experts included. Sarvam-M, from the same lab on a Mistral base, is at 0.09. The other mixture-of-experts model, gpt-oss-20b, is at 0.30.
- **K2 as run is a fix for one model in this set**, not a general speed-up: on the seven models already at the floor it changes nothing and costs its storage.
- **A finding the survey was not designed for: the uncorrected certificate is not at the floor everywhere.** Seven of the thirteen models have a p99 at or below 0.26. Gemma 4 E2B and Mistral-Nemo are at 0.29. Phi-4-mini (p99 0.396), Sarvam-30B (0.530), Sarvam-1 (0.759) and SmolLM3-3B (1.069, mean 0.476) are not. On SmolLM3-3B even the median state is at 0.41, and the p99 state costs more than reading the BF16 head.
- **Where the certificate is off the floor, one direction helps some models and not others.** SmolLM3-3B's bytes fall from 0.406, 1.069, 0.476 (median, p99, mean) to 0.337, 0.837, 0.379 with a top-1 share of only 0.28; Sarvam-1's median falls from 0.293 to 0.260 and its mean from 0.393 to 0.331, but its p99 does not move. So the top-1 share is not the whole story: how close the model sits to the cliff matters as much.
- The size of the offset tracks the top-1 share across models (Spearman +0.84 with the absolute offset share), but its sign does not (+0.14 signed): Gemma 4 E2B and E4B have large negative offsets.

Against the predictions: "at most one new model above 0.5": held (none). "Qwen3-4B, Qwen3-14B and Gemma 4 E2B below 0.35": held (0.13, 0.13, 0.33). "The two mixture-of-experts models do not both exceed 0.5": held (0.12, 0.30). "Offset share and top-1 share positively rank-correlated, above 0.5": missed as written (+0.14 signed; +0.84 with the absolute value). "`pca1` lowers p99 bytes by more than 0.05 only on models with a top-1 share above 0.5, and on every such model": the second half held (Sarvam-30B, −0.27); the first half missed, since SmolLM3-3B gains 0.23 at a share of 0.28. The falsifier (four or more new models above 0.5) was not triggered.

Limits: 64 new tokens per prompt and one prompt set, so these are not the 128-token figures of the main tables. t = 0 only; no mean-row shift was tried on the new models. Eleven models are not a census: all are small or mid-size open-weight chat models. The four models off the floor were not investigated; why SmolLM3-3B and Sarvam-1 are so wide is open.

### Is the Sarvam result an artifact? Two checks (14 Nov 2026, diagnostic, no rule file)

K3 raises the obvious worry: a method that helps one model of fourteen may be measuring something wrong on that model. Two checks were run after the fact, so they carry no predictions.

**1. Does the correction reduce the real error, not just the bound?** From K2's stored arrays (`results/hdir/<model>`): the largest actual logit error over the vocabulary, per state, with no bound involved. If the split `h · e = α r + (h − αp) · e` is right, the error should shrink by √(1 − ρ) on every model, large or small.

| Median over greedy test states | Largest error, no correction (logits) | With `pca1`, measured | √(1 − ρ) predicts |
|---|---|---|---|
| Sarvam-30B | 5.05 | × 0.609 | × 0.604 |
| Qwen3-8B | 2.33 | × 0.957 | × 0.953 |
| Gemma 4 E4B | 5.05 | × 0.955 | × 0.960 |

The correction does what the algebra says on all three, to within 0.01. It gains little on Qwen and Gemma because ρ is small there, not because it fails.

**2. Is the one-direction share real?** `spectrum_check_modal.py` recomputes it with numpy only and none of the project's analysis code (`logs/spectrum_check_<model>.json`).

| | Sarvam-30B | Qwen3-8B | Gemma 4 E4B |
|---|---|---|---|
| Top-1 share, recomputed (K2's figure) | 0.736 (0.736) | 0.138 (0.138) | 0.222 (0.222) |
| Same, from singular values of a random 4,096-state sample | 0.727 | 0.141 | 0.221 |
| Null: each state's coordinates shuffled | 0.0007 | 0.0006 | 0.0009 |
| Cosine of the top direction with the mean state | 0.995 | 0.963 | 0.936 |
| Share of the states' energy that is the mean state | 0.50 | 0.11 | 0.15 |
| Energy of the direction in its 5 largest coordinates | 0.08 | 0.67 | 0.48 |

- The share reproduces through a separate route, and the shuffled null is at 0.001, so the measure is not high by construction.
- **Sarvam's dominant direction is its mean state**: a fixed vector that is half of every state's energy. On Qwen and Gemma the mean state is 11% and 15%.
- It is a dense vector on Sarvam (8% of it in its five largest coordinates). On Qwen the top direction is mostly a few large coordinates (one coordinate carries 39% of it).
- It is present in every language group on Sarvam when fitted on that group alone (0.55 to 0.91), and at every position after the first token (median share 0.65 to 0.71; 0.38 on the first token). It is weakest on Tamil and Telugu (median share of the state 0.30 and 0.19, against 0.91 on English).

Together with the K2 controls (no bound violation, no token mismatch, the uncorrected path reproducing B1 exactly), the L4 kernels (written separately, same row counts, real time saved), control A1 of K3 (the captured Sarvam state reproduces the emitted token on 98.7% of states) and the lab's own report of a large shared offset on Sarvam, this is a property of the model, not of our measurement. Why the model has a mean state that large is not known.

## E1: the head's share of a decode step, and the step time with the certified head (14 Nov 2026)

Every timing above is the head alone, on an L4. E1 times a whole greedy decode step.

Rule `e2e_rule.json` (sha256 `81f81aed…`, logged before any run). Code `e2e_modal.py`, `e2e_summarize.py`; results `logs/e2e_<model>.json`, table `logs/e2e_summary.txt`. NVIDIA A100-80GB for all three models, since Sarvam-30B does not fit an L4. The stock step is a minimal greedy loop with the key-value cache (one forward per token, then argmax; no `generate()`, no logits processors), which favours the stock path. CUDA events split each step into body (everything up to the head's input), head and argmax. The certified head is the graph-captured step, with the K2 `pca1` correction for Sarvam. 48 prompts × 64 tokens at batch 1 and 240 prompts × 64 tokens at batch 16; the first 4 steps of each prompt are dropped.

Controls: T1, no certified token differs from the FP32 reference and there is no fallback, on any model. T2, Sarvam's BF16 head in the second container is within 1% (batch 1) and 3% (batch 16) of the stock head in the first, so the two containers are comparable. T3, the loop's tokens equal `generate()`'s on all 256 checked tokens for Qwen and for Gemma.

| Model, batch | Timed steps | Body ms (median, p99, mean) | Stock head + argmax ms (median) | Certified head ms (median, p99, mean) | Head's share of the stock step | Certified head / stock head (median) | Certified step / stock step (totals) |
|---|---|---|---|---|---|---|---|
| Qwen3-8B, 1 | 2,832 | 61.98, 75.58, 62.86 | 0.82 | 1.06, 1.17, 1.07 | 1.3% | 1.30 | 1.0038 |
| Qwen3-8B, 16 | 885 | 68.61, 85.64, 69.93 | 0.84 | 0.97, 1.15, 0.98 | 1.2% | 1.15 | 1.0018 |
| Gemma 4 E4B, 1 | 2,832 | 99.77, 133.92, 102.08 | 0.89 | 1.11, 1.22, 1.11 | 0.9% | 1.25 | 1.0022 |
| Gemma 4 E4B, 16 | 885 | 105.48, 123.79, 106.49 | 0.94 | 1.16, 1.22, 1.13 | 0.9% | 1.24 | 1.0018 |
| Sarvam-30B, 1 (two containers) | 2,832 | 28.75, 31.72, 28.76 | 1.24 | 1.26, 1.32, 1.26 | 4.1% | 1.02 | 1.0007 |
| Sarvam-30B, 16 (two containers) | 885 | 75.59, 91.59, 75.77 | 1.39 | 1.38, 1.43, 1.38 | 1.8% | 1.00 | 0.9998 |

- **The head is 1% to 4% of a decode step on this stack.** The body takes 29 to 100 ms per token and the BF16 head about 1 ms. Even a head that cost nothing would save at most that share.
- **On the A100 the certified head is not faster than the stock head.** It takes 1.30 (Qwen), 1.25 (Gemma) and 1.02 (Sarvam) times as long at batch 1. On the L4 it took 0.40.
- **So the end-to-end step is unchanged**: 1.000 to 1.004 times the stock step.
- **Why the L4 result does not carry over.** From the L4 to the A100 the cuBLAS BF16 head gets 6.2 times faster on Qwen (4.96 → 0.80 ms, about 1,560 GB/s) and the certified step only 1.9 times (2.04 → 1.06 ms). The first pass reads 0.26 of the bytes; at the A100's rate that is about 0.21 ms, a fifth of what the step takes. The kernel is bound by its own arithmetic (unpacking nibbles, 16 padded states per row), not by memory, and a faster memory system exposes that. This is the roofline question the review of the first L4 timing left open, now answered against us.
- **Why the head's share is so small here.** The body is bound by the host: 62 ms per token for Qwen3-8B, where reading its 16 GB of weights once at the A100's rate would take about 11 ms. Per-layer Python overhead in the model code inflates the body and not the head. Sarvam's body is the fastest of the three (29 ms) and its head share the largest (4.1%), on a different software stack. The lab's weight-byte arithmetic (head about 8% of bytes for Qwen3-8B, 31% for Sarvam-30B) describes a decode loop bound by memory, which this one is not.

What still stands, and what does not:

- **Stands:** the byte counts, the exactness (no mismatch here either), and the L4 head timings as measured.
- **Does not stand as a general claim:** "the certified head takes 0.40 of the BF16 head's time". That is true on an L4 and false on an A100 with these kernels.
- **Not shown either way:** a time saving in a decode loop that is bound by memory, such as an optimised server or a 4-bit body, where the head's share would be several times larger. Nothing here measures that.

Against the predictions: all the numerical ones missed, in the same direction. Head share at batch 1: predicted 2–8% (Qwen), 5–15% (Gemma), 8–25% (Sarvam); measured 1.3%, 0.9%, 4.1%. End-to-end ratio at batch 1: predicted 0.95–0.99, 0.90–0.97, 0.85–0.95; measured 1.004, 1.002, 1.001. Certified over stock head on the A100: predicted 0.4–0.7; measured 1.02 to 1.30. "The head's share is lower at batch 16": held for Qwen and Sarvam, level for Gemma (0.9% both). The stated fallback ("below 3% on all three") was not met to the letter, since Sarvam is at 4.1%, but its conclusion applies: the time claim needs a memory-bound decode loop and a reworked first-pass kernel before it means anything.

Limits: one GPU type, one run, greedy only, short prompts (context 28 to 438 tokens). Sarvam's figure joins two containers on different A100 variants (SXM4 and PCIe). The kernels' block sizes were tuned on the L4 and not retuned for the A100. Hugging Face transformers is not how a production server decodes.


## N1: where Sarvam-30B's dominant direction comes from (16 Nov 2026, exploratory)

K2 and K3 found that 74% of the second moment of Sarvam-30B's final states lies along one direction, the mean state. A note from the lab asked where it appears: only after the final norm (then the norm's gain vector makes it), or across the layers (then it is in the model).

Rule `norm_rule.json` (hashed before any run on a real model as sha256 `197c582e…`; the published copy, sha256 `4ab0cbce…`, differs only in leaving out a remark from a non-public note; both hashes are in the log). Code `norm_lib.py`, `norm_probe_modal.py`, `norm_summarize.py`, `test_norm_probe.py`; results `logs/normprobe_<model>.json`, table `logs/normprobe_summary.txt`. Sarvam-30B and Sarvam-1, this reproduction's 240 prompts, greedy, up to 64 new tokens, last position only: 15,214 and 15,025 states. Hooks capture the output of every decoder block, the input of the final norm and its output. Two derived sites: the pre-norm state over its root mean square, and the post-norm state with the gain undone (`y / g`). Both norms are RMS norms with a gain and no bias. Directions are fitted on the even prompts and ρ is measured on the odd ones.

Controls, all pass:
- **Hooks:** the norm's output equals the head's input bit for bit on all 960 steps of each model, and the last block's output equals the norm's input.
- **Reproduction:** the post-norm top-1 share is 0.743 on Sarvam-30B (K3: 0.744) and 0.438 on Sarvam-1 (K3: 0.431). On Sarvam-30B the post-norm direction has cosine 0.9999 with K2's saved one.
- **Gain:** undoing the gain reproduces the normalised pre-norm state (median cosine 0.999997).
- **Offline:** `test_norm_probe.py` passes on a small random Llama model and on synthetic states with the direction planted in the model, and made by a gain.

### Sarvam-30B: the last block writes it, and the norm's gain doubles it

| Site | Top-1 share | Mean-state share | ρ, median (odd prompts) | Cosine of top direction with the final one | Median state norm |
|---|---|---|---|---|---|
| Block 17 of 19 | 0.224 | 0.181 | 0.157 | 0.012 | 93 |
| Block 18 | 0.177 | 0.161 | 0.170 | 0.068 | 118 |
| Block 19 = input of the norm | 0.402 | 0.330 | 0.365 | 0.768 | 199 |
| Gain undone (`y / g`) | 0.367 | 0.327 | 0.363 | 0.775 | 64 |
| Output of the norm = input of the head | 0.743 | 0.518 | 0.677 | 1 | 219 |

- **Through block 18 the direction is not there.** At every block from 1 to 18 the top direction and the mean state have a cosine of at most 0.07 with the final ones.
- **The last block creates it.** After block 19 the cosine is 0.77, the top-1 share is 0.40 and the mean-state share 0.33. The median state norm grows from 118 to 199 in that one block.
- **The norm's gain then doubles its share:** 0.367 with the gain undone, 0.743 after it. The mean-state share goes from 0.327 to 0.518 and the median ρ from 0.36 to 0.68. The gain also turns the direction (cosine 0.775 between the two).
- **The gain is broad, not a few large entries.** Mean 1.68, standard deviation 1.23, range 0.045 to 9.19; its five largest entries hold 2.1% of its squared norm. It is larger on the coordinates where the gain-free states share an offset (correlation 0.64 between the gain and the size of the mean). The gain vector itself is not the direction (cosine 0.06).
- **Blocks 4 to 14 show a different effect.** Their top-1 share is high (between 0.42 and 0.83), but that direction sits in five coordinates (0.91 to 0.995 of its energy), has cosine 0 with the final one, and the median ρ is 0.01 to 0.06. So a minority of states have very large entries in a few coordinates there. It fades by block 17.

Verdict by the rule: "the norm makes it", at a ratio of 0.495 against a threshold of 0.5. That is on the line. The fair reading is both: the last block writes a shared vector into the state, and the gain doubles its weight.

### Sarvam-1: a different mechanism

Verdict "both" (0.299 with the gain undone, 0.438 after it, ratio 0.68). But its top direction is not the mean state: it sits in five coordinates (94% of its energy), the mean-state share is 0.13, and the gain raises the top-1 share while the mean-state share falls (0.22 to 0.13). Blocks 1 to 27 are dominated by another five-coordinate direction (top-1 0.50 to 0.97) unrelated to the final one (cosine below 0.02); the last block removes it. So the two Sarvam models do not share a cause, and Sarvam-30B's spread-out mean state remains particular to it among the models surveyed.

### Groups of 128 coordinates (exploratory, prompted by the lab)

A remark in the lab's note about how the bound splits over groups of 128 coordinates prompted this. The note is not public, so only our own measurement is given here (the published `norm_rule.json` leaves the remark out; the copy hashed before the run is kept unpublished). We measured the share of the first-pass variance term by group (the group's energy times the mean over rows of the squared INT4 group scale, t = 0) on the odd-prompt post-norm states. A uniform share would be 0.031.

| Language | Largest group's share, median | After removing our direction | Group 26 is the largest on |
|---|---|---|---|
| Bengali | 0.068 | 0.114 | |
| Code | 0.060 | 0.117 | |
| English | 0.056 | 0.106 | |
| Hindi | 0.058 | 0.099 | |
| Tamil | 0.333 | 0.583 | 74% of states (79% after) |
| Telugu | 0.428 | 0.593 | 53% of states (59% after) |

- **Tamil and Telugu have one heavy group.** In those two languages one group, number 26 (coordinates 3,328 to 3,455), carries a third and 0.43 of the variance term for the median state. In the other four languages no group carries more than 0.07.
- **Our direction is spread over all groups.** Its largest group holds 5.3% of its energy and its five largest coordinates 7.6%. That is why no small set of groups carries the bound in general.
- **The group-26 part is separate from our direction.** Removing our direction leaves it and raises its share to 0.58 and 0.59. This fits K2, where directions two to four were language-specific, and the spectrum check, where Tamil and Telugu had the smallest ρ (0.30 and 0.19).
- On Sarvam-1 every language has a heavy group (0.28 to 0.83), which follows from its five-coordinate direction.

### What it means for the method

- **K2's correction stands.** It acts on the head's input, whatever made the direction.
- **The direction cannot be read off the checkpoint's gain alone** (cosine 0.06). It needs states, or an analysis of the last block.
- **Untested alternative:** fold the gain into the head rows (`W diag(g)`) and bound against the gain-free state, whose median ρ is 0.36, not 0.68. This changes the group scales, so whether the bound narrows cannot be said without running it.
- **Not known:** what in block 19 writes the vector (attention, the experts, or a bias), and whether other models' last blocks and gains do the same on a smaller scale. K3's other models were not probed.

Against the predictions:
- **Sarvam-30B verdict: missed.** Predicted "in the model" with a gain-free share of at least 0.60; measured 0.367.
- **Sarvam-30B layers: missed.** Predicted a build-up across layers with the last block changing the share by less than 0.15; it is a last-block effect (+0.225, and the direction itself only appears there).
- **Gain not concentrated: held** (2.1%, predicted below 5%).
- **Sarvam-1: missed.** Predicted a ratio of at least 0.8; measured 0.68.
- **Direction is spread: held** (largest group 5.3%, predicted below 10%).
- **Groups: held.** Tamil 0.333 and Telugu 0.428 (predicted at least 0.30), the others at most 0.068 (predicted at most 0.15), and the concentration stays after removing our direction.

Deviations and our own mistakes:
- **The rule's "first block at half the pre-norm share" is uninformative.** Earlier blocks have a high top-1 share along a different direction, so it returns block 1 on both models. The cosine with the final direction is what locates the appearance, and the table uses it.
- **Sarvam-1 had no smoke run.** Its L4 container did not start in 20 minutes, so the smoke was stopped after Sarvam-30B's half passed. The function was then allowed to take the first free of L4, A10G and A100-40GB, and the full run used an A10. Sarvam-30B's smoke (12 prompts) gave the same picture as the full run (0.349 and 0.713).

Limits: states are regenerated, not the saved exp01 states. Last position only, greedy, 64 tokens, one run. The group split uses mean-row scales at t = 0, not each row's own bound. Nothing here says which part of block 19 writes the vector.

## M1: one group in INT8 for every row, with and without the K2 correction (17 Nov 2026)

N1 found that in Tamil and Telugu one group of 128 coordinates (group 26, coordinates 3,328 to 3,455) carries a third to a half of the first-pass variance term, separately from the direction K2 removes. A note from the lab (not public) prompted the question: does storing that group in INT8 for every row help, alone or together with K2?

Rule `mix_rule.json` (sha256 `35576903…`, logged before any run on real states). Code `mix_lib.py`, `mix_modal.py`, `mix_summarize.py`, `test_mix.py`; results `logs/mix_summary.json`, tables `logs/mix_summary.txt`. Sarvam-30B, all 29,826 greedy and 29,507 sampled states, the K2 addendum's certificate (sub-Gaussian bound, INT8 stage, shift 0.5).

The mixed head is the dithered INT4 copy with group 26 stored as dithered INT8 (step = largest |w| / 126.5, same dither pattern, same bound). It costs 64 more bytes per row: 0.0078 of the BF16 head on every state. Arms: `none`; `k2` (K2's correction, direction fitted without the state's language); `int8g` (mixed head alone); `both`; and, exploratory, `k2_g26dir` (K2 plus a second cancelled direction inside group 26, also fitted without the language, 4 bytes per row).

Controls, all pass: `none` and `k2` equal the K2 addendum's stored values on all 59,333 states (0 differences); no bound violation and no token mismatch in any arm; on the saved even-prompt states group 26 is the largest group for 77% of Tamil and 46% of Telugu states and for under 0.1% elsewhere; `test_mix.py` passes (with no INT8 group the quantizer is `cert_core`'s bit for bit, and a planted fault is caught).

### Greedy and Gumbel: the combination fails the criterion

The criterion: `both` must beat `k2` and `int8g` in Tamil and Telugu (p99 and mean) without hurting the other four languages. It fails on greedy and on both Gumbel temperatures.

Greedy, bytes as a fraction of the BF16 head (median, p99):

| Language | `none` | `k2` | `int8g` | `both` |
|---|---|---|---|---|
| Bengali | 0.258, 0.467 | 0.2583, 0.2589 | 0.266, 0.468 | 0.2661, 0.2666 |
| Code | 0.258, 0.526 | 0.2583, 0.2584 | 0.266, 0.530 | 0.2661, 0.2662 |
| English | 0.265, 0.521 | 0.2583, 0.2591 | 0.273, 0.525 | 0.2661, 0.2669 |
| Hindi | 0.258, 0.463 | 0.2583, 0.2589 | 0.266, 0.465 | 0.2661, 0.2667 |
| Tamil | 0.258, 0.503 | 0.2583, 0.2603 | 0.266, 0.507 | 0.2661, 0.2665 |
| Telugu | 0.258, 0.483 | 0.2583, 0.2604 | 0.266, 0.486 | 0.2661, 0.2669 |

- **`both` is worse than `k2` on every state.** The mean difference is +0.0076 to +0.0078 in every language and rule, and no state reads fewer bytes with `both`. With Gumbel the p99 in Tamil and Telugu goes from 0.261–0.262 to 0.267–0.268.
- **The reason is that K2 alone is already at the floor.** Its p99 in Tamil and Telugu is 0.260, against a floor of 0.2583. There is at most 0.002 to 0.004 left to gain, and the INT8 group adds 0.0078 to every state.
- **The group does what it should to the bound.** With K2, the median bound in Tamil falls from 0.86 of the uncorrected one to 0.50, and in Telugu from 0.91 to 0.55. The INT8-stage shortlist at the p99 falls from 1,033 rows to 163 (Tamil) and from 1,065 to 368 (Telugu). But 870 fewer INT8 rows save 3.6 MB at the p99, and the group costs 16.8 MB on every state.
- **The INT8 group alone is worse than no fix** on greedy and Gumbel in every language (mean +0.006 to +0.007). The long shortlists come from the mean direction, which it does not touch.

### Top-50: it helps Tamil and Telugu and costs the others

Top-50, mean bytes, and the difference `both − k2` with a 95% interval from resampling prompts:

| Language | `k2` | `both` | `both − k2` |
|---|---|---|---|
| Bengali | 0.2732 | 0.2798 | +0.0066 [+0.0065, +0.0067] |
| Code | 0.2724 | 0.2779 | +0.0054 [+0.0052, +0.0056] |
| English | 0.2735 | 0.2791 | +0.0056 [+0.0054, +0.0058] |
| Hindi | 0.2679 | 0.2749 | +0.0070 [+0.0069, +0.0071] |
| Tamil | 0.2909 | 0.2727 | −0.0181 [−0.0210, −0.0149] |
| Telugu | 0.2876 | 0.2773 | −0.0103 [−0.0128, −0.0077] |
| All | 0.2777 | 0.2769 | −0.0007 [−0.0022, +0.0008] |

Over all states the mean is unchanged and the p99 goes from 0.374 to 0.357. Top-50 still misses its gate with every arm (BF16 rows at the p99: 10,144 with `k2`, 7,830 with `both`, against a limit of 2,621). On min-p, `both` is worse than `k2` in every language (+0.0065 to +0.0078).

### A second direction inside the group does nothing

`k2_g26dir` changes Tamil's top-50 mean by −0.0004 and Telugu's by −0.0008, and costs 0.0005 elsewhere. The direction does not transfer: fitted with Tamil left out and with Telugu left out, the two directions have a cosine of 0.08, although each holds 0.92 of the group's remaining energy in its fit. So Tamil and Telugu use the same 128 coordinates along different directions. This is K2's "directions two to four are language-specific" again, now inside one group.

### Estimate, not a run: reading the group only for shortlisted rows

The mixed head differs from the plain one only in group 26, with the same dither. So reading a row's INT8 codes for that group (130 bytes) after the first pass gives exactly the mixed head's interval for that row. A staged variant would pay 130 bytes per shortlisted row, not 64 bytes for every row. From the per-state results (`mix_staged_estimate.py`, `logs/mix_staged_estimate.txt`; it assumes the later stages read what `both` reads):

| Rule | `k2`, mean | Staged estimate, mean | Tamil: `k2` → staged | Telugu: `k2` → staged |
|---|---|---|---|---|
| Greedy | 0.2584 | 0.2584 | 0.2585 → 0.2583 | 0.2585 → 0.2584 |
| Min-p 0.05 | 0.2593 | 0.2589 | 0.2599 → 0.2587 | 0.2599 → 0.2589 |
| Top-50 | 0.2777 | 0.2696 | 0.2909 → 0.2658 | 0.2876 → 0.2703 |

It is estimated to cost nothing where K2 is at the floor and to keep most of the top-50 gain in Tamil and Telugu without charging the other languages. This has not been run as a certificate.

Against the predictions:
- **Criterion fails on greedy and both Gumbel rules: held.** Predicted a median near 0.266 and a p99 0.004 to 0.008 above `k2` in every language; measured 0.2661 and +0.0053 to +0.0078.
- **`int8g` alone, partly missed.** Held: bound factor 0.83 in Tamil, under 0.01 change elsewhere, English and code p99 above 0.40, nowhere near `k2`. Missed: Telugu's bound factor is 0.96 (predicted 0.75 to 0.85), and the Tamil and Telugu p99 did not fall to 0.30–0.48 but stayed at 0.51 and 0.49, no better than no fix.
- **Top-50: held.** Tamil and Telugu better by 0.018 and 0.010 (predicted at least 0.005), code and Hindi worse by 0.005 and 0.007, overall mean within 0.003, gate still missed.
- **`k2_g26dir`: the "does nothing" branch** of an even-odds prediction.

Limits: one model, one group, one shift. Group 26 was first found on regenerated odd-prompt states (N1) and is confirmed here on the saved even-prompt states, but all states are then measured. The staged figures are an estimate from stored shortlist sizes. Bytes only; no timing.

## M2: the heavy group read only for the rows the rule still needs (17 Nov 2026)

M1's estimate said that reading group 26 in INT8 only for shortlisted rows would keep the top-50 gain in Tamil and Telugu at no cost elsewhere. M2 runs that as a certificate.

Rule `stage_rule.json` (sha256 `7e98e2b8…`, logged before any run on real states). Code `stage_lib.py`, `stage_modal.py`, `stage_summarize.py`, `test_stage.py`; results `logs/stage_summary.json`, tables `logs/stage_summary.txt`. Sarvam-30B, all 59,333 states, five rules.

Stages: (1) the plain dithered INT4 head with K2's correction, every row, exactly M1's arm `k2`; (G) for the rows the rule still needs, the row's group 26 in INT8, which gives that row the mixed head's interval (the two heads differ only in that group and share the dither); (2) INT8 rows; (3) BF16 rows. Each row read in stage G is charged 204 bytes: 128 codes, 2 for their step, 4 for the row's K2 number for the mixed copy, 4 for that copy's row norm, and 66 for re-reading the row's INT4 codes and step. A row read in stage G carries two dither bounds, so its failure probability is at most 2δ.

Controls, all pass: the arm `k2` equals M1's stored values on all 59,333 states; no reference logit outside the first-pass or the stage-G interval; no token mismatch on any rule; `test_stage.py` passes (a stage G fed the first-pass intervals changes nothing, and a planted fault is caught).

**The criterion passes, both parts.**

(a) Greedy and both Gumbel rules: in every language the median and p99 are within 0.0005 of `k2`, and in fact at or below it.

| Rule | Tamil p99: `k2` → staged | Telugu p99: `k2` → staged | All states, mean: `k2` → staged |
|---|---|---|---|
| Greedy | 0.2603 → 0.2588 | 0.2604 → 0.2591 | 0.2584 → 0.2584 |
| Gumbel T = 0.6 | 0.2613 → 0.2592 | 0.2622 → 0.2598 | 0.2585 → 0.2584 |
| Gumbel T = 1 | 0.2614 → 0.2591 | 0.2620 → 0.2598 | 0.2585 → 0.2584 |

(b) Top-50, mean bytes and the difference with a 95% interval from resampling prompts:

| Language | `k2` | `both` (M1) | staged | staged − `k2` |
|---|---|---|---|---|
| Bengali | 0.2732 | 0.2798 | 0.2724 | −0.0008 [−0.0010, −0.0007] |
| Code | 0.2724 | 0.2779 | 0.2703 | −0.0022 [−0.0024, −0.0019] |
| English | 0.2735 | 0.2791 | 0.2716 | −0.0020 [−0.0022, −0.0018] |
| Hindi | 0.2679 | 0.2749 | 0.2673 | −0.0006 [−0.0007, −0.0005] |
| Tamil | 0.2909 | 0.2727 | 0.2664 | −0.0245 [−0.0276, −0.0214] |
| Telugu | 0.2876 | 0.2773 | 0.2707 | −0.0169 [−0.0190, −0.0142] |
| All | 0.2777 | 0.2769 | 0.2697 | −0.0079 [−0.0093, −0.0066] |

- **Top-50 improves in every language,** most in Tamil and Telugu. Over all states: median 0.270 → 0.265, p99 0.374 → 0.352, mean 0.278 → 0.270.
- **Top-50 still misses its gate.** BF16 rows at the p99 fall from 10,145 to 7,813 against a limit of 2,621.
- **Min-p improves in the tail:** the p99 goes from 0.280 to 0.264 in Tamil and from 0.283 to 0.269 in Telugu; over all states from 0.276 to 0.269.
- **It beats M1's every-row form everywhere,** because it pays for the group only where the rule asks for it. Stage G reads a median of 4 rows per greedy state and 3,265 per top-50 state, out of 262,144.
- **Not every state gains.** On greedy, 59% of states read a few bytes more (at most 0.0001 of the head), because stage G is paid for rows it does not help. The means and the tails are lower.
- With 138 bytes a row (a kernel that keeps each row's partial sum for the group) the figures change by at most 0.0005.

Against the predictions: all held. Criterion passes; greedy and Gumbel within 0.0002 of `k2` in the mean; top-50 overall mean 0.2697 (predicted 0.2695 to 0.2715), Tamil 0.2664 (0.265 to 0.269), Telugu 0.2707 (0.270 to 0.273); min-p p99 0.264 and 0.269 (0.263 to 0.272); gate still missed; M1's estimate was within 0.0006 of the measured mean on every rule and language.

Limits: one model, one group, chosen from this model's states. Bytes only: stage G is one more data-dependent gather, and nothing here measures what it costs in time. The group helps two languages; a model or language with no such group gains nothing from it.

## T1: top-50 through its gate with a dithered INT8 stage; a leaner top-k rule; top-p (17 Nov 2026)

Top-50 missed its gate on Gemma 4 and Sarvam-30B because too many rows needed the BF16 read at the p99. Reading the cascade showed why: the INT8 stage has always used a worst-case bound on an undithered copy. T1 replaces it with a dithered INT8 copy and the same sub-Gaussian bound as the first pass, tries a leaner top-k rule, and tests top-p for the first time.

Rule `t1_rule.json` (sha256 `79244c93…`, logged before any run on real states). Code `rules2_lib.py`, `pipe_lib.py`, `t1_modal.py`, `run3_summarize.py`, `test_rules2.py`; results `logs/t1_summary.json`, tables `logs/t1_summary.txt`. Qwen3-8B, Gemma 4 E4B and Sarvam-30B, all 176,359 saved states.

- **`i8`**: the INT8-row copy with its worst-case bound, as in every earlier run.
- **`d8`**: a dithered INT8 copy with a step per group of 128, the first pass's dither pattern and the sub-Gaussian bound; on Sarvam-30B with K2's correction applied to it. It costs 40 to 68 bytes a row more than `i8` (the steps per group, and on Sarvam-30B the row's K2 number). A row read there carries one more dither bound: at most 2δ, or 3δ after M2's group stage.

Controls: no reference logit outside any interval and no token mismatch, for every arm and rule including the two new rules. The `base` arm reproduces the stored B1 and M2 rows on every state. `test_rules2.py` passes.

**One control failed on the first run and passed on the rerun.** The first run used batches of 64 states, where B1 and M2 used 128. Two stored values of about a million differed on Sarvam-30B, because the reference product rounds differently with another batch shape. The rule says any difference invalidates the run, so it was rerun at 128: zero differences, and every mean in the tables is the same to five decimals. The first run is kept as `logs/t1_summary_batch64.json`.

### The dithered INT8 stage

Its bound is 0.14 to 0.16 of the worst-case one. Top-50:

| Model | Arm | Bytes (median, p99, mean) | BF16 rows at the p99 | Gate |
|---|---|---|---|---|
| Qwen3-8B | `i8` | 0.2578, 0.2736, 0.2584 | 10 | passes |
| Qwen3-8B | `d8` | 0.2578, 0.2734, 0.2584 | 2 | passes |
| Gemma 4 E4B | `i8` | 0.2578, 0.3208, 0.2717 | 4,833 | misses |
| Gemma 4 E4B | `d8` | 0.2578, 0.3151, 0.2714 | 2 | **passes** |
| Sarvam-30B, K2 | `i8` | 0.2698, 0.3738, 0.2777 | 10,145 | misses |
| Sarvam-30B, K2 | `d8` | 0.2646, 0.3655, 0.2733 | 48 | **passes** |
| Sarvam-30B, K2 + group stage (M2) | `i8` | 0.2649, 0.3522, 0.2697 | 7,813 | misses |
| Sarvam-30B, K2 + group stage | `d8` | 0.2621, 0.3436, 0.2662 | 48 | **passes** |

- **Top-50 now passes its gate on all three models.**
- **The BF16 head is almost never read, on any rule.** At the p99, greedy needs 2 to 3 BF16 rows (9 to 48 before), Gumbel 2 to 4, min-p 3 to 17 (20 to 537 before).
- **Bytes change little.** The rows move from the BF16 stage to the INT8 stage, they do not disappear: Gemma 4's top-50 mean goes from 0.2717 to 0.2714. Top-50 remains the most expensive rule, with a p99 of 0.32 to 0.34 on Gemma 4 and Sarvam-30B.
- On greedy, Gumbel and min-p the mean changes by less than 0.001.

### The leaner top-k rule does not work

It reads only the race candidates and the rows overlapping their intervals. It reads fewer rows in INT8, and then far more in BF16, because rows it skipped keep their wide first-pass intervals. On Gemma 4 the top-50 mean rises from 0.272 to 0.317 and the p99 to 0.78; on Sarvam-30B it reads the whole vocabulary at the p99. Not adopted.

### Top-p 0.9: exact, but cheap only on Qwen3-8B

No token differs from the float64 reference decision on any state. Bytes with `d8`:

| Model | Bytes (median, p99, mean) | Gate |
|---|---|---|
| Qwen3-8B | 0.2578, 0.3031, 0.2598 | passes |
| Gemma 4 E4B | 0.2579, 0.7653, 0.3266 | misses |
| Sarvam-30B, K2 + group stage | 0.2834, 0.8007, 0.4250 | misses |

On Gemma 4 and Sarvam-30B the certificate reads the whole vocabulary in INT8 on more than 1% of states. Pinning the nucleus boundary needs the probability mass above each candidate, and with a flat distribution that takes most of the rows. Top-p is a stated limit in this form.

Against the predictions:
- **Bound ratio 0.08 to 0.20: held** (0.14 to 0.16).
- **BF16 rows fall at least fourfold: held** (4,833 to 2; 10,145 to 48).
- **Top-50 passes on Gemma 4 and Sarvam-30B: held.**
- **Top-50 mean below 0.262 on Gemma 4 and below 0.266 on Sarvam-30B: missed** (0.2714 and 0.2662). The bytes sit in the INT8 stage, which `d8` does not shrink.
- **Other rules change by less than 0.001: held.**
- **The leaner rule does not help: held.** This prediction was written after the synthetic test showed it; before that test we expected the opposite.
- **Top-p: partly missed.** No mismatch and Qwen3-8B passes, as predicted. Gemma 4 was predicted to pass and does not; Sarvam-30B's p99 was predicted below 0.35 and is 0.80.

Limits: the gate counts BF16 rows, so moving reads into a dithered INT8 stage passes it without saving many bytes; the byte columns are the honest measure. The exactness of every stage is now probabilistic (2δ to 3δ per row read). One dither draw here; R1 covers other draws for Sarvam-30B.

## R1: a second prompt set and four dither draws (17 Nov 2026)

Rule `r1_rule.json` (sha256 `4e9e73ff…`, logged before any certificate ran on the new states or seeds). Code `gen2_modal.py`, `r1_modal.py`; results `logs/r1_summary.json`, tables `logs/r1_summary.txt`. Sarvam-30B.

**Second prompt set** (`prompts2.jsonl`, private like the first): 240 prompts that share none with the first. English: MT-Bench questions 6 to 10 of each category. Code: HumanEval tasks 40 to 79. Hindi, Bengali, Tamil, Telugu: a passage and a question in the language from XQuAD-IN, where the first set was translation from English. 19,044 greedy and 24,168 sampled states. **Nothing is refitted on it:** the direction is K2's, fitted on the first set's even prompts, the group is 26, the shift is 0.5.

**Dither seeds:** 1234 (used so far), 1, 2, 3, on both prompt sets.

Controls: 359,039 state evaluations over the eight combinations. No reference logit outside an interval, no token mismatch. Seed 1234 on the first set reproduces M2's stored rows.

Second prompt set, seed 1234, bytes (median, p99, mean):

| Rule | No correction | K2 | K2 + group stage + `d8` |
|---|---|---|---|
| Greedy | 0.2589, 0.5220, 0.2881 | 0.2583, 0.2601, 0.2584 | 0.2583, 0.2593, 0.2584 |
| Gumbel T = 1 | 0.2596, 0.4962, 0.2862 | 0.2583, 0.2613, 0.2586 | 0.2583, 0.2603, 0.2584 |
| Min-p 0.05 | 0.2660, 0.5667, 0.3143 | 0.2584, 0.2830, 0.2599 | 0.2583, 0.2738, 0.2591 |
| Top-50 | 0.4237, 0.6848, 0.4359 | 0.2702, 0.3987, 0.2782 | 0.2622, 0.3767, 0.2675 (passes) |

- **The problem and the fix both carry over.** With no correction the greedy p99 is 0.52; with the direction from the first prompt set it is 0.260. The median bound is 0.55 of the uncorrected one (0.58 on the first set, all states).
- **By language on the new set,** the greedy p99 with K2 is 0.258 to 0.260 in five languages and 0.264 in Telugu.
- **The dither draw does not matter.** Across the four seeds the greedy mean with K2 is 0.2584 on both sets, and the p99 is 0.2596 to 0.2597 on the first set and 0.2601 to 0.2602 on the second. Every gate status is the same for every seed.

Against the predictions: all held. Uncorrected greedy p99 above 0.40 (0.52); with K2 at most 0.265 (0.260) and the mean at most 0.2595 (0.2584); greedy, Gumbel and min-p pass; top-50 passes only with `d8`; bound ratio between 0.55 and 0.80 (0.551, at the edge); seeds within the stated spread.

Limits: still one model for this check, and the second set comes from the same benchmarks' families (MT-Bench, HumanEval, IndicGenBench). It changes the questions and the Indian-language task, not the genre of every prompt.

## G1: the parts on the other off-floor models (17 Nov 2026)

K3 found three more models off the byte floor and ran them untuned. G1 applies the method's parts to them.

Rule `g1_rule.json` (sha256 `67d8c11f…`, logged before any certificate ran on these states). Code `gen2_modal.py`, `g1_modal.py`; results `logs/g1_<model>.json`, tables `logs/g1_summary.txt`. Sarvam-1, SmolLM3-3B, Phi-4-mini; the 240 prompts, greedy and T = 1, 128 new tokens.

Everything is chosen on the even prompts (by mean greedy bytes) and reported on the odd ones: the shift from 0, 0.5, 1, 1.5; the number of cancelled directions from 0, 1, 4, 16; one or two staged groups, or none.

| Model | Chosen: shift, directions, groups | Greedy, plain (median, p99, mean) | Greedy, full method | Gate |
|---|---|---|---|---|
| Sarvam-1 | 0.5, 4, one group | 0.284, 0.759, 0.386 | 0.2618, 0.2676, 0.2621 | passes |
| SmolLM3-3B | 1.0, 4, two groups | 0.409, 1.086, 0.482 | 0.2639, 0.3423, 0.2699 | passes |
| Phi-4-mini | 1.0, 4, two groups | 0.258, 0.385, 0.275 | 0.2605, 0.2827, 0.2619 | passes |

The floor with four directions is 0.2617 on the two models with d = 2,048 and 0.2604 on Phi-4-mini.

- **Greedy passes its gate on all three with the full method.** Two meet the stricter bar set in the rule (p99 at most 0.30): Sarvam-1 and Phi-4-mini. SmolLM3-3B's p99 is 0.342, down from 1.086.
- **Each part matters on a different model.**
  - Sarvam-1: the directions. Four of them carry 85% of the state energy; the bound falls to 0.36 of the plain one and the mean from 0.380 to 0.262.
  - SmolLM3-3B: the shift. It alone takes the mean from 0.482 to 0.306; directions, groups and `d8` take it to 0.270.
  - Phi-4-mini: all of them a little; `d8` cuts the p99 from 0.295 to 0.283.
- **Gumbel passes on all three** with the full method. **Min-p** passes on SmolLM3-3B only. **Top-50** passes on none: Sarvam-1 0.278, 0.819, 0.300; Phi-4-mini 0.327, 0.526, 0.337; SmolLM3-3B 0.423, 0.656, 0.424.

Controls: no reference logit outside an interval and no token mismatch on any of the 171,511 states, in any arm or rule.

Against the predictions: we expected at most one of the three to be fixed, and two are, with the third passing its gate. Sarvam-1: K = 4 as predicted, but its p99 was predicted to stay above 0.30 and is 0.268. SmolLM3-3B: predicted a median above 0.28; it is 0.264. Phi-4-mini: predicted a p99 below 0.32; it is 0.283.

**Addendum, Qwen3-0.6B** (`g1_addendum_small.json`, hashed before its run; `logs/g1_qwen3_0.6b.json`). A small model whose output layer is 21% of its weights, d = 1,024. It sits at the floor untuned: greedy 0.2578, 0.2585, 0.2579 with no shift and no correction, and every rule passes its gate, top-50 included (0.2590, 0.2797, 0.2609 with the full arm). No violation, no mismatch on 57,422 states. We predicted an untuned p99 above 0.30; that missed.

Limits: the choices are made per model on half of the same prompt set, so they are tuned to this kind of text; R1's check on new prompts was done for Sarvam-30B only. The plain arm is K3's setting but on 128-token states, so its numbers differ a little from K3's.

## The head's share of the weights (17 Nov 2026, from public model metadata)

`logs/head_share.json`. The output layer as a share of all parameters: 2.5% to 8.4% for the 8B-and-larger models surveyed, 16% for Phi-4-mini, about 21% for Qwen3-0.6B and SmolLM2-135M. Where input and output tables are untied, the input table is not read per token, so the head's share of the weights read per token is a little higher (Qwen3-8B: 8.2%). For a decode step limited by memory, that share is an upper limit on what a cheaper head can save: with the head at 0.26 of its bytes, about 6% of a step on Qwen3-8B and about 15% on Qwen3-0.6B. Nothing here measures such a step.

## E2: the head's share of a decode step on a small model, and the step with the certified head by substitution (17 Nov 2026)

E1 found the head to be 1% to 4% of a decode step on 8B-and-larger models in an eager Hugging Face loop. E2 takes the most favourable case we can measure: Qwen3-0.6B, whose output layer is 21% of its weights and which sits at the byte floor untuned, with Hugging Face's static cache, which removes most of the host overhead.

Rule `e2_rule.json` (sha256 `cd115191…`, logged before any E2 code ran). Code `e2_modal.py`; results `logs/e2_<gpu>.json`. Batch 1, greedy, 24 prompts. The step is the time per new token of `generate` ((time for 72 tokens − time for 8) / 64). The heads are timed alone on 512 saved states, graph-captured: the stock BF16 product with argmax, and the unified kernel (`kernel/fused2.py`, shift 0.5). **The certified step is an estimate by substitution** (step − stock head + certified head); the certified head was not run inside `generate`.

| GPU | Step, eager | Step, static cache | Stock head | Certified head | Head's share of the static-cache step | Estimated certified step / stock step |
|---|---|---|---|---|---|---|
| A100-80GB | 37.3 ms | 4.51 ms | 0.229 ms | 0.219 ms | 5.1% | 0.998 |
| A10G | 25.8 ms | 5.65 ms | 0.697 ms | 0.289 ms | 12.3% | 0.928 |

- **On the A10G the estimate is a 7% faster step.** The head is 12% of a static-cache step there, and the certified head takes 0.41 of the stock head's time.
- **On the A100 there is nothing to gain on this model.** The stock head takes 0.23 ms for 311 MB, and the certified head 0.22 ms: at this size both are close to the fixed cost of a few kernel launches.
- **The static cache matters more than anything we do.** It takes the step from 37 to 4.5 ms on the A100. In the eager loop the head is 0.6% (A100) and 2.7% (A10G) of a step.
- No certified token differs from the FP32 reference; the stock BF16 head differs on 3 of 512 states, through ties.
- Compiling `model.forward` explicitly with `torch.compile(mode="reduce-overhead")` on top of the static cache failed on both GPUs with a CUDA-graph error inside `generate`; recorded in the results, not pursued.

Against the predictions: tokens exact, held. Head share in the eager loop under 3%, held. Head share in the fastest configuration, held on both GPUs (5.1% and 12.3%). Head ratio: held on the A10G (0.41), missed on the A100 (0.96 against a predicted 0.60 to 0.85). Estimated step ratio: both just outside the predicted ranges, the A100 on the side of less gain (0.998) and the A10G on the side of more (0.928).

Limits: an estimate, not an integration; one small model; batch 1; `generate`'s own per-token overhead is part of the step. The A10G sits between the L4 and the A100 in memory bandwidth, and that is where the head's time is large enough to matter.

## E3: the certified head inside the decode loop, measured (17 Nov 2026)

E2's figure was an estimate by substitution. E3 puts the certified head in the loop and measures the step.

Rule `e3_rule.json` (sha256 `d4c85b62…`, logged before any E3 code ran). Code `e3_modal.py`; results `logs/e3_<gpu>.json`. Qwen3-0.6B, batch 1, greedy, E2's 24 prompts, Hugging Face `generate` with the static cache.

- **Stock:** the model as shipped. The BF16 output layer runs inside the forward pass and `generate` takes the argmax.
- **Certified:** the output layer is replaced by an identity, so the forward pass returns the final hidden state. A logits processor runs the unified kernel (`kernel/fused2.py`, shift 0.5, graph-captured) on that state and hands `generate` a score vector whose argmax is the certified token. All three copies are on the GPU.

Both generate exactly the requested number of tokens (end-of-sequence stopping off, repetition penalty 1). The step is (time for 72 new tokens − time for 8) / 64 per prompt, in the order stock, certified, stock again.

| GPU | Stock | Certified | Stock again | Certified / stock (first) | Certified / mean of the two stock runs |
|---|---|---|---|---|---|
| A10G | 5.50 ms | 5.18 ms | 5.67 ms | 0.942 | 0.928 |
| A100-80GB | 4.11 ms | 4.14 ms | 4.18 ms | 1.008 | 1.000 |

- **On the A10G the decode step is 6% to 7% faster with the certified head.** This is a measured end-to-end gain, with the token coming from the certified head on every step. Per prompt the ratio has a median of 0.939; the second largest of the 24 is 0.984 (the largest was not saved).
- **On the A100 there is no gain,** as E2 estimated.
- **Every certified token equals the reference.** In a separate untimed pass the loop recorded all 1,728 states and tokens on each GPU: each token is the argmax of the FP32 product of its state with the BF16 head.
- **The stock model's own text differs.** On 12 of the 24 prompts the stock continuation departs from the certified one within 72 tokens, because the stock BF16 head breaks near-ties differently from the FP32 reference.

Against the predictions: tokens exact, held. A10G ratio 0.93 to 0.97, held (0.942); E2's estimate of 0.928 is what the mean of the two stock runs gives. A100 ratio 0.99 to 1.03, held (1.008). Stock drift within 2%: held on the A100 (1.6%), missed on the A10G (3.0%), which is why both ratios are shown.

Limits: one small model whose output layer is 21% of its weights, batch 1, greedy, one run per GPU, 24 prompts. `generate`'s per-token overhead is in both variants. The stock measurement drifted by 3% on the A10G over the run, which is half the size of the effect. The all-GPU layout takes 1.77 times the stock head's memory (see the memory tiers in `kernel/README.md`). Not run on an L4.
