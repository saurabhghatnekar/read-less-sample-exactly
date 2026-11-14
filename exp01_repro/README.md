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
- Run-to-run variation (added in review, 14 Nov). Two later runs of the same bench (`logs/l4_graph_timing_all.log`, `logs/l4_graph_timing_sarvam_rerun.log`) gave different certified-step medians, while the BF16 head stayed within 1% and the kernel within 3%:

  | Certified / BF16, median | this table | later runs |
  |---|---|---|
  | Qwen3-8B, 1 | 0.81 | 0.86 |
  | Gemma 4, 1 | 0.77 | 0.66 |
  | Sarvam-30B, 1 | 0.64 | 0.56, 0.66 |
  | Qwen3-8B, 16 | 0.85 | 0.91 |
  | Gemma 4, 16 | 0.92 | 0.82 |
  | Sarvam-30B, 16 | 1.04 | 0.99, 1.04 |

  The host-bound part of the step moves by up to 15% between Modal containers, so the certified ratios are good to about ±0.1. Of the median verdicts above, only "Qwen misses at batch 1" and "Sarvam misses at batch 16" are the same in every run. Gemma at batch 1 (0.66), Sarvam at batch 1 in one run (0.66) and both Qwen and Gemma at batch 16 land on the other side of the predicted band in the later runs. Sarvam's batch-1 p99 stays under BF16 in all three (0.82 to 0.90).

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
