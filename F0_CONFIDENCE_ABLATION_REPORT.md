# Frozen-F0 confidence ablation: completed results

The [prespecified protocol](F0_CONFIDENCE_ABLATION_PROTOCOL.md) held the answer policy fixed at F0, which had learned only JeV fold0. Confidence training, development and sealed JeV testing all used F0's actions on its unseen fold1; Holmes used F0's actions on all 1,837 questions. The fold1 video split was 20,063 train / 2,000 development / 2,000 sealed test. There was **no video-filename overlap** among fold0, these three sets and Holmes. All four confidence LoRAs used the same 20,063 frozen actions and squared error on mean confidence; only initialization and cosine learning rate differed. Six checkpoints per variant were evaluated on development data, and each variant's **lowest development Brier** checkpoint was selected before the sealed tests were scored.

| Initialization / LR | Selected step | Dev Brier ↓ | JeV sealed Brier ↓ | JeV AUC ↑ | JeV ECE ↓ | Holmes Brier ↓ | Holmes AUC ↑ | Holmes ECE ↓ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Original Qwen, 2e-6 | 20,000 | 0.19156 | 0.19339 | 0.72874 | 0.04452 | 0.26868 | 0.57914 | 0.16868 |
| Original Qwen, 2e-5 | 20,063 | 0.17524 | 0.17555 | 0.78283 | 0.03803 | 0.24126 | 0.62424 | 0.07131 |
| F0 copy, 2e-6 | 20,063 | 0.18006 | 0.18573 | 0.75620 | 0.05304 | 0.24565 | 0.60963 | 0.09368 |
| **F0 copy, 2e-5** | **20,000** | **0.16977** | **0.17302** | **0.79683** | **0.03710** | **0.22945** | **0.68922** | **0.09419** |
| **Platt on the same 20,063 actions** | — | **0.16179** | **0.16855** | **0.80026** | **0.02165** | **0.22724** | 0.66783 | **0.04239** |
| Raw F0 chosen-answer probability | — | 0.16248 | 0.16997 | 0.80026 | 0.03990 | 0.22839 | 0.66783 | 0.05002 |

The sealed JeV set contains 2,000 questions from 1,608 videos. F0 answered **66.3%** correctly; its constant-at-accuracy Brier is **0.22343**. Holmes contains 1,837 questions from 270 videos. F0 answered **43.17%** correctly; its constant-at-accuracy Brier is **0.24533**. ECE uses ten fixed equal-width bins. Platt and raw action probability have identical AUC because Platt is a monotone transformation of that probability.

## What the 2×2 comparison says

Higher learning rate helped under **both** initializations. On sealed JeV, raising the learning rate lowered Brier by **0.01784** from the original-Qwen start and **0.01271** from the F0 start. On Holmes, the corresponding reductions were **0.02742** and **0.01620**. All four video-cluster bootstrap 95% intervals exclude zero.

Copying F0 also helped. At low learning rate it lowered Brier by **0.00766** on sealed JeV and **0.02303** on Holmes. At high learning rate the gains were **0.00253** on JeV, with interval **[-0.00105, 0.00616]**, and **0.01181** on Holmes, with interval **[0.00699, 0.01643]**. Positive values here mean the second configuration is better. The full set of paired contrasts and intervals is in [the compact contrast file](f0_confidence_factorial_contrasts.json). Thus both factors matter, while the learning-rate change is the larger and more consistent effect on JeV in this tested range. The effect of F0 initialization is especially visible on Holmes.

The best development-selected configuration, **F0 copy at 2e-5**, did **not establish a Brier advantage over Platt**. Its Brier minus Platt is **+0.00447** on sealed JeV (video bootstrap 95% interval **[-0.00061, 0.00956]**) and **+0.00222** on Holmes (**[-0.00397, 0.00843]**). Both intervals include zero. Its Holmes AUC point estimate is higher (0.68922 versus 0.66783), but Holmes ECE is worse (0.09419 versus 0.04239); no AUC uncertainty analysis was used to select or claim a winner. The best LoRA's mean confidence is 0.6967 on JeV versus 0.6630 accuracy, and 0.5118 on Holmes versus 0.4317 accuracy, showing residual overconfidence.

Keeping F0 fixed removes the earlier mismatch between half-data answer models used for confidence training and the full-data answer model used for testing. The remaining gap to Platt therefore cannot be attributed to that particular mismatch. The experiment shows that both starting from F0 and using the higher learning rate improve the confidence model, but neither was sufficient to beat Platt on Brier in these sealed tests. This does not justify transferring the confidence LoRA to the full-data answer model P yet. Holmes still differs from JeV in source and F0 accuracy, so transfer remains a separate issue.

## Reproducibility and audit

The [audit script](audit_f0_confidence_ablation.py) independently verified 20,063 / 2,000 / 2,000 / 1,837 rows, zero pairwise video-filename overlaps among fold0, train, dev, sealed JeV test and Holmes, the same frozen F0 action for every variant, selected-checkpoint IDs, and recomputed Brier/AUC. The [selection record](f0_confidence_selection.json) lists all 24 development checkpoint scores. [Final metrics](f0_confidence_final_summary.json), [split manifest](f0_confidence_ablation_manifest.json) and [Platt fit](f0_confidence_platt_parameters.json) are compact and committed. Model weights, optimizer states and per-question predictions remain only in `/pfs/hyx/videojev-rlcd`.

## Exploratory Platt calibration of the best confidence LoRA

On 2026-10-08, fit a separate two-parameter Platt calibrator on the 2,000 development predictions of the best development-selected confidence adapter (`f0_high`, step 20,000). For a matched comparator, fit another Platt calibrator on the F0 chosen-answer probabilities from those exact development questions, using the same binary-cross-entropy objective and 1e-4 slope penalty. Both calibrators take the logit of their respective probability as input. No test labels enter either fit. This analysis is **exploratory**: the development labels had already selected the LoRA checkpoint, and the 2,000 test results had already been inspected.

| Method on the same 2,000 JeV test questions | Brier ↓ | AUC ↑ | Ten-bin ECE ↓ |
| --- | ---: | ---: | ---: |
| Raw confidence LoRA | 0.17302 | 0.79683 | 0.03710 |
| Development-fitted Platt(confidence LoRA) | 0.17182 | 0.79683 | 0.03898 |
| Raw F0 answer probability | 0.16997 | 0.80026 | 0.03990 |
| Development-fitted Platt(F0 answer probability) | **0.16827** | **0.80026** | **0.01648** |
| Previous training-fitted Platt(F0 answer probability) | 0.16855 | 0.80026 | 0.02165 |

Platt(LoRA) improves the raw LoRA Brier by 0.00120, but its Brier exceeds the matched Platt(answer probability) by **0.00355**. A 5,000-draw video-cluster bootstrap across the 1,608 test videos gives a 95% interval **[-0.00133, 0.00847]** for that difference. It does not establish that Platt(LoRA) is better; the point estimate favors answer probability. Fitted slope/intercept are 0.94283 / -0.05922 for LoRA and 0.91729 / -0.11061 for answer probability. Both slopes are positive, so each transformation preserves its input ranking and AUC. This test does not support a confidence advantage hidden solely by the LoRA's probability scale.

The exact inputs, parameters and compact results are in [f0_lora_platt_exploratory_summary.json](f0_lora_platt_exploratory_summary.json); the reproducible analysis is [calibrate_f0_confidence_exploratory.py](calibrate_f0_confidence_exploratory.py). This analysis does not alter the ongoing joint-action/confidence experiment or its fresh test split.
