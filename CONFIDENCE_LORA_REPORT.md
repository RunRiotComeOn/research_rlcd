# Qwen-native confidence LoRA experiment

Status: complete. All artifacts are under `/pfs/hyx/videojev-rlcd`.

## Setup

- Frozen answer policy: `runs/ablation_none_200/checkpoints/step_0200` (Qwen3.5-2B).
- The answer is generated once by the frozen policy. A separate LoRA, initialized from that adapter, receives the same video and the fixed `Action: X\nConfidence: ` prefix. Only its LoRA parameters are trainable.
- The confidence score is the expected value of 10 restricted digit-token probabilities, with bin centers 0.05 through 0.95. Training labels are the binary correctness of the frozen answer. No teacher confidence labels are used.
- Data: 1,000 fit examples; 300 select examples for checkpoint choice; 300 blind examples. These are disjoint from the original SFT 100 and RL 200. Holmes 92 is exploratory and was not used for model choice.
- Each run trained for 1,000 examples and saved checkpoints at 250, 500, 750, and 1,000. We chose the checkpoint with lowest **discrete** Brier on select before opening blind and Holmes.

## Selection results

| Objective | Learning rate | Best step | Select Brier | Select AUC |
|---|---:|---:|---:|---:|
| Brier | 1e-5 | 750 | 0.2062 | 0.7625 |
| BCE | 1e-5 | 750 | 0.2160 | 0.7349 |
| Brier | 1e-6 | 250 | 0.2404 | 0.5997 |

The chosen artifact is `calibration/confidence_lora_v1/checkpoints/step_0750/confidence`.

## Comparison (discrete 0–9 output)

| Method | Select Brier | Blind Brier | Blind AUC | Holmes Brier | Holmes AUC |
|---|---:|---:|---:|---:|---:|
| Original explicit digit | 0.2353 | 0.2330 | — | 0.2766 | — |
| Platt on native selected-action probability | 0.1910 | **0.1862** | **0.7774** | 0.2438 | 0.6233 |
| Eight-feature MLP head | **0.1858** | 0.1908 | 0.7702 | **0.2358** | **0.6353** |
| Confidence LoRA (Brier, step 750) | 0.2062 | 0.2124 | 0.7437 | 0.3088 | 0.5612 |

Frozen action accuracy is 59.7% on blind and 38.0% on Holmes. The new LoRA keeps the action string exactly unchanged and all outputs satisfy the two-line format. Its confidence spans all ten bins on blind; the issue is calibration and ranking quality, not merely lack of bin diversity.

## Interpretation

This version does not meet the replacement criterion: its blind Brier is 0.0262 higher than Platt, and its Holmes Brier is 0.0650 higher. Training was unstable: the 1e-5 Brier run overpredicted near step 250, improved by step 750, then spread toward extreme bins by step 1000. BCE and a tenfold smaller learning rate did not improve select Brier. The evidence favors retaining Platt as the deployed confidence method and redirecting the next experiment toward the answer policy's cross-dataset accuracy gap.

## Files and checks

- Training/evaluation code: `confidence_lora.py`; audit: `audit_confidence_lora.py`.
- Run directories: `calibration/confidence_lora_v1`, `calibration/confidence_lora_bce_v1`, `calibration/confidence_lora_low_lr_v1`.
- Final per-example outputs and metrics: `calibration/confidence_lora_v1/evals/selected`.
- Audit passed for 300 blind and 92 Holmes rows: exact original answer retained, correctness retained, strict output parsed, bin consistent with score.
- `python -m py_compile` passed; existing 3 project tests passed.
