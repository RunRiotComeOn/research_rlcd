# VideoJev RLCD prototype

This repository contains the research code, experiment protocols, and compact result summaries. The video data, model weights, checkpoints, caches, and per-question predictions stay on the training server and are excluded from Git.

The completed [full-JeV / full-Holmes experiment](JEV_FULL_HOLMES_REPORT.md) trained the answer adapter on all 48,126 JeV questions, used two-fold out-of-fold labels for confidence training, and tested all 1,837 Holmes questions. Holmes answer accuracy was 48.61%; Platt calibration beat the independent confidence LoRA on Brier (0.22853 versus 0.24882).

The later [frozen-F0 confidence ablation](F0_CONFIDENCE_ABLATION_REPORT.md) compared original-Qwen versus F0 initialization and 2e-6 versus 2e-5 learning rates while keeping the answer policy fixed. Both F0 initialization and higher learning rate helped. Its best development-selected confidence LoRA nearly matched Platt on sealed JeV and Holmes Brier, without establishing a Brier advantage.

## Current JeV decision-model result

The current inference baseline is the Qwen3.5-2B answer LoRA at `runs/jev10_action_sft_v1/checkpoints/step_4813`, with corrected `Action: <LETTER>` token scoring and Platt calibration fitted on the JeV validation split. The original answer LoRA was trained with a token-position mistake; see [the token audit and correction report](JEV10_DECISION_CONTEXT_REPORT.md).

A later CE+Brier model trained from the base Qwen weights with the corrected token labels did not improve accuracy. On a fresh, same-source, video-disjoint 2,000-question blind set, the current baseline scored **67.75%** and CE+Brier scored **65.95%**. Their integer-percent confidence Brier scores were 0.17419 and 0.17349, respectively. See [the prespecified protocol](JEV10_NEW_BLIND_PROTOCOL.md) and [the blind-test report](JEV10_NEW_BLIND_REPORT.md); the compact paired result is in [`jev10_blind_comparison.json`](jev10_blind_comparison.json). This blind set has now been inspected and should not be reused to select later models.

The historical RLCD prototype and its original reproduction notes follow below. Later same-source JeV experiments are documented in the `JEV10_*.md` reports.

The [new RLCD reward experiment](JEV10_REWARD_20261003_REPORT.md) compares `y + β(2yq-q²)` with the earlier Brier reward using corrected action tokens. On a separate 1,000-question JeV holdout, it improved directly generated confidence Brier from 0.2616 to 0.2217 with no clear accuracy difference. The existing Platt-calibrated production baseline still had a lower Brier of 0.1744 on those same questions, so the production inference model was not changed.

The [beta and training-dynamics diagnostic](JEV10_REWARD_BETA_DIAGNOSTIC.md) decomposes the 300-step reward logs and validation trajectories. It proposes a controlled confidence-weight schedule for a future ablation; that schedule has not been trained yet.

The [β=0.4 trial](JEV10_REWARD_BETA040_REPORT.md) raised the proposed reward weight from 0.2 under matched training conditions. At the prespecified step 300 on the reused 481-question JeV validation set, direct Brier worsened from 0.2430 to 0.2654 while accuracy was 309/481 versus 312/481. The β=0.4 model was not promoted.

The [dynamic confidence-weight experiment](JEV10_REWARD_SCHEDULE_REPORT.md) held the correctness coefficient at 1.2 and tapered the confidence-error weight from 0.2 to 0.05 after step 100. On a newly frozen, video-disjoint 1,000-question JeV holdout, direct Brier improved from 0.23228 to 0.22188; accuracy was 657/1000 versus 654/1000, an uncertain difference. The production model remains unchanged.

The [exact expected-Brier pilot](JEV10_EXPECTED_BRIER_REPORT.md) removed confidence sampling, separated correctness-only action REINFORCE from confidence loss, and trained confidence on the greedy action. It gave calibration gradients even when all four action rollouts agreed. At the prespecified step 300 on the reused JeV validation set, however, actual output-bin Brier was 0.24275 versus 0.22080 for the dynamic-weight model; this pilot was not promoted to a new blind test.

The [frozen-action, separate-confidence pilot](JEV10_FROZEN_ACTION_MEANQ_REPORT.md) trained a separate LoRA directly against Brier of the ten-bin softmax mean, and exposed that mean as a percentage. All 481 validation actions stayed identical to the production answer LoRA. The prespecified step-300 confidence Brier worsened from 0.23500 before training to 0.29221; no new blind test was run.

A video-conditioned Qwen3.5-2B model generates exactly:

```text
Action: B
Confidence: 8
```

Confidence bins 0–9 map to probabilities 0.05–0.95. The reward is correctness minus `lambda_cal * (confidence - correctness)^2`, with -1 for an invalid format. `rl_only` sets lambda to zero; `rlcd` uses 0.2. Both branches start from the same format-SFT adapter and use the same deterministic training order.

## Data and isolation

All created files, caches, code, checkpoints, and logs are under `/pfs/hyx/videojev-rlcd`. The source JSON files, Qwen checkpoint, and media are read only. `prepare_data.py` selects exactly 2,406 of 48,126 video multiple-choice rows and 92 of 1,837 Holmes rows using fixed seeds. The manifest records source/output hashes. Video files and existing frame-cache tar files are referenced rather than copied. Training uses the tar cache; Holmes evaluation seeks four video frames in memory. Train/test media filenames are checked for overlap.

## Framework choice

A shallow `verl` clone is in `vendor/verl` at commit `fbb4b3a8bf636f290c9c59fc346f756849e9c241`. Its current dependencies require Torch 2.13 and vLLM 0.29, whereas this machine has Torch 2.10 and vLLM 0.14. This prototype therefore uses a local, audited grouped-policy objective with PPO-style clipping, rather than claiming a validated verl training run. The full Qwen3.5 video processor is used for both rollout and log-probability updates. The vision encoder is frozen; LoRA is applied to language-model Linear layers. Four rollouts per video share a group-normalized advantage. Group-constant rewards are skipped and recorded.


## Environment

```bash
ssh tali6
cd /pfs/hyx/videojev-rlcd
source ./env.sh
.venv/bin/python -m pytest -q tests
```

## Reproduce

```bash
.venv/bin/python prepare_data.py
.venv/bin/python train.py --mode sft --run-name sft_format --limit 100
.venv/bin/python train.py --mode rl_only --run-name rl_only --adapter runs/sft_format/adapter
.venv/bin/python train.py --mode rlcd --run-name rlcd --adapter runs/sft_format/adapter
.venv/bin/python evaluate.py --run-name vanilla
.venv/bin/python evaluate.py --run-name rl_only --adapter runs/rl_only/adapter
.venv/bin/python evaluate.py --run-name rlcd --adapter runs/rlcd/adapter
```

Use unique run names: existing runs are never overwritten. `--limit N` runs a smoke subset. The full RL commands iterate through all 2,406 selected training records. The two RL runs should be launched sequentially on the same GPU so the comparison has the same resource conditions. Evaluation writes per-example predictions, summary metrics, reliability SVG, and risk-coverage SVG under `evaluations/<name>`.

The default GPU is the first visible CUDA device. Set `CUDA_VISIBLE_DEVICES` before launching. Edit `config.yaml` for frame budget, generation, learning rate, and reward weight.

## Caveats

- The first format-SFT stage teaches the output contract with balanced arbitrary confidence bins; calibration is assessed only after RL.
- The 92-question Holmes test slice is small. Report uncertainty or rerun with additional held-out data before making research claims.
- Test `holmes.json` and training sources have no exact media filename overlap in the selected subsets; semantic duplicates across source datasets were not audited.
- Native action confidence is an auxiliary restricted-letter softmax. It is not used in the reward.
- This prototype has no fast-path DeltaNet kernels, distributed rollout engine, or optimizer-resume checkpoint. Runtime is much slower than a production verl stack.
