# Full JeV training and full Holmes evaluation

Completed 2026-10-06 (Asia/Shanghai), following [the frozen protocol](JEV_FULL_HOLMES_PROTOCOL.md). The production answer LoRA was trained for one pass on all 48,126 JeV questions. Two additional answer LoRAs were trained on video-disjoint halves (24,063 questions each) and predicted the opposite half, yielding 48,126 out-of-fold correctness labels. The independent confidence LoRA and the two-parameter Platt baseline used those labels. All 1,837 Holmes questions were reserved for the final test; no Holmes label selected a checkpoint or fitted a parameter.

| Holmes method, same answer actions | Brier ↓ | AUC ↑ | Mean confidence |
| --- | ---: | ---: | ---: |
| Chosen-answer softmax probability | 0.22975 | 0.66890 | 0.51152 |
| OOF-fitted Platt | **0.22853** | **0.66890** | 0.50028 |
| Independent confidence LoRA | 0.24882 | 0.57526 | 0.53390 |
| Constant confidence at Holmes accuracy | 0.24981 | 0.50000 | 0.48612 |

The full-data answer LoRA got **48.61%** correct (893/1,837). The confidence LoRA's Brier exceeds Platt's by **0.02029**. A 5,000-replicate bootstrap resampling Holmes videos, rather than individual questions, gives a 95% interval of **[0.01449, 0.02629]** for that difference. Positive values favor Platt. The confidence LoRA barely improves on the constant-confidence Brier and loses much of the rank information in the answer probability. It should **not** replace Platt for this evaluation setting.

The OOF answer models scored 68.91% on JeV; their fitted Platt Brier on the same OOF labels was 0.16333. Those are development figures, not Holmes test performance. The 20.30 percentage-point OOF-to-Holmes accuracy gap suggests substantial distribution shift, although the OOF models saw half as many JeV training questions as the final answer model. This experiment cannot isolate how much of the gap comes from dataset shift versus model-training size.

## Integrity checks

- Independently audited 48,126 JeV rows, two 24,063-row folds, two 24,063-row OOF prediction files, and 1,837 matching Holmes action and confidence predictions.
- JeV and Holmes share **zero video filenames**; the two JeV folds also share **zero video filenames**. Holmes has 270 video filenames. OOF predictions match their target folds by ID and are produced by the opposite fold's answer model.
- Recomputed accuracy, Brier, mean confidence, AUC and constant Brier from per-question Holmes predictions; all match the saved summary. The video bootstrap interval is from the frozen evaluation script, with seed 20261005.
- All generated data, model weights, optimizer checkpoints, logs and per-question predictions remain under `/pfs/hyx/videojev-rlcd` and are excluded from Git. The source datasets were read only. The one unusable distractor repaired in the derived JeV copy is documented in [the manifest](jev_full_holmes_data_manifest.json).

Compact artifacts: [Holmes summary](jev_full_holmes_summary.json), [OOF Platt fit](jev_full_holmes_platt_parameters.json), [data manifest](jev_full_holmes_data_manifest.json), and [fold manifest](jev_full_holmes_folds_manifest.json). The repeatable row and metric audit is [audit_jev_full_holmes.py](audit_jev_full_holmes.py).

The subsequent [online JeV and Holmes reliability diagnostics](JEV_FULL_HOLMES_DIAGNOSTICS.md) show that the confidence LoRA also trails a prefix-fitted Platt model on the same 5,000 JeV questions before their individual updates (Brier 0.17796 versus 0.16290).
