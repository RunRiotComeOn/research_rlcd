# Frozen F0: one LoRA for action and confidence

The six training runs and all 30 development-checkpoint evaluations are complete. **No checkpoint met the preregistered deployment gate:** at least 99% action agreement with frozen F0 and no development accuracy loss. The minimum-development-Brier checkpoint, teacher weight λ=1, action KL weight 5, after 16,000 training questions, was evaluated on the reserved JeV set and Holmes **for diagnosis only**. It is not a replacement for F0 or the Platt baseline.

## Design and selection

F0 learned only JeV fold0. Its unseen fold1 training pool was split by video filename into 16,063 confidence-training, 2,000 development and 2,000 newly reserved JeV test questions. The same F0 generated actions, correctness labels and action probabilities for all three sets and Holmes. A Platt teacher was fitted only on the 16,063 training questions. Each student started from F0 and used the same LoRA for its own greedy `Action` and conditional `Confidence` digit, with loss `(E[q]−y)² + λ(E[q]−q_teacher)² + K·KL(F0 action || student action)`. We tested λ ∈ {0, 0.5, 1} and K ∈ {5, 50}; K=50 was a development-only amendment after the first K=5 result missed the action-agreement gate. All six ran one epoch at 2e-5 → 2e-6 and were checked at 4,000, 8,000, 12,000, 16,000 and 16,063 questions. No new answer model was trained.

All 30 checkpoints missed 99% agreement. The highest observed development agreement was **97.6%** (K=50, λ=1, step 16,063), with 67.7% accuracy versus F0's 67.9%. The lowest development Brier was **0.17035** (K=5, λ=1, step 16,000): 97.1% agreement and 67.9% accuracy, equal to F0. Stronger KL did not attain the agreement requirement and its best Brier was worse. The [selection record](f0_joint_confidence_selection.json) contains every checkpoint. The development-only selection was saved at 03:04 China time on October 9, before the reserved JeV result at 03:19 and Holmes result at 03:38.

## Reserved results for the diagnostic model

The primary confidence is the mean over the model's ten output-digit probabilities, conditioned on **its own** selected action. There is no inference-time Platt adjustment. Lower Brier and ECE are better; higher AUC is better. ECE uses ten fixed equal-width bins.

| Data / direct readout | Brier ↓ | AUC ↑ | ECE ↓ |
| --- | ---: | ---: | ---: |
| New JeV test: joint model mean `q` | 0.17324 | 0.77278 | 0.02324 |
| New JeV test: joint model's own answer probability | 0.16592 | 0.79414 | 0.02631 |
| New JeV test: F0 probability + train-fitted Platt | **0.16550** | **0.79459** | **0.01305** |
| Holmes: joint model mean `q` | 0.23390 | 0.66228 | 0.08461 |
| Holmes: joint model's own answer probability | 0.22936 | 0.66448 | 0.04778 |
| Holmes: F0 probability + train-fitted Platt | **0.22733** | **0.66783** | **0.04361** |

On the new JeV test, the joint model's direct mean confidence is **0.00774 Brier worse** than F0 + Platt; the paired video-cluster bootstrap 95% interval is **[0.00380, 0.01172]**. On Holmes it is **0.00657 worse**, interval **[0.00227, 0.01076]**. Both intervals favor Platt. The joint model's own answer-token probability also outperforms its generated confidence digits. Taking the most likely digit rather than their probability-weighted mean is worse again: Brier **0.19205** on JeV and **0.24080** on Holmes.

The joint model matched F0's action on **96.5%** of the new JeV test and **94.45%** of Holmes questions. It answered **68.90%** and **43.55%** correctly, versus F0's **68.85%** and **43.17%**. Paired video-cluster 95% intervals for the accuracy differences are **[-0.70, +0.79]** and **[-0.49, +1.19] percentage points**, respectively. The small accuracy point gains do not establish an improvement and do not fix the failed action-agreement gate. Constant-at-accuracy Brier baselines for the joint model are **0.21428** (JeV) and **0.24584** (Holmes).

## Interpretation and limits

The single-adapter design works mechanically: one trained LoRA supplies both action and confidence at two autoregressive positions. Here, neither the teacher term nor a tenfold stronger action KL preserved 99% of F0's choices. Its explicit confidence digits lost to the action probability on the same model and to train-fitted Platt on F0. This experiment therefore does not support adopting the joint LoRA or transferring it to the full-data answer model P. A future action-preserving attempt would need to enforce the action constraint more directly and establish its benefit on development data before opening another reserved test.

This JeV test is newly reserved for this experiment. Holmes contains 1,837 questions from 270 videos and has been inspected in earlier experiments, so its result is **not a fresh blind test**. The JeV pool includes some training-source entries named `Video-Holmes` and `OneThinker-Holmes-train`, but the audit found **zero shared video filenames or question IDs** among F0 fold0, the three new JeV splits and Holmes. Video-cluster intervals reflect the 1,572 JeV and 270 Holmes videos, not independent-question resampling.

## Reproducibility and audit

The [independent audit](audit_f0_joint_confidence.py) checked the 24,063 F0-training, 16,063 confidence-training, 2,000 development, 2,000 test and 1,837 Holmes rows; source hashes; pairwise video-filename and question-ID nonoverlap; F0 action alignment; all 30 checkpoint statuses and gate decisions; selected version; final action/correctness alignment; and recomputed Brier, AUC, ECE and the constant baselines. Its compact [output](f0_joint_confidence_audit.json), [final metrics](f0_joint_confidence_final_summary.json), [split manifest](f0_joint_confidence_manifest.json) and [protocol](F0_JOINT_CONFIDENCE_PROTOCOL.md) are committed. Checkpoints and per-question predictions remain under `/pfs/hyx/videojev-rlcd`.
