# Frozen-F0 confidence ablation protocol

This experiment isolates confidence-model initialization and learning rate while holding the answer policy fixed. The answer policy is the completed F0 adapter at `/pfs/hyx/videojev-rlcd/runs/jevfull_action_fold0_v1/checkpoints/examples_24063`, trained only on JeV fold0. It is **never updated** in this experiment. Existing F0 predictions on fold1 provide actions, correctness labels and chosen-answer probabilities; a separate pass collects F0 predictions on all 1,837 Holmes questions.

With seed 20261007, split the 24,063 F0-unseen fold1 questions by **video filename** into 20,063 confidence-training questions, 2,000 development questions and 2,000 sealed same-source test questions. The exact split and input hashes are in [the manifest](f0_confidence_ablation_manifest.json). The three sets share no video filenames, and JeV and Holmes share no video filenames. Only derived files under `/pfs/hyx/videojev-rlcd` may be written.

Train four independent rank-8 confidence LoRAs, all on the identical 20,063 questions and frozen F0 actions:

| Variant | Initialization | Cosine learning rate, one pass |
| --- | --- | --- |
| `base_low` | Original Qwen3.5-2B | 2e-6 → 2e-7 |
| `base_high` | Original Qwen3.5-2B | 2e-5 → 2e-6 |
| `f0_low` | Copy of frozen F0 answer adapter | 2e-6 → 2e-7 |
| `f0_high` | Copy of frozen F0 answer adapter | 2e-5 → 2e-6 |

The loss is `(E[q] - y)^2`, where `E[q]` is the mean of the ten confidence-bin softmax and `y` is whether the frozen F0 answer was correct. The copied F0 adapter is updated **only in the confidence branch**; the answer policy and all scored actions stay fixed. Use effective batch 8 and save adapter checkpoints at steps 4,000, 8,000, 12,000, 16,000, 20,000 and 20,063. The latest optimizer state is retained for recovery; previous adapter weights are retained for development evaluation.

Fit one two-parameter Platt model on the same 20,063 F0 actions and correctness labels, with the original 1e-4 slope penalty. Evaluate all six checkpoints of each confidence variant on the same 2,000 development questions. **Select the minimum development Brier checkpoint independently for each variant**, breaking exact ties toward the earlier checkpoint. The ten-bin ECE and AUC are reported, but do not participate in checkpoint selection.

After freezing the four choices in `selection.json`, evaluate each selected checkpoint once on the sealed 2,000 JeV questions and once on all 1,837 Holmes questions. Compare confidence LoRA with Platt and the raw F0 chosen-answer probability using Brier, AUC and ten equal-width-bin ECE. Report the test-set constant-confidence Brier and video-cluster bootstrap intervals for the LoRA-minus-Platt Brier difference. No test or Holmes label is used for checkpoint selection or training. Comparing these test results identifies whether either initialization or learning rate improves confidence under a fixed answer policy; causal attribution to either factor still depends on the full 2×2 pattern and uncertainty.

Scripts: [split](prepare_f0_confidence_ablation.py), [Holmes F0 actions](collect_f0_holmes_actions.py), [training](train_f0_confidence_ablation.py), [Platt](fit_f0_confidence_platt.py), [evaluation](eval_f0_confidence_ablation.py), [selection](select_f0_confidence_checkpoints.py), [final summary](summarize_f0_confidence_ablation.py), and [pipeline](run_f0_confidence_pipeline.sh).
