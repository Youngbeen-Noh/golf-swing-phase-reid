# New baselines (2nd revision instruction 4)

Status of each requested comparison. Run on the server, after/alongside the
main sweep -- nothing here needs new preprocessing, only new training runs
plus (for CE) one new eval script.

## Done, ready to run

### Cross-entropy classifier (필수)
Isolates whether metric learning itself matters, vs. a plain closed-set
classifier. No new code -- the model already has an optional classifier head
(`--lambda-classification`), just unused by the main sweep (`lambda_classification=0.0`
default). Zero out the contrastive losses so training is CE-only:

```bash
python 03_preprocess_and_build_features.py --sport Golf --preprocess-method Golf-D \
  --run-id ATCN_Golf-D-CE --overwrite   # reuses the winning preprocessing method
python 04-1_train_atcn.py --sport Golf --run-id ATCN_Golf-D-CE-s42 \
  --lambda-sequence 0 --lambda-step 0 --lambda-classification 1.0 --seed 42
# repeat seeds 43-46, and 04-2/04-3 for Transformer/STGCN
```

Evaluate the classifier head DIRECTLY (bypasses retrieval/prototypes entirely
-- this is the actual "CE classifier" number):
```bash
python 21_evaluate_ce_classifier.py \
  --checkpoint data/Golf/experiments/ATCN_Golf-D-CE-s42/models/atcn_step_sequence_best.pt \
  --user-feature-dir data/Golf/experiments/ATCN_Golf-D-CE-s42/features/user \
  --out data/Golf/experiments/ATCN_Golf-D-CE-s42/results/evaluation/pro_holdout_user_metrics.json
```
Output is `pro_holdout_user_metrics.json`-compatible, so it plugs directly
into `10_aggregate_seed_metrics.py`, `11_paired_significance.py`,
`19_paired_significance_hierarchical.py`, `20_query_level_logistic_variance.py`
unchanged.

### CE embedding + nearest prototype (필수)
Same checkpoint as above, no new code at all -- just run it through the
EXISTING retrieval pipeline instead of (or in addition to) the direct-argmax
eval, since 05/06/07 only ever look at the embedding, never the classifier
head:
```bash
python 05_build_expert_prototypes.py --sport Golf --run-id ATCN_Golf-D-CE-s42
python 06_infer_user_similarity.py --sport Golf --run-id ATCN_Golf-D-CE-s42
python 07_evaluate.py --sport Golf --run-id ATCN_Golf-D-CE-s42
```
This isolates whether nearest-prototype retrieval adds anything beyond
whatever discriminative structure CE training alone puts in the embedding.

### Triplet loss (필수)
New `triplet_batch_hard_loss()` in `train_common.py` (FaceNet-style
batch-hard: per anchor, hardest same-label positive vs. hardest
different-label negative in the P*K batch), wired through a new
`--metric-loss {supcon,triplet}` flag on all three trainers:
```bash
python 03_preprocess_and_build_features.py --sport Golf --preprocess-method Golf-D \
  --run-id ATCN_Golf-D-triplet --overwrite
python 04-1_train_atcn.py --sport Golf --run-id ATCN_Golf-D-triplet-s42 \
  --metric-loss triplet --triplet-margin 0.3 --seed 42
```
Then the normal 05/06/07 retrieval pipeline -- directly comparable to the
main SupCon results.

Verified locally (2-expert, 2-epoch smoke test only, no real training run):
loss decreases and backpropagates cleanly for both `--metric-loss triplet`
and the CE-only config; `21_evaluate_ce_classifier.py` produces a valid,
schema-correct report.

### DMSM reimplementation (강력 권장) -- done
Found the actual paper (`sensors-25-07073.pdf`, Lee, S.-S.; Choi, J.-H.;
Byun, J.; Hwang, K.-I. "Dynamic Golf Swing Analysis Framework Based on
Efficient Similarity Assessment." *Sensors* 2025, 25, 7073 -- this paper's
own reference [3]) via the author. Not a training method -- no learned
model, so no server run needed at all. `22_evaluate_dmsm_baseline.py`
reimplements the paper's Eq. 1-2 directly: per-phase, per-joint trapezoidal
integral of |Δx|+|Δy| between address-aligned trajectories, using the
existing per-phase `steps` feature array (already phase-synchronized-
resampled, the same idea as the paper's own phase-wise alignment). Two
things the paper leaves unspecified are documented, not derived, choices:
the raw-dissimilarity-to-"similarity" conversion (used `1/(1+D)`, a strictly
monotonic function of D so it cannot change *which* expert ranks first,
only the reported score's scale), and clip-vs-gallery aggregation for this
paper's retrieval-style evaluation (used nearest-neighbor over each
expert's training clips, since the original paper only evaluates pairwise
clip-vs-clip, not retrieval against a gallery).

Verified locally against real feature data (3 experts, no training needed):
```bash
python 22_evaluate_dmsm_baseline.py \
  --expert-feature-dir data/Golf/experiments/ATCN_Golf-D/features/expert \
  --user-feature-dir data/Golf/experiments/ATCN_Golf-D/features/user \
  --out data/Golf/experiments/ATCN_Golf-D/results/evaluation/dmsm_pro_holdout_metrics.json
```
Output is `pro_holdout_user_metrics.json`-compatible, so it plugs directly
into `10_aggregate_seed_metrics.py`, `11/19_paired_significance*.py`, and
`20_query_level_logistic_variance.py` unchanged -- can be run against the
*current* preprocessed features right now, doesn't need to wait for the
server sweep.

## Not yet done

- **Prototypical/episodic-prototype loss (권장)** -- not started.
- **Gallery control (필수)** -- apply the same capped/uncapped candidate
  gallery across every baseline comparison above; this is a protocol
  requirement (use the same `--max-samples-per-expert` value everywhere
  being compared), not new code, but I haven't gone through and confirmed
  every planned run uses a consistent value.
- **Event robustness: manual vs. automatic events (필수)** -- compare
  retrieval trained/evaluated on manually-labeled event positions vs. the
  automatic detector's positions. `12a_make_event_gt_template.py` /
  `12b_label_events_tool.py` suggest partial manual GT labels may already
  exist; haven't checked coverage or wired this comparison up.
- **Event perturbation ±5/±10/±20 frames (권장)**, **prototype robustness
  variants (mean/medoid/trimmed/multi-prototype) (권장)** -- not started.
