# Frame-budget-controlled phase-vs-sequence ablation (2nd revision instruction 2.4)

Run these on the server, after the main validation sweep. All commands assume
the working directory is the repo root and use the winning preprocessing
method from the main sweep in place of `Golf-D` below.

## Why

The legacy sequence branch resamples the whole PSR-normalized span
(optionally including a synthetic finish-hold tail the phase branch never
sees) in one global interpolation to 160 frames. The phase branch resamples
each of the 7 real keyframe-to-keyframe segments to 20 frames each (140
total) independently. Any Seq-only vs Phase-only/Seq+Step gap is therefore
confounded with input length, tail content, and resampling granularity, not
just "phase decomposition helps." `03_preprocess_and_build_features.py` now
supports `--sequence-mode phase-concat`, which builds the sequence branch
from the *same* per-segment resampling the phase branch uses, so the two
branches see informationally identical content unless explicitly varied.

## Required conditions

```bash
# Sequence-only-140: real 7 phases, concatenated, no synthetic tail,
# no extra resampling (7 x 20 = 140 exactly -- phase_concat_sequence's
# output already IS 140 frames, so sequence-length-golf=140 is a no-op resample)
python 03_preprocess_and_build_features.py --sport Golf --preprocess-method Golf-D \
  --run-id ATCN_Golf-D-seq140 --sequence-mode phase-concat --no-finish-hold \
  --sequence-length-golf 140 --overwrite
python 04-1_train_atcn.py --sport Golf --run-id ATCN_Golf-D-seq140 --lambda-step 0 --seed 42
# repeat --seed for 43,44,45,46; repeat with 04-2/04-3 for Transformer/STGCN

# Sequence-only-160: same real content, one extra holistic resample to 160
# (isolates "does matching the legacy 160-frame length matter" from the
# tail/granularity confound already removed above)
python 03_preprocess_and_build_features.py --sport Golf --preprocess-method Golf-D \
  --run-id ATCN_Golf-D-seq160 --sequence-mode phase-concat --no-finish-hold \
  --sequence-length-golf 160 --overwrite
python 04-1_train_atcn.py --sport Golf --run-id ATCN_Golf-D-seq160 --lambda-step 0 --seed 42

# Phase-only: unchanged, already exists (--lambda-step 1 / Step-only training option
# in the existing 09_run_golf_experiments.py sweep)

# Seq+Phase, frame-budget-matched: fuse using the seq140 run's sequence branch
# instead of the legacy 160-with-tail one (default training options already
# combine sequence+step; just point at the seq140 feature run-id)
python 04-1_train_atcn.py --sport Golf --run-id ATCN_Golf-D-seq140 --seed 42
```

Run all four x {ATCN, Transformer, STGCN} x 5 seeds, same as the main sweep,
then feed the resulting `pro_holdout_user_metrics.json` files into
`19_paired_significance_hierarchical.py` and `20_query_level_logistic_variance.py`
for the fair-comparison significance test (Sequence-only-140 vs Phase-only vs
Seq+Phase-140), and into `10_aggregate_seed_metrics.py` for the seed-mean
Top-1/MRR table.

## Recommended conditions

```bash
# Shuffled-phase: same as seq140, phase order randomized before concatenation
# (tests whether the sequence encoder's result depends on temporal phase
# order specifically, not just phase content)
python 03_preprocess_and_build_features.py --sport Golf --preprocess-method Golf-D \
  --run-id ATCN_Golf-D-seqshuffled --sequence-mode phase-concat --no-finish-hold \
  --sequence-length-golf 140 --shuffle-phase-order --shuffle-seed 0 --overwrite
python 04-1_train_atcn.py --sport Golf --run-id ATCN_Golf-D-seqshuffled --lambda-step 0 --seed 42
```

```bash
# Equal-capacity: scale Transformer/STGCN hidden-dim up to A-TCN's parameter
# count (848,896 retrieval params total / 424,448 per branch), so an
# architecture comparison isn't confounded by A-TCN having ~7-12x more
# capacity (Section 5.3's open question). Found by direct parameter-count
# search (see below); A-TCN itself is left unchanged.
python 04-2_train_transformer.py --sport Golf --run-id ATCN_Golf-D-eqcap \
  --model-dim 192 --num-layers 1 --feedforward-dim 384 --seed 42
  # -> 869,268 total params (vs A-TCN 848,896, +2.4%)
python 04-3_train_stgcn.py --sport Golf --run-id ATCN_Golf-D-eqcap \
  --hidden-dim 192 --num-blocks 3 --seed 42
  # -> 847,616 total params (vs A-TCN 848,896, -0.15%)
```

Parameter-count search method: instantiated `TemporalTransformerEncoder` /
`STGCNEncoder` directly at a grid of hidden-dim values (no training needed,
just `sum(p.numel() for p in model.parameters())`), picked the closest match
to A-TCN's existing 424,448-per-branch count. Transformer defaults
(`embedding-dim 128`, `num-heads 4`) and ST-GCN defaults (`embedding-dim 64`)
were kept fixed; only `model-dim`/`feedforward-dim`/`num-layers` (Transformer)
and `hidden-dim`/`num-blocks` (ST-GCN) were swept.

## Completion criterion (per the instruction)

Phase-only or Seq+Phase's advantage over Sequence-only-140 must hold up
without being explained by input length, the synthetic tail, or parameter
count. If Sequence-only-140 (or -160) closes most of the gap to Phase-only,
the paper's phase-decomposition claim needs to be softened accordingly.
