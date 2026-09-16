# Learning Golf Swing Signatures from Monocular 2D Pose

Analysis code for the manuscript:

> Noh, Y.-B.; Jeong, H.-Y.; Byun, J.-E.; Hwang, K.-i.
> *Learning Golf Swing Signatures from Monocular 2D Pose: Closed-Set
> Professional Re-Identification with Phase-Level Interpretability.*
> Submitted to *Sensors* (MDPI).

The pipeline extracts 18-joint 2D pose sequences from monocular golf swing
videos, segments each swing into 8 events / 7 phases with a rule-based
detector, trains sequence- and phase-level embeddings jointly with a
supervised contrastive objective (A-TCN / Transformer / ST-GCN encoders,
five coordinate-normalization conditions, three training options), and
evaluates closed-set professional re-identification with expert prototypes
and phase-wise similarity profiles.

## Data availability

No data are included in this repository. The original videos are
third-party copyrighted content and cannot be redistributed. The extracted
2D pose sequences, split manifests, raw result files, and the expert
rosters (withheld here under the paper's anonymization policy — experts are
referenced only by anonymized IDs) are available from the corresponding
author upon reasonable request. Scripts expect the directory layout
`data/Golf/{videos,jsons,tensor}/...` described in `ENVIRONMENT.md`.

## Pipeline overview (script numbering = execution order)

| Stage | Scripts |
|---|---|
| Pose extraction + event detection | `01-1_extraction_golf.py` |
| Splits (group-aware holdout, validation/locked test) | `01-3_*.py`, `01-3b_*.py`, `01-3c_*.py` |
| Tensorization + preprocessing/features | `02_json_to_tensor.py`, `03_preprocess_and_build_features.py` |
| Training (A-TCN / Transformer / ST-GCN) | `04-1/2/3_train_*.py`, `train_common.py` |
| Prototypes, inference, evaluation | `05`–`08`, `encoder_loader.py` |
| Experiment sweep driver | `09_run_golf_experiments.py`, `run_both_experiments.sh` |
| Aggregation + statistics | `10`, `11`, `13`, `19`, `20` |
| Event-detection accuracy + sensitivity | `12`, `12a`, `12b`, `24`, `26` |
| Baselines (cosine/DTW/DMSM, CE/triplet) | `baseline_retrieval.py`, `21`, `22` |
| Ablations (frame budget, shuffled phase, prototypes, capacity, finish hold) | `run_*.sh`, `27`, `28`, `ABLATION_FRAME_BUDGET.md`, `BASELINES.md` |
| Confusion/recall, efficiency, t-SNE, qualitative cases | `15`, `16`, `17`, `23`, `29` |
| Near-duplicate session review | `18_near_duplicate_dtw_reverify.py` |
| Locked-test confirmatory evaluation | `finalize_locked_test_eval.sh` |
| Post-split re-evaluation repair | `repair_stale_user_holdout.py` |

## Environment

See `ENVIRONMENT.md` and `requirements.txt` (Python 3.12, PyTorch 2.6.0 +
CUDA 12.4, Ultralytics 8.3.53 for pose extraction). Statistical tests use a
two-level hierarchical bootstrap (queries × seeds, 10,000 resamples); see
`19_paired_significance_hierarchical.py`.

## License

MIT — see `LICENSE`.
