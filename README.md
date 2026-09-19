# Learning Golf Swing Signatures from Monocular 2D Pose

Analysis code for the manuscript:

> Noh, Y.-B.; Jeong, H.-Y.; Byun, J.-E.; Hwang, K.-i.
> *Learning Professional Golf Swing Signatures from Monocular 2D Pose:
> A Controlled Study of Coordinate Normalization and Phase Decomposition.*
> Submitted to *Sensors* (MDPI).

The pipeline extracts 18-joint 2D pose sequences from monocular golf swing
videos, segments each swing into 8 events / 7 phases with a rule-based
detector, trains sequence- and phase-level embeddings jointly with a
supervised contrastive objective (A-TCN / Transformer / ST-GCN encoders,
five coordinate-normalization conditions, three training options), and
evaluates closed-set professional re-identification with expert prototypes
and phase-wise similarity profiles.

## Data availability

The anonymized extracted 2D pose sequences (450 clips, 25 experts) and the
validation / held-out-test split manifests **are included** in this
repository under `data/Golf/jsons/` and `data/Golf/splits/` — see
[`data/README.md`](data/README.md) for the schema and the anonymization
policy (experts are referenced only as `expert_01`–`expert_25`; the
identity mapping is not released). The original videos are third-party
copyrighted content and cannot be redistributed; raw result files are
available from the corresponding author upon reasonable request. Scripts
expect the directory layout `data/Golf/{videos,jsons,tensor}/...` described
in `ENVIRONMENT.md`; with the released jsons in place, everything from
`02_json_to_tensor.py` onward can be reproduced (only `01-1` pose
extraction requires the videos).

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
CUDA 12.4; pose extraction uses the official yolo26l-pose.pt weights, checkpoint version Ultralytics 8.3.222 -- note versions before YOLO26 support cannot load these weights). Statistical tests in
the manuscript use a crossed (seed × query) bootstrap (10,000 resamples,
shared resampled indices across compared conditions, Holm correction); see
`30_crossed_bootstrap_stats.py`. `19_paired_significance_hierarchical.py`
implements the earlier two-level hierarchical scheme and is kept for
reference.

## License

MIT — see `LICENSE`.
