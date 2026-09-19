# Released data: anonymized 2D pose sequences

This directory contains the extracted 2D pose data used in the manuscript —
**no video frames or imagery**, only per-frame joint coordinates derived
from publicly posted golf swing videos, plus the detected swing-event
indices and the evaluation split manifests.

## Anonymization policy

All 25 professional golfers are referenced only by anonymized identifiers
`expert_01` … `expert_25` (directory names, file names, `sample_id`,
`video`, `video_name`). The mapping between anonymized IDs and player
identities is **not released**, and the original videos are third-party
copyrighted content that cannot be redistributed.

## Layout

```
data/Golf/jsons/expert/expert_NN/expert_NN_MMM.json    359 clips  (training / gallery pool)
data/Golf/jsons/user/expert_NN/expert_NN_MMM.json       39 clips  (validation queries)
data/Golf/jsons/user_locked_test/expert_NN/...          52 clips  (held-out test queries)
data/Golf/splits/validation_manifest.txt                39 ids
data/Golf/splits/held_out_test_manifest.txt             52 ids
```

The subset membership of the json directories **is** the split used in the
paper (session-aware, fixed seed; see `01-3c_validation_locked_test_split.py`).
The manifests list the same membership as flat id lists. Total 450 clips.

## JSON schema (one file = one swing clip)

| Field | Meaning |
|---|---|
| `sport`, `subject_type` | `"Golf"`; `"expert"` (gallery pool) or `"user"` (query subsets) |
| `sample_id` | `expert_NN_MMM` (matches the file name) |
| `video`, `video_name` | anonymized placeholder paths (videos not released) |
| `fps`, `width`, `height`, `frame_count` | source clip properties |
| `empty_frames` | frame indices where pose detection returned nothing (zero-filled downstream) |
| `steps` | detected swing events → frame index: `address`, `take_away`, `half`, `top`, `down_half`, `impact`, `follow_through`, `finish` (8 events → 7 phases) |
| `coordinates` | list over frames: `{frame, landmarks}` |
| `coordinates[i].landmarks` | 18 named joints → `{x, y, pixel_x, pixel_y, score}`; `x`/`y` are normalized to the frame size, `score` is the detector confidence |
| `model`, `device` | extraction backbone (`yolo26l-pose.pt`) and device |

With these files in place, the pipeline is reproducible from
`02_json_to_tensor.py` onward (only `01-1_extraction_golf.py` requires the
original videos).
