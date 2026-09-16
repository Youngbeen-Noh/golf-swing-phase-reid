"""Silhouette score and t-SNE seed-stability for the holdout embedding space (6.6).

t-SNE is used only as an auxiliary visualization (Figure 4b) -- never as
quantitative evidence -- but the instruction doc still asks for a
quantitative sanity check on that plot: does the embedding space actually
separate experts (silhouette score, computed in the real 128-d embedding
space, not the 2D projection), and is the 2D t-SNE layout itself stable
across seeds (a t-SNE run can look qualitatively different run to run).

Usage:
  python 17_tsne_stability_check.py --checkpoint "D:/AISL/experiments_archive/golf_experiment_2/ATCN_Golf-D-s42/models/atcn_step_sequence_best.pt" --features-dir data/Golf/experiments/baseline_Golf-D/features/user
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score

from encoder_loader import load_training_module


def load_sequence(path: Path) -> np.ndarray:
    with np.load(path, allow_pickle=False) as data:
        return data["sequence"].astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--features-dir", required=True, help="Holdout feature npz directory (per-expert subfolders).")
    parser.add_argument("--tsne-seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--perplexity", type=float, default=30.0)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    mod = load_training_module(ckpt["model_script"])
    model = mod.build_model_from_config(ckpt["model_config"]).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    files = sorted(Path(args.features_dir).glob("*/*.npz"))
    labels, embeddings = [], []
    with torch.no_grad():
        for f in files:
            seq = load_sequence(f)
            x = torch.from_numpy(seq).unsqueeze(0).to(device)  # (1, L, J, C)
            emb = model.encode_sequence(x).cpu().numpy()[0]
            embeddings.append(emb)
            labels.append(f.parent.name)

    X = np.stack(embeddings)
    y = np.array(labels)
    n_classes = len(set(labels))
    print(f"n_samples={len(y)}, n_classes={n_classes}")

    sil_highdim = silhouette_score(X, y, metric="cosine")
    print(f"silhouette score (128-d embedding space, cosine): {sil_highdim:.4f}")

    sil_tsne = []
    for seed in args.tsne_seeds:
        coords = TSNE(n_components=2, perplexity=min(args.perplexity, len(y) - 1),
                      random_state=seed, init="pca").fit_transform(X)
        s = silhouette_score(coords, y)
        sil_tsne.append(s)
        print(f"  t-SNE seed={seed}: silhouette (2D projection) = {s:.4f}")

    sil_tsne_arr = np.array(sil_tsne)
    print(f"t-SNE silhouette across {len(args.tsne_seeds)} seeds: "
          f"mean={sil_tsne_arr.mean():.4f} sd={sil_tsne_arr.std(ddof=1):.4f}")

    out = Path("results_tsne_stability.csv")
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value"])
        w.writerow(["n_samples", len(y)])
        w.writerow(["n_classes", n_classes])
        w.writerow(["silhouette_highdim_cosine", round(sil_highdim, 4)])
        for seed, s in zip(args.tsne_seeds, sil_tsne):
            w.writerow([f"silhouette_tsne_seed{seed}", round(s, 4)])
        w.writerow(["silhouette_tsne_mean", round(sil_tsne_arr.mean(), 4)])
        w.writerow(["silhouette_tsne_sd", round(sil_tsne_arr.std(ddof=1), 4)])
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
