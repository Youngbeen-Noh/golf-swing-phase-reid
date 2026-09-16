"""Dump sequence embeddings + a t-SNE 2D layout for Figure S3 (V10 revision).

Same loading/encoding path as 17_tsne_stability_check.py (which only PRINTS
its numbers and saves a summary CSV) -- this variant additionally saves:
  {out_prefix}_embeddings.npz   X (N x 128 float32), labels (N object)
  {out_prefix}_tsne_coords.csv  label,sample,x,y  (one t-SNE fit, fixed seed)
so the paper machine can draw Figure S3 (with anonymized expert IDs) without
needing torch/checkpoints locally. No algorithmic change: encode_sequence on
each features npz, cosine silhouette in the 128-d space, TSNE(init="pca").

Usage (server, repo root):
  python 29_dump_embeddings_tsne.py \
    --checkpoint experiments_archive/golf_experiment_2/ATCN_Golf-D-s42/models/atcn_step_sequence_best.pt \
    --features-dir experiments_archive/golf_experiment_2/ATCN_Golf-D/features/user \
    --out-prefix table11_recheck_out/figS3_validation
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
    parser.add_argument("--features-dir", required=True,
                        help="Feature npz directory (per-expert subfolders).")
    parser.add_argument("--tsne-seed", type=int, default=42)
    parser.add_argument("--perplexity", type=float, default=30.0)
    parser.add_argument("--out-prefix", required=True)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    mod = load_training_module(ckpt["model_script"])
    model = mod.build_model_from_config(ckpt["model_config"]).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    files = sorted(Path(args.features_dir).glob("*/*.npz"))
    if not files:
        raise SystemExit(f"no feature npz files under {args.features_dir}")
    labels, samples, embeddings = [], [], []
    with torch.no_grad():
        for f in files:
            seq = load_sequence(f)
            x = torch.from_numpy(seq).unsqueeze(0).to(device)  # (1, L, J, C)
            emb = model.encode_sequence(x).cpu().numpy()[0]
            embeddings.append(emb)
            labels.append(f.parent.name)
            samples.append(f.stem)

    X = np.stack(embeddings)
    y = np.array(labels)
    print(f"n_samples={len(y)}, n_classes={len(set(labels))}")

    sil = silhouette_score(X, y, metric="cosine")
    print(f"silhouette score (128-d embedding space, cosine): {sil:.4f}")

    coords = TSNE(n_components=2, perplexity=min(args.perplexity, len(y) - 1),
                  random_state=args.tsne_seed, init="pca").fit_transform(X)

    out_prefix = Path(args.out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    np.savez(f"{out_prefix}_embeddings.npz", X=X, labels=y,
             samples=np.array(samples), silhouette_cosine=np.array([sil]))
    with open(f"{out_prefix}_tsne_coords.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["label", "sample", "x", "y"])
        for lab, smp, (cx, cy) in zip(labels, samples, coords):
            w.writerow([lab, smp, f"{cx:.4f}", f"{cy:.4f}"])
    print(f"Saved: {out_prefix}_embeddings.npz / {out_prefix}_tsne_coords.csv")


if __name__ == "__main__":
    main()
