"""FLOPs/MACs, embedding inference latency, and pose extraction latency (6.4).

Complements 14_count_parameters.py (parameter counts). Uses thop to count
MACs for one forward pass of encode_sequence (L=160) and encode_steps
(S=7, Ls=20), and times actual forward passes (GPU if available, batch=1,
matching real single-query retrieval latency) with warmup iterations
discarded. Also times YOLO pose extraction (01-1_extraction_golf.py's model)
per-frame on one sample video. Local hardware differs from the training
server (see paper 5.4 for training GPU) -- report this as a relative
comparison across encoders on the SAME local machine, not an absolute
latency claim for the original training hardware.

Usage:
  python 16_efficiency_benchmark.py
  python 16_efficiency_benchmark.py --pose-video data/Golf/videos/expert/SOME_EXPERT/001.mp4
  python 16_efficiency_benchmark.py --skip-pose   # only encoder MACs/latency
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import torch
from thop import profile

from encoder_loader import load_training_module

J, C, L, LS, S = 18, 5, 160, 20, 7

CONFIGS = {
    "A-TCN": {
        "script": "04-1_train_atcn.py",
        "build": lambda mod: mod.build_model_from_config({
            "input_dim": J * C, "hidden_dim": 128, "embedding_dim": 128,
            "num_blocks": 4, "dropout": 0.2, "num_classes": 0,
        }),
    },
    "Transformer": {
        "script": "04-2_train_transformer.py",
        "build": lambda mod: mod.build_model_from_config({
            "input_dim": J * C, "model_dim": 64, "embedding_dim": 128,
            "num_layers": 1, "num_heads": 2, "feedforward_dim": 128,
            "dropout": 0.1, "num_classes": 0,
        }),
    },
    "ST-GCN": {
        "script": "04-3_train_stgcn.py",
        "build": lambda mod: mod.build_model_from_config({
            "input_channels": C,
            "joint_names": [
                "left_wrist", "right_wrist", "left_hip", "right_hip",
                "left_shoulder", "right_shoulder", "left_knee", "right_knee",
                "left_heel", "right_heel", "left_toe", "right_toe",
                "left_eye", "right_eye", "left_elbow", "right_elbow",
                "left_ear", "right_ear",
            ],
            "hidden_dim": 64, "embedding_dim": 64, "num_blocks": 2,
            "dropout": 0.1, "num_classes": 0,
        }),
    },
}


def make_inputs(model_name: str, device: torch.device):
    if model_name == "ST-GCN":
        seq = torch.randn(1, L, J, C, device=device)
        steps = torch.randn(1, S, LS, J, C, device=device)
    else:
        seq = torch.randn(1, L, J * C, device=device)
        steps = torch.randn(1, S, LS, J * C, device=device)
    return seq, steps


def time_forward(fn, n_warmup: int, n_iter: int, device: torch.device) -> tuple[float, float]:
    for _ in range(n_warmup):
        fn()
    if device.type == "cuda":
        torch.cuda.synchronize()
    times = []
    for _ in range(n_iter):
        start = time.perf_counter()
        fn()
        if device.type == "cuda":
            torch.cuda.synchronize()
        times.append(time.perf_counter() - start)
    times.sort()
    return sum(times) / len(times), times[len(times) // 2]


def benchmark_pose_extraction(video_path: str, device: str) -> tuple[float, float, int]:
    import cv2
    from ultralytics import YOLO

    model = YOLO("yolo26l-pose.pt", task="pose")
    model.to(device)

    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ok, img = cap.read()
        if not ok:
            break
        frames.append(img)
    cap.release()

    for img in frames[:5]:
        model.predict(img, device=device, verbose=False)

    times = []
    for img in frames:
        start = time.perf_counter()
        model.predict(img, device=device, verbose=False)
        times.append(time.perf_counter() - start)
    times.sort()
    return sum(times) / len(times), times[len(times) // 2], len(frames)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--pose-video", default="data/Golf/videos/expert/SOME_EXPERT/001.mp4")
    parser.add_argument("--skip-pose", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'cpu'})")
    rows = []

    for name, cfg in CONFIGS.items():
        mod = load_training_module(cfg["script"])
        model = cfg["build"](mod).to(device).eval()

        # encode_sequence/encode_steps always take (B, L, J, C) / (B, S, Ls, J, C) --
        # the internal reshape-to-flat happens inside those methods for A-TCN/Transformer.
        seq_input = torch.randn(1, L, J, C, device=device)
        step_input = torch.randn(1, S, LS, J, C, device=device)

        with torch.no_grad():
            seq_macs, _ = profile(model.sequence_encoder if name != "ST-GCN"
                                   else model.sequence_encoder,
                                   inputs=(seq_input if name == "ST-GCN" else seq_input.reshape(1, L, J * C),),
                                   verbose=False)
        # step encoder processes (B*S, Ls, ...) internally; profile it directly with that shape
        step_flat_shape = (S, LS, J, C) if name == "ST-GCN" else (S, LS, J * C)
        step_flat_input = torch.randn(*step_flat_shape, device=device)
        with torch.no_grad():
            step_macs, _ = profile(model.step_encoder, inputs=(step_flat_input,), verbose=False)

        def run_once():
            with torch.no_grad():
                model.encode_sequence(seq_input)
                model.encode_steps(step_input)

        mean_s, median_s = time_forward(run_once, n_warmup=20, n_iter=100, device=device)

        rows.append({
            "model": name,
            "sequence_encoder_MACs": int(seq_macs),
            "step_encoder_MACs_per_phase": int(step_macs),
            "step_encoder_MACs_total_7phases": int(step_macs) * S,
            "retrieval_MACs_total": int(seq_macs) + int(step_macs) * S,
            "embed_latency_mean_ms": round(mean_s * 1000, 3),
            "embed_latency_median_ms": round(median_s * 1000, 3),
            "device": str(device),
        })
        print(f"{name:<12} MACs(seq+7*step)={int(seq_macs)+int(step_macs)*S:>14,}  "
              f"latency mean={mean_s*1000:.3f}ms median={median_s*1000:.3f}ms")

    out = Path("results_efficiency_benchmark.csv")
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nSaved: {out}")

    if not args.skip_pose:
        mean_s, median_s, n_frames = benchmark_pose_extraction(args.pose_video, str(device))
        print(f"\npose extraction ({args.pose_video}, {n_frames} frames): "
              f"mean={mean_s*1000:.2f}ms median={median_s*1000:.2f}ms per frame "
              f"(~{1000/(mean_s*1000):.1f} fps achievable)")
        pose_out = Path("results_pose_extraction_latency.csv")
        with pose_out.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["video", "n_frames", "mean_ms", "median_ms", "device"])
            w.writerow([args.pose_video, n_frames, round(mean_s * 1000, 3), round(median_s * 1000, 3), str(device)])
        print(f"Saved: {pose_out}")


if __name__ == "__main__":
    main()
