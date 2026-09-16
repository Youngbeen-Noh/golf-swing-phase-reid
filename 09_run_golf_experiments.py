from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path


GOLF_METHODS = ["Golf-A", "Golf-B", "Golf-C", "Golf-D", "Golf-E"]

MODEL_SCRIPTS = {
    "ATCN": "04-1_train_atcn.py",
    "Transformer": "04-2_train_transformer.py",
    "STGCN": "04-3_train_stgcn.py",
}

PIPELINE_STEPS = ["03", "04", "05", "06", "07", "08"]


def unique_preserve_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for item in items:
        if item not in seen:
            output.append(item)
            seen.add(item)
    return output


def selected_steps(start_at: str, end_at: str) -> set[str]:
    start_idx = PIPELINE_STEPS.index(start_at)
    end_idx = PIPELINE_STEPS.index(end_at)
    if start_idx > end_idx:
        raise ValueError("--start-at must be earlier than or equal to --end-at")
    return set(PIPELINE_STEPS[start_idx : end_idx + 1])


def command_to_string(command: list[str]) -> str:
    return " ".join(f'"{part}"' if " " in part else part for part in command)


def run_command(command: list[str], cwd: Path, dry_run: bool) -> None:
    print(f"\n$ {command_to_string(command)}")
    if dry_run:
        return
    subprocess.run(command, cwd=str(cwd), check=True)


def read_metric_score(root: Path, run_id: str) -> tuple[float, float] | None:
    """Return (top1_accuracy, mrr) for one run_id, used to pick the best method per model.

    Prefers the known-pro holdout evaluation (the actual "identify this pro as a user"
    task) and falls back to the plain expert-retrieval metrics if holdout reports
    aren't available.
    """
    eval_dir = root / "data" / "Golf" / "experiments" / run_id / "results" / "evaluation"

    pro_holdout_path = eval_dir / "pro_holdout_user_metrics.json"
    if pro_holdout_path.exists():
        data = json.loads(pro_holdout_path.read_text(encoding="utf-8"))
        final = data.get("summary", {}).get("final_ranking", {})
        if final.get("count", 0):
            return float(final.get("top1_accuracy", 0.0)), float(final.get("mrr", 0.0))

    fallback_path = eval_dir / "expert_retrieval_metrics.json"
    if fallback_path.exists():
        data = json.loads(fallback_path.read_text(encoding="utf-8"))
        metrics = data.get("metrics", {})
        return float(metrics.get("top1_accuracy", 0.0)), float(metrics.get("MRR", 0.0))

    return None


def build_commands(
    python_exe: str,
    root: Path,
    model_name: str,
    method: str,
    args: argparse.Namespace,
    include_visualize: bool = True,
) -> list[tuple[str, list[str]]]:
    suffix = getattr(args, "run_id_suffix", "")
    base_run_id = f"{model_name}_{method}"
    run_id = f"{base_run_id}{suffix}"
    commands: list[tuple[str, list[str]]] = []

    if not suffix:
        preprocess = [
            python_exe,
            "03_preprocess_and_build_features.py",
            "--root",
            str(root),
            "--sport",
            "Golf",
            "--preprocess-method",
            method,
            "--run-id",
            run_id,
        ]
        if args.overwrite_features:
            preprocess.append("--overwrite")
        if args.max_samples_per_expert > 0:
            preprocess.extend(["--max-samples-per-expert", str(args.max_samples_per_expert)])
        if getattr(args, "expert_ids", None):
            preprocess.extend(["--expert-ids", args.expert_ids])
        if getattr(args, "no_finish_hold", False):
            preprocess.append("--no-finish-hold")
        commands.append(("03", preprocess))

    train = [
        python_exe,
        MODEL_SCRIPTS[model_name],
        "--root",
        str(root),
        "--sport",
        "Golf",
        "--run-id",
        run_id,
    ]
    if suffix:
        base_feature_dir = root / "data" / "Golf" / "experiments" / base_run_id / "features" / "expert"
        train.extend(["--feature-dir", str(base_feature_dir)])
    if args.epochs is not None:
        train.extend(["--epochs", str(args.epochs)])
    if args.batch_size is not None:
        train.extend(["--batch-size", str(args.batch_size)])
    for extra_args in args.train_extra:
        train.extend(shlex.split(extra_args))
    commands.append(("04", train))

    build_expert_prototypes = [
        python_exe,
        "05_build_expert_prototypes.py",
        "--root",
        str(root),
        "--sport",
        "Golf",
        "--run-id",
        run_id,
    ]
    if suffix:
        base_expert_feature_dir = root / "data" / "Golf" / "experiments" / base_run_id / "features" / "expert"
        build_expert_prototypes.extend(["--feature-dir", str(base_expert_feature_dir)])
    commands.append(("05", build_expert_prototypes))

    infer_user_similarity = [
        python_exe,
        "06_infer_user_similarity.py",
        "--root",
        str(root),
        "--sport",
        "Golf",
        "--run-id",
        run_id,
    ]
    if suffix:
        base_user_feature_dir = root / "data" / "Golf" / "experiments" / base_run_id / "features" / "user"
        infer_user_similarity.extend(["--feature-dir", str(base_user_feature_dir)])
    if getattr(args, "alpha", None) is not None:
        infer_user_similarity.extend(["--alpha", str(args.alpha)])
    commands.append(("06", infer_user_similarity))
    evaluate = [
        python_exe,
        "07_evaluate.py",
        "--root",
        str(root),
        "--sport",
        "Golf",
        "--run-id",
        run_id,
    ]
    if suffix:
        evaluate.extend(["--feature-dir", str(base_expert_feature_dir)])
    commands.append(("07", evaluate))

    if include_visualize:
        commands.append(("08", build_visualize_command(python_exe, root, run_id, args)))
    return commands


def build_visualize_command(python_exe: str, root: Path, run_id: str, args: argparse.Namespace) -> list[str]:
    visualize = [
        python_exe,
        "08_visualize_user_similarity.py",
        "--root",
        str(root),
        "--sport",
        "Golf",
        "--run-id",
        run_id,
    ]
    if args.overwrite_visualization:
        visualize.append("--overwrite")
    if args.skip_tsne:
        visualize.append("--skip-tsne")
    return visualize


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Golf-A~E experiments through steps 03~08 for selected model families."
    )
    parser.add_argument("--root", default=str(Path(__file__).resolve().parent))

    model_group = parser.add_argument_group("model selection")
    model_group.add_argument("--ATCN", "--atcn", dest="models", action="append_const", const="ATCN")
    model_group.add_argument(
        "--Transformer",
        "--transformer",
        dest="models",
        action="append_const",
        const="Transformer",
    )
    model_group.add_argument("--STGCN", "--stgcn", dest="models", action="append_const", const="STGCN")
    model_group.add_argument(
        "--all-models",
        action="store_true",
        help="Run ATCN, Transformer, and STGCN.",
    )

    parser.add_argument(
        "--methods",
        nargs="+",
        default=GOLF_METHODS,
        choices=GOLF_METHODS,
        help="Golf preprocessing presets to run. Default: Golf-A Golf-B Golf-C Golf-D Golf-E.",
    )
    parser.add_argument("--start-at", choices=PIPELINE_STEPS, default="03")
    parser.add_argument("--end-at", choices=PIPELINE_STEPS, default="08")
    parser.add_argument("--epochs", type=int, default=None, help="Optional epoch override passed to step 04.")
    parser.add_argument("--batch-size", type=int, default=None, help="Optional batch-size override passed to step 04.")
    parser.add_argument(
        "--train-extra",
        action="append",
        default=[],
        help=(
            "Extra arguments appended to each training command as a quoted string. "
            'Can be repeated, e.g. --train-extra "--lr 0.0003 --dropout 0.0"'
        ),
    )
    parser.add_argument(
        "--max-samples-per-expert",
        type=int,
        default=0,
        help="Cap samples per expert passed to step 03 (0 = no limit).",
    )
    parser.add_argument(
        "--no-finish-hold",
        action="store_true",
        help="Pass --no-finish-hold through to step 03 (P0-4b finish-hold ablation, paper 7.2).",
    )
    parser.add_argument(
        "--expert-ids",
        default=None,
        help=(
            "Comma-separated expert ids passed to step 03 to restrict this run to a subset "
            "of the shared tensor pool (both expert and holdout/user data). "
            "Default: all experts."
        ),
    )
    parser.add_argument(
        "--no-overwrite-features",
        action="store_false",
        dest="overwrite_features",
        help="Do not pass --overwrite to step 03. Default is to regenerate features for repeatable runs.",
    )
    parser.add_argument(
        "--no-overwrite-visualization",
        action="store_false",
        dest="overwrite_visualization",
        help="Do not pass --overwrite to step 08. Default is to regenerate visualizations.",
    )
    parser.add_argument("--skip-tsne", action="store_true", help="Pass --skip-tsne to step 08.")
    parser.add_argument(
        "--visualize-best-only",
        action="store_true",
        help=(
            "Run step 08 only for the best-performing method per model (ATCN/Transformer/STGCN), "
            "instead of every model x method combination. Steps 03-07 still run for all combinations "
            "since evaluation metrics are needed to pick the best one. Ranked by known-pro holdout "
            "top1 accuracy, falling back to plain expert-retrieval top1 accuracy."
        ),
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=None,
        help="Alpha passed to step 06 (sequence weight). Default 0.5. Use 0.0 for step-only, 1.0 for sequence-only inference.",
    )
    parser.add_argument(
        "--run-id-suffix",
        default="",
        help=(
            "Suffix appended to every run_id (e.g. '-seq0'). "
            "Step 03 is skipped and features are reused from the base run_id. "
            "Use with --train-extra \"--lambda-step 0\" to run sequence-only training."
        ),
    )
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running them.")
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue with the next experiment when one command fails.",
    )
    parser.set_defaults(overwrite_features=True, overwrite_visualization=True)
    args = parser.parse_args()

    selected_models = []
    if args.all_models:
        selected_models.extend(MODEL_SCRIPTS)
    if args.models:
        selected_models.extend(args.models)
    args.models = unique_preserve_order(selected_models)
    if not args.models:
        parser.error("Select at least one model with --ATCN, --Transformer, --STGCN, or --all-models.")
    return args


def main() -> None:
    args = parse_args()
    root = Path(args.root).resolve()
    active_steps = selected_steps(args.start_at, args.end_at)
    python_exe = sys.executable

    print("Golf experiment runner")
    print(f"root={root}")
    print(f"models={args.models}")
    print(f"methods={args.methods}")
    print(f"steps={args.start_at}~{args.end_at}")
    if args.run_id_suffix:
        print(f"run_id_suffix={args.run_id_suffix!r}  (step 03 skipped, features reused from base run)")

    defer_visualize = args.visualize_best_only and "08" in active_steps
    if defer_visualize:
        print("visualize_best_only=True  (step 08 deferred until the best method per model is picked)")

    failures: list[tuple[str, str, str]] = []
    run_ids_by_model: dict[str, list[str]] = {model_name: [] for model_name in args.models}
    for model_name in args.models:
        for method in args.methods:
            run_id = f"{model_name}_{method}"
            actual_run_id = f"{run_id}{args.run_id_suffix}"
            run_ids_by_model[model_name].append(actual_run_id)
            print("\n" + "=" * 80)
            print(f"run_id={run_id}")
            print("=" * 80)
            commands = build_commands(python_exe, root, model_name, method, args, include_visualize=not defer_visualize)
            for step, command in commands:
                if step not in active_steps:
                    continue
                try:
                    run_command(command, cwd=root, dry_run=args.dry_run)
                except subprocess.CalledProcessError as exc:
                    failures.append((run_id, step, str(exc.returncode)))
                    print(f"[error] run_id={run_id}, step={step}, returncode={exc.returncode}")
                    if not args.continue_on_error:
                        raise
                    break

    if defer_visualize and not args.dry_run:
        print("\n" + "=" * 80)
        print("Selecting best method per model for visualization (step 08)")
        print("=" * 80)
        for model_name, candidate_run_ids in run_ids_by_model.items():
            best_run_id = None
            best_score: tuple[float, float] | None = None
            for candidate_run_id in candidate_run_ids:
                score = read_metric_score(root, candidate_run_id)
                if score is None:
                    print(f"[warn] No evaluation metrics found for run_id={candidate_run_id}; skipping it.")
                    continue
                if best_score is None or score > best_score:
                    best_score = score
                    best_run_id = candidate_run_id
            if best_run_id is None:
                print(f"[warn] No evaluation metrics found for model={model_name}; skipping visualization.")
                continue
            top1, mrr = best_score
            print(f"[best] {model_name}: run_id={best_run_id} top1_accuracy={top1:.4f} mrr={mrr:.4f}")
            visualize = build_visualize_command(python_exe, root, best_run_id, args)
            try:
                run_command(visualize, cwd=root, dry_run=args.dry_run)
            except subprocess.CalledProcessError as exc:
                failures.append((best_run_id, "08", str(exc.returncode)))
                print(f"[error] run_id={best_run_id}, step=08, returncode={exc.returncode}")
                if not args.continue_on_error:
                    raise

    if failures:
        print("\nFailures:")
        for run_id, step, returncode in failures:
            print(f"- run_id={run_id}, step={step}, returncode={returncode}")
        raise SystemExit(1)

    print("\nDone.")


if __name__ == "__main__":
    main()
