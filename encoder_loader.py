from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import torch


MODEL_SCRIPT_BY_TYPE = {
    "atcn": "04-1_train_atcn.py",
    "a-tcn": "04-1_train_atcn.py",
    "transformer": "04-2_train_transformer.py",
    "transformer_encoder": "04-2_train_transformer.py",
    "stgcn": "04-3_train_stgcn.py",
    "st-gcn": "04-3_train_stgcn.py",
}

BEST_CHECKPOINT_PATTERNS = [
    "*_step_sequence_best.pt",
    "atcn_step_sequence_best.pt",
    "transformer_step_sequence_best.pt",
    "stgcn_step_sequence_best.pt",
]


def checkpoint_model_type(checkpoint: dict[str, Any]) -> str:
    """Return checkpoint model type, defaulting old checkpoints to atcn."""
    model_type = checkpoint.get("model_type")
    if model_type:
        return str(model_type).lower()
    train_args = checkpoint.get("train_args", {})
    if isinstance(train_args, dict) and train_args.get("model_type"):
        return str(train_args["model_type"]).lower()
    return "atcn"


def training_script_for_checkpoint(checkpoint: dict[str, Any]) -> str:
    model_type = checkpoint_model_type(checkpoint)
    if checkpoint.get("model_script"):
        return str(checkpoint["model_script"])
    if model_type not in MODEL_SCRIPT_BY_TYPE:
        raise ValueError(f"Unsupported checkpoint model_type={model_type}")
    return MODEL_SCRIPT_BY_TYPE[model_type]


def load_training_module(script_name: str) -> Any:
    module_path = Path(__file__).resolve().with_name(script_name)
    if not module_path.exists():
        raise FileNotFoundError(f"Training module not found: {module_path}")
    module_name = module_path.stem.replace("-", "_")
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load training module: {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def build_model_from_checkpoint(checkpoint: dict[str, Any], device: torch.device) -> Any:
    """Instantiate the correct encoder architecture from a checkpoint."""
    module = load_training_module(training_script_for_checkpoint(checkpoint))
    if hasattr(module, "build_model_from_checkpoint"):
        model = module.build_model_from_checkpoint(checkpoint, device)
    else:
        model_config = checkpoint["model_config"]
        model = module.StepSequenceATCN(
            input_dim=int(model_config["input_dim"]),
            hidden_dim=int(model_config["hidden_dim"]),
            embedding_dim=int(model_config["embedding_dim"]),
            num_blocks=int(model_config["num_blocks"]),
            dropout=float(model_config["dropout"]),
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        model.eval()
    return model


def default_checkpoint_path(models_dir: Path) -> Path | None:
    """Pick the newest available best checkpoint in one experiment models directory."""
    if not models_dir.exists():
        return None

    candidates: dict[Path, None] = {}
    for pattern in BEST_CHECKPOINT_PATTERNS:
        for path in models_dir.glob(pattern):
            if path.is_file():
                candidates[path] = None
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)
