from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    config_path = Path(path) if path else PROJECT_ROOT / "configs" / "experiments.json"
    return json.loads(config_path.read_text())


def write_json(path: str | Path, value: Any) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def torch_dtype(name: str) -> torch.dtype:
    try:
        return getattr(torch, name)
    except AttributeError as exc:
        raise ValueError(f"Unknown torch dtype: {name}") from exc


def baseline_cross_entropy(
    embeddings: torch.Tensor,
    classifier: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    return F.cross_entropy((embeddings @ classifier.T).float(), targets)


def tensor_error_metrics(
    reference: torch.Tensor,
    candidate: torch.Tensor,
    *,
    atol: float,
    rtol: float,
) -> dict[str, float | bool]:
    ref = reference.detach().float().flatten()
    cand = candidate.detach().float().flatten()
    diff = cand - ref
    ref_norm = torch.linalg.vector_norm(ref)
    diff_norm = torch.linalg.vector_norm(diff)
    denominator = max(float(ref_norm), torch.finfo(torch.float32).eps)
    cosine = 1.0
    if ref.numel() and float(ref_norm) > 0 and float(torch.linalg.vector_norm(cand)) > 0:
        cosine = float(F.cosine_similarity(ref, cand, dim=0))
    max_abs = float(diff.abs().max()) if diff.numel() else 0.0
    mean_abs = float(diff.abs().mean()) if diff.numel() else 0.0
    return {
        "max_abs_error": max_abs,
        "mean_abs_error": mean_abs,
        "relative_l2_error": float(diff_norm) / denominator,
        "cosine_similarity": cosine,
        "passes_loose_tolerance": bool(
            torch.allclose(ref, cand, atol=atol, rtol=rtol)
        ),
        "atol": atol,
        "rtol": rtol,
    }


def scalar_error(reference: float, candidate: float) -> dict[str, float]:
    absolute = abs(candidate - reference)
    return {
        "absolute_error": absolute,
        "relative_error": absolute / max(abs(reference), math.ulp(1.0)),
    }
