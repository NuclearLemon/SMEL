from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


def array_sha256(array: np.ndarray) -> str:
    value = np.ascontiguousarray(array)
    header = json.dumps({"shape": value.shape, "dtype": str(value.dtype)}, sort_keys=True)
    digest = hashlib.sha256(header.encode("ascii"))
    digest.update(memoryview(value).cast("B"))
    return digest.hexdigest()


def load_manifest(data_root: str | Path) -> dict:
    path = Path(data_root) / "reproduction_manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing {path}. Use the complete supplied data directory, including its segmentation guides.")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("format_version") != 1 or manifest.get("segmentation_backend") != "historical_python_ers_v1":
        raise ValueError(f"Unsupported reproduction manifest: {path}")
    if manifest.get("ers_lambda") != 0.5 or manifest.get("ers_sigma") != 5.0:
        raise ValueError(f"Historical partition parameters changed: {path}")
    return manifest


def check_dataset(dataset: str, cube: np.ndarray, gt: np.ndarray, data_root: str | Path) -> None:
    expected = load_manifest(data_root)["datasets"][dataset]
    if list(cube.shape) != expected["cube_shape"]:
        raise ValueError(f"Dataset shape does not match the released data: {dataset}")
    if array_sha256(cube) != expected["cube_sha256"] or array_sha256(gt) != expected["gt_sha256"]:
        raise ValueError(f"Dataset values differ from the verified release: {dataset}. Do not replace its packed archives when reproducing the reported results.")
