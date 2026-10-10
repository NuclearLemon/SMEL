from __future__ import annotations
import gzip
import io
from pathlib import Path
from typing import Iterable
import numpy as np

try:
    from .reproducibility import check_dataset
except ImportError:
    from reproducibility import check_dataset


MAT_SPECS = {
    "PaviaUniversity": ("PaviaU.mat", "PaviaU_gt.mat", "paviaU", "paviaU_gt"),
    "PaviaCenter": ("Pavia.mat", "Pavia_gt.mat", "pavia", "pavia_gt"),
    "Botswana": ("Botswana.mat", "Botswana_gt.mat", "Botswana", "Botswana_gt"),
    "Trento": ("Trento.mat", "Trento_gt.mat", "HSI_Trento", "GT_Trento"),
    "LongKou": ("LongKou.mat", "LongKou_gt.mat", "data", "groundT"),
}


def resolve_dataset(name: str) -> str:
    name = str(name).strip()
    if name not in MAT_SPECS:
        raise KeyError(f"Unsupported dataset {name!r}. Available: {', '.join(MAT_SPECS)}")
    return name


def parse_datasets(text: str) -> list[str]:
    out: list[str] = []
    for token in str(text).split(","):
        token = token.strip()
        if not token:
            continue
        name = resolve_dataset(token)
        if name not in out:
            out.append(name)
    return out


def list_datasets() -> Iterable[str]:
    return tuple(MAT_SPECS.keys())


def _load_mat(path: Path, key: str) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(path)
    try:
        import scipy.io as sio
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("Reading .mat datasets requires scipy. Install project requirements or pack from an environment with scipy.") from exc
    mat = sio.loadmat(path)
    if key not in mat:
        keys = [k for k in mat if not k.startswith("__")]
        raise KeyError(f"Key {key!r} not found in {path}. Available keys: {keys}")
    return mat[key]


def _read_split_bytes(path: Path) -> bytes:
    parts = sorted(path.parent.glob(path.name + ".part*"))
    if not parts:
        raise FileNotFoundError(path)
    return b"".join(p.read_bytes() for p in parts)


def _load_npz_archive(path: Path) -> tuple[np.ndarray, np.ndarray, Path] | None:
    candidates = [path / "dataset.npz", path.with_suffix(".npz"), path / (path.name + ".npz")]
    candidates += [p.with_suffix(p.suffix + ".gz") for p in candidates]
    for p in candidates:
        if p.exists() or list(p.parent.glob(p.name + ".part*")):
            if p.suffix == ".gz":
                print(f"  [DATA] loading packed archive: {p}", flush=True)
                raw = p.read_bytes() if p.exists() else _read_split_bytes(p)
                payload = gzip.decompress(raw)
                data = np.load(io.BytesIO(payload), allow_pickle=False)
            else:
                print(f"  [DATA] loading packed archive: {p}", flush=True)
                data = np.load(p, allow_pickle=False)
            return np.asarray(data["cube"], dtype=np.float32), np.asarray(data["gt"], dtype=np.int32), p
    return None


def load_hsi(dataset: str, data_root: str | Path) -> tuple[np.ndarray, np.ndarray]:
    name = resolve_dataset(dataset)
    root = Path(data_root)
    data_file, gt_file, data_key, gt_key = MAT_SPECS[name]

    print(f"[DATA] preparing {name} from {root}", flush=True)
    packed = _load_npz_archive(root / name)
    if packed is not None:
        cube, gt, source = packed
        source_desc = str(source)
    else:
        print(f"  [DATA] loading MATLAB files: {root / data_file}, {root / gt_file}", flush=True)
        cube = _load_mat(root / data_file, data_key)
        gt = _load_mat(root / gt_file, gt_key)
        source_desc = f"{root / data_file}, {root / gt_file}"

    cube = np.asarray(cube, dtype=np.float32)
    gt = np.asarray(gt, dtype=np.int32)
    if cube.ndim != 3 or gt.ndim != 2 or cube.shape[:2] != gt.shape:
        raise ValueError(f"Bad dataset shapes for {name}: cube={cube.shape}, gt={gt.shape}")
    check_dataset(name, cube, gt, root)
    labels = np.unique(gt[gt > 0])
    print(
        f"  [DATA] loaded {name}: cube={cube.shape} gt={gt.shape} classes={labels.size} "
        f"cube_dtype={cube.dtype} gt_dtype={gt.dtype} source={source_desc}",
        flush=True,
    )
    return cube, gt


def n_classes(gt: np.ndarray) -> int:
    labels = np.unique(np.asarray(gt)[np.asarray(gt) > 0])
    return int(labels.size)
