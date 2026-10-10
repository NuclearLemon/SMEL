from __future__ import annotations

import heapq
import math
from pathlib import Path

import numpy as np

try:
    from .reproducibility import array_sha256, load_manifest
except ImportError:
    from reproducibility import array_sha256, load_manifest

EPS = 1e-12
SQRT2 = math.sqrt(2.0)
BACKEND = "historical_python_ers_v1"


def _relabel_segments(segments: np.ndarray, start_label: int = 0) -> np.ndarray:
    segments = np.asarray(segments, dtype=np.int64)
    uniq = np.unique(segments)
    out = np.empty_like(segments)
    for new_id, old_id in enumerate(uniq.tolist(), start=start_label):
        out[segments == old_id] = new_id
    return out


def _xlog2x(x: float) -> float:
    return 0.0 if x <= 0.0 else x * math.log2(x)


def _entropy_rate_gain(w: float, ci: float, cj: float) -> float:
    return (
        _xlog2x(w + ci)
        + _xlog2x(w + cj)
        - _xlog2x(ci)
        - _xlog2x(cj)
        - 2.0 * _xlog2x(w)
    )


def _balancing_gain(n_vertices: int, si: int, sj: int) -> float:
    Si = si / float(n_vertices)
    Sj = sj / float(n_vertices)
    return -_xlog2x(Si + Sj) + _xlog2x(Si) + _xlog2x(Sj) + 1.0


class _UnionFind:
    def __init__(self, n: int):
        self.parent = np.arange(n, dtype=np.int64)
        self.size = np.ones(n, dtype=np.int64)
        self.n_components = n

    def find(self, x: int) -> int:
        parent = self.parent
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(self, a: int, b: int) -> tuple[int, int]:
        a = self.find(a)
        b = self.find(b)
        if a == b:
            return a, b
        if self.size[a] < self.size[b]:
            a, b = b, a
        self.parent[b] = a
        self.size[a] += self.size[b]
        self.size[b] = 0
        self.n_components -= 1
        return a, b


def _build_8n_edges(gray: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    gray = np.asarray(gray, dtype=np.float64)
    h, w = gray.shape
    idx = np.arange(h * w, dtype=np.int64).reshape(h, w)
    a_list, b_list, d_list = [], [], []
    a_list.append(idx[:, :-1].reshape(-1))
    b_list.append(idx[:, 1:].reshape(-1))
    d_list.append(np.abs(gray[:, :-1] - gray[:, 1:]).reshape(-1))
    a_list.append(idx[:-1, :].reshape(-1))
    b_list.append(idx[1:, :].reshape(-1))
    d_list.append(np.abs(gray[:-1, :] - gray[1:, :]).reshape(-1))
    a_list.append(idx[:-1, :-1].reshape(-1))
    b_list.append(idx[1:, 1:].reshape(-1))
    d_list.append((SQRT2 * np.abs(gray[:-1, :-1] - gray[1:, 1:])).reshape(-1))
    a_list.append(idx[1:, :-1].reshape(-1))
    b_list.append(idx[:-1, 1:].reshape(-1))
    d_list.append((SQRT2 * np.abs(gray[1:, :-1] - gray[:-1, 1:])).reshape(-1))
    a = np.concatenate(a_list)
    b = np.concatenate(b_list)
    d = np.concatenate(d_list).astype(np.float64, copy=False)
    return a, b, d


def ers_segment(gray: np.ndarray, n_segments: int, lam: float = 0.5, sigma: float = 5.0) -> np.ndarray:
    """Python ERS approximation on an eight-neighbor image graph."""
    gray = np.asarray(gray)
    if gray.ndim != 2:
        raise ValueError(f"ERS expects a single-channel image, got {gray.shape}")
    h, w = gray.shape
    n_vertices = h * w
    n_segments = int(n_segments)
    if n_segments <= 0:
        raise ValueError("n_segments must be positive")
    if n_segments >= n_vertices:
        return np.arange(n_vertices, dtype=np.int64).reshape(h, w)
    a, b, diff = _build_8n_edges(gray.astype(np.float64, copy=False))
    edge_w = np.exp(-(diff * diff) / max(2.0 * sigma * sigma, EPS))
    degree = np.bincount(a, weights=edge_w, minlength=n_vertices) + np.bincount(b, weights=edge_w, minlength=n_vertices)
    total_degree = float(degree.sum())
    edge_w = edge_w / max(total_degree, EPS)
    degree = degree / max(total_degree, EPS)
    uf = _UnionFind(n_vertices)
    comp_degree = degree.copy()
    init_b = _balancing_gain(n_vertices, 1, 1)
    max_er = 0.0
    init_er = np.empty_like(edge_w)
    for i in range(edge_w.shape[0]):
        w_ij = float(edge_w[i])
        gi = max(float(comp_degree[a[i]]) - w_ij, 0.0)
        gj = max(float(comp_degree[b[i]]) - w_ij, 0.0)
        g = _entropy_rate_gain(w_ij, gi, gj)
        init_er[i] = g
        if g > max_er:
            max_er = g
    balancing = (lam * n_segments) * max_er / max(abs(init_b), EPS)
    init_gain = init_er + balancing * init_b
    heap = [(-float(init_gain[i]), int(i)) for i in range(init_gain.shape[0])]
    heapq.heapify(heap)
    while uf.n_components > n_segments:
        neg_gain, idx = heapq.heappop(heap)
        u = int(a[idx])
        v = int(b[idx])
        w_uv = float(edge_w[idx])
        ru = uf.find(u)
        rv = uf.find(v)
        if ru == rv:
            continue
        deg_ru = float(comp_degree[ru])
        deg_rv = float(comp_degree[rv])
        cur_er = _entropy_rate_gain(w_uv, max(deg_ru - w_uv, 0.0), max(deg_rv - w_uv, 0.0))
        cur_b = _balancing_gain(n_vertices, int(uf.size[ru]), int(uf.size[rv]))
        cur_gain = cur_er + balancing * cur_b
        old_gain = -float(neg_gain)
        if old_gain - cur_gain > 1e-15:
            heapq.heappush(heap, (-cur_gain, idx))
            continue
        old_deg_ru = deg_ru
        old_deg_rv = deg_rv
        new_root, old_root = uf.union(ru, rv)
        new_deg = max(old_deg_ru + old_deg_rv - 2.0 * w_uv, 0.0)
        comp_degree[new_root] = new_deg
        comp_degree[old_root] = 0.0
    roots = np.empty(n_vertices, dtype=np.int64)
    for i in range(n_vertices):
        roots[i] = uf.find(i)
    return _relabel_segments(roots.reshape(h, w), start_label=0)


def build_segmentation(cube: np.ndarray, n_segments: int, cache_path: Path | None = None, *, guide_path: Path) -> np.ndarray:
    guide_path = Path(guide_path)
    dataset = guide_path.parent.name
    manifest = load_manifest(guide_path.parent.parent)
    expected = manifest["datasets"][dataset]
    if not guide_path.is_file():
        raise FileNotFoundError(f"Missing {guide_path}. The grayscale guide is a supplied data input, not a disposable partition cache.")
    with np.load(guide_path, allow_pickle=False) as data:
        gray = data["gray"]
    if gray.dtype != np.uint8 or gray.shape != cube.shape[:2] or array_sha256(gray) != expected["guide_sha256"]:
        raise ValueError(f"Segmentation guide does not match the released data: {dataset}")
    count = int(n_segments)
    if count <= 0 or count > gray.size:
        raise ValueError("Superpixel count must be between 1 and the number of pixels")
    cache_path = Path(cache_path) if cache_path is not None else None
    if cache_path is not None and cache_path.exists():
        with np.load(cache_path, allow_pickle=False) as data:
            if not {"seg", "guide_sha256", "n_segments", "backend", "seg_sha256"}.issubset(data.files):
                raise ValueError(f"Unverified partition cache: {cache_path}")
            if str(data["backend"].item()) != BACKEND or str(data["guide_sha256"].item()) != expected["guide_sha256"] or int(data["n_segments"].item()) != count:
                raise ValueError(f"Partition cache belongs to a different input or backend: {cache_path}")
            seg = data["seg"]
            if array_sha256(seg) != str(data["seg_sha256"].item()):
                raise ValueError(f"Partition cache checksum failed: {cache_path}")
    else:
        print(f"  [SP] computing Python ERS approximation: {dataset}, segments={count}", flush=True)
        seg = ers_segment(gray, count, lam=float(manifest["ers_lambda"]), sigma=float(manifest["ers_sigma"])).astype(np.int32)
    if seg.dtype != np.int32 or seg.shape != gray.shape or np.unique(seg).size != count:
        raise ValueError(f"Invalid partition for {dataset}")
    checksum = array_sha256(seg)
    if count == int(expected["default_segments"]) and checksum != expected["default_segmentation_sha256"]:
        raise ValueError(f"Rebuilt partition differs from the verified experiment: {dataset}. No alternate backend will be used.")
    if cache_path is not None and not cache_path.exists():
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache_path, seg=seg, guide_sha256=expected["guide_sha256"], n_segments=count, backend=BACKEND, seg_sha256=checksum)
    return seg
