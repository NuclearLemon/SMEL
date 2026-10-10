from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from scipy.signal import savgol_filter
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, cohen_kappa_score, normalized_mutual_info_score
from sklearn.preprocessing import StandardScaler

try:
    from .superpixels import build_segmentation
except ImportError:
    from superpixels import build_segmentation


VIEW_NAMES = ("pool_d1_16", "mix_pca_pool8_d1pool8_sp8", "pxnode_pool_d1_16")


def set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def standardize(x: np.ndarray) -> np.ndarray:
    return StandardScaler().fit_transform(np.asarray(x, dtype=np.float32)).astype(np.float32)


def evaluate(gt_flat: np.ndarray, pred: np.ndarray, prefix: str = "") -> Dict[str, float]:
    gt_flat = np.asarray(gt_flat).reshape(-1).astype(np.int64)
    pred = np.asarray(pred).reshape(-1).astype(np.int64)
    mask = gt_flat > 0
    y_true_raw = gt_flat[mask]
    y_pred = pred[mask]
    raw = np.unique(y_true_raw)
    remap = {v: i for i, v in enumerate(raw.tolist())}
    y_true = np.array([remap[x] for x in y_true_raw], dtype=np.int64)

    pred_labels = np.unique(y_pred)
    true_labels = np.unique(y_true)
    mat = np.zeros((len(pred_labels), len(true_labels)), dtype=np.int64)
    for i, p in enumerate(pred_labels):
        for j, t in enumerate(true_labels):
            mat[i, j] = np.sum((y_pred == p) & (y_true == t))
    ri, ci = linear_sum_assignment(mat.max() - mat)
    mapping = {pred_labels[r]: true_labels[c] for r, c in zip(ri, ci)}
    aligned = np.array([mapping.get(x, x) for x in y_pred], dtype=np.int64)

    out = {
        "ACC": float(np.mean(aligned == y_true)),
        "Kappa": float(cohen_kappa_score(y_true, aligned)),
        "NMI": float(normalized_mutual_info_score(y_true, y_pred)),
        "ARI": float(adjusted_rand_score(y_true, y_pred)),
    }
    return {f"{prefix}{k}": v for k, v in out.items()}


def usage_entropy(labels: np.ndarray, k: int) -> float:
    count = np.bincount(np.asarray(labels).reshape(-1), minlength=int(k)).astype(np.float64)
    p = count / max(1.0, float(count.sum()))
    p = p[p > 0]
    return float(-(p * np.log(p)).sum() / max(np.log(float(k)), 1e-8))


def relabel(seg: np.ndarray) -> np.ndarray:
    _, inv = np.unique(np.asarray(seg), return_inverse=True)
    return inv.reshape(seg.shape).astype(np.int32)


def spectral_pool(x: np.ndarray, groups: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    groups = max(1, min(int(groups), x.shape[1]))
    splits = np.array_split(np.arange(x.shape[1]), groups)
    pooled = [x[:, idx].mean(axis=1, keepdims=True).astype(np.float32) for idx in splits if len(idx)]
    return np.concatenate(pooled, axis=1).astype(np.float32)


def spectral_derivative(x: np.ndarray, order: int = 1, window: int = 11, poly: int = 2) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    c = x.shape[1]
    win = min(int(window), c if c % 2 == 1 else c - 1)
    win = max(win, int(poly) + 2)
    if win % 2 == 0:
        win += 1
    if win > c:
        win = c if c % 2 == 1 else c - 1
    win = max(win, 3 if c >= 3 else c)
    try:
        return savgol_filter(
            x,
            window_length=win,
            polyorder=int(poly),
            deriv=int(order),
            axis=1,
            mode="interp",
        ).astype(np.float32)
    except Exception:
        out = x.copy()
        for _ in range(int(order)):
            out = np.gradient(out, axis=1, edge_order=1)
        return out.astype(np.float32)


def sp_mean(x: np.ndarray, seg_flat: np.ndarray) -> np.ndarray:
    seg = np.asarray(seg_flat, dtype=np.int32).reshape(-1)
    out = np.zeros_like(x, dtype=np.float32)
    for sid in range(int(seg.max()) + 1):
        idx = np.where(seg == sid)[0]
        if idx.size:
            out[idx] = x[idx].mean(axis=0, keepdims=True)
    return out


def build_views(
    cube: np.ndarray,
    seg_flat: np.ndarray,
    pca_dim: int = 20,
    coarse_groups: int = 8,
    fine_groups: int = 16,
) -> tuple[list[np.ndarray], np.ndarray]:
    flat = standardize(cube.reshape(-1, cube.shape[-1]))
    d1 = spectral_derivative(flat, order=1, window=11, poly=2)
    pca_dim = max(1, min(int(pca_dim), flat.shape[1]))
    pca = PCA(n_components=pca_dim, random_state=0).fit_transform(flat).astype(np.float32)

    coarse_groups = max(1, int(coarse_groups))
    fine_groups = max(coarse_groups, int(fine_groups))
    pool_coarse = spectral_pool(flat, coarse_groups)
    pool_fine = spectral_pool(flat, fine_groups)
    d1pool_coarse = spectral_pool(d1, coarse_groups)
    d1pool_fine = spectral_pool(d1, fine_groups)
    sp_pool_coarse = sp_mean(pool_coarse, seg_flat)

    view1 = np.concatenate([pool_fine, d1pool_fine], axis=1).astype(np.float32)
    view2 = np.concatenate([pca, pool_coarse, d1pool_coarse, sp_pool_coarse], axis=1).astype(np.float32)
    view3 = np.concatenate([view1, sp_mean(view1, seg_flat)], axis=1).astype(np.float32)

    views = [standardize(v) for v in (view1, view2, view3)]
    teacher_x = standardize(np.concatenate(views, axis=1))
    return views, teacher_x


def superpixel_centers(x: np.ndarray, seg_flat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=np.float32)
    seg = np.asarray(seg_flat, dtype=np.int64).reshape(-1)
    _, inv = np.unique(seg, return_inverse=True)
    m = int(inv.max()) + 1
    sums = np.zeros((m, x.shape[1]), dtype=np.float64)
    counts = np.bincount(inv, minlength=m).astype(np.float64).clip(min=1.0)
    np.add.at(sums, inv, x.astype(np.float64))
    return (sums / counts[:, None]).astype(np.float32), counts.astype(np.float64)


def teacher_labels(x: np.ndarray, seg_flat: np.ndarray, k: int, seed: int, n_init: int = 5) -> np.ndarray:
    sp_x, sp_size = superpixel_centers(x, seg_flat)
    init = KMeans(n_clusters=int(k), random_state=int(seed) + 37, n_init=int(n_init))
    init.fit(sp_x, sample_weight=sp_size)
    km = KMeans(
        n_clusters=int(k),
        init=init.cluster_centers_.astype(np.float32),
        n_init=1,
        max_iter=50,
        random_state=int(seed) + 37,
    )
    return km.fit_predict(np.asarray(x, dtype=np.float32)).astype(np.int32)


def choose_train_indices(n: int, max_samples: int, seed: int) -> np.ndarray:
    if int(max_samples) <= 0 or int(max_samples) >= int(n):
        return np.arange(int(n), dtype=np.int64)
    rng = np.random.default_rng(int(seed))
    return rng.choice(int(n), size=int(max_samples), replace=False).astype(np.int64)


def iter_batches(n: int, batch_size: int, seed: int, epoch: int) -> Iterable[np.ndarray]:
    rng = np.random.default_rng(int(seed) + int(epoch) * 104729)
    order = rng.permutation(int(n))
    for start in range(0, int(n), int(batch_size)):
        yield order[start : start + int(batch_size)]


def pca_basis(x: np.ndarray, rank: int, seed: int, max_samples: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    rank = int(min(rank, x.shape[1]))
    rng = np.random.default_rng(int(seed))
    if 0 < int(max_samples) < x.shape[0]:
        x = x[rng.choice(x.shape[0], size=int(max_samples), replace=False)]
    x = x - x.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(x, full_matrices=False)
    return vt[:rank].T.astype(np.float32)


@torch.no_grad()
def predict_model(model, xs_all: Sequence[torch.Tensor], batch_size: int) -> Dict[str, np.ndarray]:
    model.eval()
    probs: List[List[np.ndarray]] = [[] for _ in xs_all]
    for start in range(0, xs_all[0].shape[0], int(batch_size)):
        sl = slice(start, min(start + int(batch_size), xs_all[0].shape[0]))
        out = model([x[sl] for x in xs_all])
        for i, p in enumerate(out["probs"]):
            probs[i].append(p.detach().cpu().numpy())
    prob_np = [np.concatenate(items, axis=0).astype(np.float32) for items in probs]
    return {"probs": prob_np, "fused_prob": np.mean(np.stack(prob_np, axis=0), axis=0).astype(np.float32)}


def spatial_refine(prob: np.ndarray, cube: np.ndarray, shape: Tuple[int, int], lam: float, iterations: int) -> np.ndarray:
    h, w = int(shape[0]), int(shape[1])
    p = np.asarray(prob, dtype=np.float32)
    k = p.shape[1]
    unary = -np.log(np.maximum(p, 1e-8)).reshape(h, w, k)
    x = cube.reshape(h, w, -1).astype(np.float32)
    x = x / np.maximum(np.linalg.norm(x, axis=2, keepdims=True), 1e-8)
    wr = np.exp(-np.sum((x[:, :-1] - x[:, 1:]) ** 2, axis=2) / 0.08).astype(np.float32)
    wd = np.exp(-np.sum((x[:-1, :] - x[1:, :]) ** 2, axis=2) / 0.08).astype(np.float32)
    labels = p.argmax(axis=1).reshape(h, w).astype(np.int32)
    for _ in range(int(iterations)):
        e = unary.copy()
        for kk in range(k):
            e[:, 1:, kk] += float(lam) * wr * (labels[:, :-1] != kk)
            e[:, :-1, kk] += float(lam) * wr * (labels[:, 1:] != kk)
            e[1:, :, kk] += float(lam) * wd * (labels[:-1, :] != kk)
            e[:-1, :, kk] += float(lam) * wd * (labels[1:, :] != kk)
        new_labels = e.argmin(axis=2).astype(np.int32)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
    refined = np.zeros_like(p, dtype=np.float32)
    refined[np.arange(p.shape[0]), labels.reshape(-1)] = 1.0
    return refined
