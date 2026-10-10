from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from skimage.segmentation import slic

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover - tqdm is only for progress display
    tqdm = None

try:
    from .data_loader import load_hsi, n_classes, parse_datasets
    from . import util
except ImportError:
    from data_loader import load_hsi, n_classes, parse_datasets
    import util


ROOT = Path(__file__).resolve().parent
DEFAULT_SEED = 7
DEFAULT_REPEATS = 5
DEFAULT_N_INIT = 10
DEFAULT_MAX_ITER = 300
DEFAULT_KMEANS_FIT_SAMPLES = 0
DEFAULT_KMEANS_VERBOSE = 0
DEFAULT_RAW_KMEANS_VERBOSE = 0
DEFAULT_SP_TARGET_AREA = 200
DEFAULT_SLIC_COMPACTNESS = 10.0
DEFAULT_SLIC_SIGMA = 1.0
DEFAULT_GRAPH_K = 8
DEFAULT_GRAPH_ALPHA = 0.65

DESCRIPTORS = [
    ("Raw spectra", "raw"),
    ("PCA20", "pca20"),
    ("Sub PCA20", "sub_pca20"),
    ("Spectral mean", "spectral_mean"),
    ("Spectral slope", "spectral_slope"),
    ("Super-pixel mean", "sp_mean"),
    ("SP graph", "sp_graph"),
    ("SP graph weighted", "sp_graph_weighted"),
    ("View 1", "view1"),
    ("View 2", "view2"),
    ("View 3", "view3"),
    ("Naive concat", "concat"),
]


def progress(seq: Iterable, **kwargs):
    if tqdm is None:
        return seq
    return tqdm(seq, **kwargs)


def load_config(path: str | Path) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def band_split_pca(x: np.ndarray, total_dim: int, seed: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    splits = [idx for idx in np.array_split(np.arange(x.shape[1]), 2) if len(idx)]
    dims = [int(total_dim) // len(splits)] * len(splits)
    for i in range(int(total_dim) - sum(dims)):
        dims[i % len(dims)] += 1
    views = []
    for i, (idx, dim) in enumerate(zip(splits, dims)):
        sub = x[:, idx]
        dim = max(1, min(int(dim), sub.shape[1], sub.shape[0]))
        views.append(PCA(n_components=dim, random_state=int(seed) + i).fit_transform(sub).astype(np.float32))
    return np.concatenate(views, axis=1).astype(np.float32)


def sp_node_features(x: np.ndarray, seg_flat: np.ndarray, weighted: bool) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    seg = np.asarray(seg_flat, dtype=np.int64).reshape(-1)
    nodes = []
    for sid in range(int(seg.max()) + 1):
        idx = np.flatnonzero(seg == sid)
        vals = x[idx]
        mean = vals.mean(axis=0)
        if weighted:
            dist = np.sum((vals - mean[None, :]) ** 2, axis=1)
            scale = float(np.median(dist) + 1e-8)
            w = np.exp(-dist / scale).astype(np.float32)
            mean = (vals * w[:, None]).sum(axis=0) / max(float(w.sum()), 1e-8)
        nodes.append(mean.astype(np.float32))
    return util.standardize(np.asarray(nodes, dtype=np.float32))


def graph_smooth(nodes: np.ndarray, graph_k: int, weighted: bool, alpha: float) -> np.ndarray:
    nodes = util.standardize(nodes)
    n = nodes.shape[0]
    kk = max(1, min(int(graph_k) + 1, n))
    nn = NearestNeighbors(n_neighbors=kk, metric="euclidean")
    nn.fit(nodes)
    dist, ind = nn.kneighbors(nodes, return_distance=True)
    ind = ind[:, 1:]
    dist = dist[:, 1:].astype(np.float32)
    if ind.shape[1] == 0:
        return nodes
    if weighted:
        local = np.maximum(dist[:, -1:], 1e-6)
        weights = np.exp(-((dist / local) ** 2)).astype(np.float32)
    else:
        weights = np.ones_like(dist, dtype=np.float32)
    weights = weights / np.maximum(weights.sum(axis=1, keepdims=True), 1e-8)
    neigh = np.zeros_like(nodes, dtype=np.float32)
    for j in range(ind.shape[1]):
        neigh += weights[:, j : j + 1] * nodes[ind[:, j]]
    return util.standardize((1.0 - float(alpha)) * nodes + float(alpha) * neigh)


def broadcast(nodes: np.ndarray, seg_flat: np.ndarray) -> np.ndarray:
    return np.asarray(nodes, dtype=np.float32)[np.asarray(seg_flat, dtype=np.int64).reshape(-1)]


def build_descriptor_pool(cube: np.ndarray, seg_flat: np.ndarray, args: argparse.Namespace) -> Dict[str, np.ndarray]:
    flat = util.standardize(cube.reshape(-1, cube.shape[-1]))
    pca_dim = max(1, min(int(args.pca_dim), flat.shape[1]))
    pca = PCA(n_components=pca_dim, random_state=0).fit_transform(flat).astype(np.float32)
    d1 = util.spectral_derivative(flat)
    pool8 = util.spectral_pool(flat, 8)
    d1pool8 = util.spectral_pool(d1, 8)
    pool16 = util.spectral_pool(flat, 16)
    d1pool16 = util.spectral_pool(d1, 16)
    sp_pool8 = util.sp_mean(pool8, seg_flat)
    sp_pool16 = util.sp_mean(pool16, seg_flat)

    view1 = np.concatenate([pool16, d1pool16], axis=1).astype(np.float32)
    view2 = np.concatenate([pca, pool8, d1pool8, sp_pool8], axis=1).astype(np.float32)
    view3 = np.concatenate([view1, util.sp_mean(view1, seg_flat)], axis=1).astype(np.float32)

    graph_base = view1
    simple_nodes = graph_smooth(sp_node_features(graph_base, seg_flat, weighted=False), args.graph_k, False, args.graph_alpha)
    weighted_nodes = graph_smooth(sp_node_features(graph_base, seg_flat, weighted=True), args.graph_k, True, args.graph_alpha)

    pool = {
        "raw": flat,
        "pca20": pca,
        "sub_pca20": band_split_pca(flat, pca_dim, int(args.seed)),
        "spectral_mean": pool16,
        "spectral_slope": d1pool16,
        "sp_mean": sp_pool16,
        "sp_graph": broadcast(simple_nodes, seg_flat),
        "sp_graph_weighted": broadcast(weighted_nodes, seg_flat),
        "view1": view1,
        "view2": view2,
        "view3": view3,
    }
    pool["concat"] = np.concatenate(
        [util.standardize(pool["view1"]), util.standardize(pool["view2"]), util.standardize(pool["view3"])],
        axis=1,
    ).astype(np.float32)
    return {k: util.standardize(v) for k, v in pool.items()}


def resolve_sp_seg(cube: np.ndarray, args: argparse.Namespace) -> tuple[int, str]:
    if int(args.sp_seg) > 0:
        return int(args.sp_seg), "fixed"
    area = max(1, int(args.sp_target_area))
    return max(2, int(round(float(cube.shape[0] * cube.shape[1]) / float(area)))), f"area{area}"


def build_slic_segmentation(cube: np.ndarray, n_segments: int, seed: int = DEFAULT_SEED) -> np.ndarray:
    flat = util.standardize(cube.reshape(-1, cube.shape[-1]))
    n_components = min(3, cube.shape[-1])
    vis = PCA(n_components=n_components, random_state=int(seed)).fit_transform(flat).astype(np.float32)
    vis = vis - vis.min(axis=0, keepdims=True)
    vis = vis / np.maximum(vis.max(axis=0, keepdims=True), 1e-8)
    if n_components < 3:
        vis = np.tile(vis, (1, int(np.ceil(3 / n_components))))[:, :3]
    vis = vis.reshape(cube.shape[0], cube.shape[1], 3)
    seg = slic(
        vis,
        n_segments=int(n_segments),
        compactness=DEFAULT_SLIC_COMPACTNESS,
        sigma=DEFAULT_SLIC_SIGMA,
        start_label=0,
        channel_axis=-1,
        convert2lab=False,
        enforce_connectivity=True,
    )
    return util.relabel(seg)


def fit_predict_kmeans(
    x: np.ndarray,
    k: int,
    seed: int,
    args: argparse.Namespace,
    descriptor: str,
    repeat: int,
) -> tuple[np.ndarray, float]:
    x = np.asarray(x, dtype=np.float32)
    rng = np.random.default_rng(int(seed))
    if int(args.kmeans_fit_samples) > 0 and x.shape[0] > int(args.kmeans_fit_samples):
        fit_idx = rng.choice(x.shape[0], size=int(args.kmeans_fit_samples), replace=False)
        x_fit = x[fit_idx]
    else:
        x_fit = x
    verbose = int(args.raw_kmeans_verbose) if descriptor == "Raw spectra" else int(args.kmeans_verbose)
    print(
        f"      [KMeans] {descriptor} repeat={repeat + 1}/{args.repeats} "
        f"fit_shape={tuple(x_fit.shape)} predict_pixels={x.shape[0]} n_init={args.n_init}",
        flush=True,
    )
    t0 = time.time()
    km = KMeans(
        n_clusters=int(k),
        n_init=int(args.n_init),
        random_state=int(seed),
        max_iter=int(args.max_iter),
        algorithm="lloyd",
        verbose=verbose,
    )
    km.fit(x_fit)
    pred = km.predict(x).astype(np.int32)
    return pred, float(time.time() - t0)


def run_descriptor(
    dataset: str,
    descriptor: str,
    key: str,
    x: np.ndarray,
    gt_flat: np.ndarray,
    k: int,
    args: argparse.Namespace,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for rep in progress(range(int(args.repeats)), desc=f"{dataset}:{descriptor}", leave=False):
        pred, runtime = fit_predict_kmeans(x, k, int(args.seed) + rep, args, descriptor, rep)
        met = util.evaluate(gt_flat, pred, "")
        rows.append(
            {
                "dataset": dataset,
                "descriptor": descriptor,
                "key": key,
                "repeat": rep,
                "dim": int(x.shape[1]),
                "pixels": int(x.shape[0]),
                "ACC": met["ACC"],
                "NMI": met["NMI"],
                "ARI": met["ARI"],
                "Kappa": met["Kappa"],
                "runtime_sec": runtime,
            }
        )
        print(f"        -> ACC={met['ACC']:.4f} NMI={met['NMI']:.4f} ARI={met['ARI']:.4f} time={runtime:.1f}s", flush=True)
    return rows


def run_dataset(dataset: str, args: argparse.Namespace) -> pd.DataFrame:
    t0 = time.time()
    cube, gt = load_hsi(dataset, args.data_root, verify_reproduction=False)
    gt_flat = gt.reshape(-1).astype(np.int32)
    k = n_classes(gt)
    sp_seg, sp_rule = resolve_sp_seg(cube, args)
    print(f"\n[DATASET] {dataset} shape={cube.shape} classes={k} sp={sp_seg} ({sp_rule})", flush=True)
    print(f"  [SLIC] computing partition: target_segments={sp_seg}", flush=True)
    seg = build_slic_segmentation(cube, sp_seg, seed=int(args.seed))
    print(f"  [SLIC] actual_segments={np.unique(seg).size}", flush=True)
    print("  [BUILD] spectral, regional and graph descriptors", flush=True)
    pool = build_descriptor_pool(cube, seg.reshape(-1), args)
    selected = [(name, key) for name, key in DESCRIPTORS if name in args.descriptor_names]

    rows: List[Dict[str, Any]] = []
    for name, key in progress(selected, desc=f"{dataset}:descriptors", leave=True):
        x = pool[key]
        print(f"  [DESC] {name:18s} dim={x.shape[1]:4d}", flush=True)
        rows.extend(run_descriptor(dataset, name, key, x, gt_flat, k, args))
    df = pd.DataFrame(rows)
    df["sp_seg"] = int(sp_seg)
    df["sp_rule"] = sp_rule
    df["graph_k"] = int(args.graph_k)
    df["graph_alpha"] = float(args.graph_alpha)
    df["dataset_runtime_sec"] = float(time.time() - t0)
    return df


def summarize(detail: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metrics = ["ACC", "NMI", "ARI", "Kappa", "runtime_sec"]
    summary = (
        detail.groupby(["dataset", "descriptor", "key", "dim"], as_index=False)[metrics]
        .agg(["mean", "std"])
        .reset_index()
    )
    summary.columns = ["_".join([str(x) for x in col if x]) for col in summary.columns.to_flat_index()]
    average = (
        summary.groupby(["descriptor", "key"], as_index=False)[["ACC_mean", "NMI_mean", "ARI_mean"]]
        .mean()
        .sort_values("ACC_mean", ascending=False)
    )
    wide_rows = []
    order = [name for name, _ in DESCRIPTORS]
    for dataset, g in summary.groupby("dataset", sort=False):
        row: Dict[str, Any] = {"Dataset": dataset}
        for name in order:
            sub = g[g["descriptor"] == name]
            if sub.empty:
                continue
            item = sub.iloc[0]
            row[f"{name}_ACC"] = round(float(item["ACC_mean"]), 3)
            row[f"{name}_NMI"] = round(float(item["NMI_mean"]), 3)
            row[f"{name}_ARI"] = round(float(item["ARI_mean"]), 3)
        wide_rows.append(row)
    paper = pd.DataFrame(wide_rows)
    return summary, average, paper


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run structural prior diagnosis for SMEL.")
    p.add_argument("--config", default=str(ROOT / "config.json"))
    p.add_argument("--datasets", default="", help="Comma-separated dataset names. Empty uses config.")
    p.add_argument("--data-root", default="", help="Dataset directory. Empty uses config.")
    args = p.parse_args(argv)
    cfg = load_config(args.config)
    args.datasets = args.datasets or ",".join(cfg["datasets"])
    args.data_root = args.data_root or str(ROOT / cfg.get("data_root", "data"))
    args.seed = DEFAULT_SEED
    args.repeats = DEFAULT_REPEATS
    args.n_init = DEFAULT_N_INIT
    args.max_iter = DEFAULT_MAX_ITER
    args.kmeans_fit_samples = DEFAULT_KMEANS_FIT_SAMPLES
    args.kmeans_verbose = DEFAULT_KMEANS_VERBOSE
    args.raw_kmeans_verbose = DEFAULT_RAW_KMEANS_VERBOSE
    args.pca_dim = int(cfg["training"].get("pca_dim", 20))
    args.sp_seg = 0
    args.sp_target_area = DEFAULT_SP_TARGET_AREA
    args.graph_k = DEFAULT_GRAPH_K
    args.graph_alpha = DEFAULT_GRAPH_ALPHA
    args.descriptor_names = [name for name, _ in DESCRIPTORS]
    return args


def main() -> None:
    args = parse_args()
    datasets = parse_datasets(args.datasets)

    all_rows = []
    for dataset in progress(datasets, desc="datasets", leave=True):
        all_rows.append(run_dataset(dataset, args))
    detail = pd.concat(all_rows, ignore_index=True)
    summary, average, paper = summarize(detail)

    print("\n[STRUCTURAL PRIOR | BY DATASET]")
    cols = ["dataset", "descriptor", "dim", "ACC_mean", "NMI_mean", "ARI_mean"]
    print(summary[cols].round(4).to_string(index=False))
    print("\n[STRUCTURAL PRIOR | PAPER TABLE]")
    print(paper.to_string(index=False))
    print("\n[STRUCTURAL PRIOR | AVERAGE]")
    print(average[["descriptor", "ACC_mean", "NMI_mean", "ARI_mean"]].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
