from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import torch

try:
    from .data_loader import load_hsi, n_classes, parse_datasets
    from .model import SMVELNet, main_loss, warmup_loss
    from . import util
except ImportError:
    from data_loader import load_hsi, n_classes, parse_datasets
    from model import SMVELNet, main_loss, warmup_loss
    import util


ROOT = Path(__file__).resolve().parent


def load_config(path: str | Path) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def parse_ints(text: str) -> list[int]:
    return [int(x.strip()) for x in str(text).split(",") if x.strip()]


def train_one(dataset: str, seed: int, cfg: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    util.set_seed(seed)
    t0 = time.time()
    train_cfg = cfg["training"]
    loss_cfg = cfg["loss"]
    evidence_cfg = cfg.get("evidence", {})
    ds_cfg = dict(cfg["dataset_params"][dataset])
    if args.sp_seg > 0:
        ds_cfg["sp_seg"] = int(args.sp_seg)
    if args.rank > 0:
        ds_cfg["rank"] = int(args.rank)

    cube, gt = load_hsi(dataset, args.data_root)
    gt_flat = gt.reshape(-1).astype(np.int32)
    k = n_classes(gt)
    cache_path = Path(args.cache_dir) / f"{dataset}_ers_{int(ds_cfg['sp_seg'])}.npz"
    guide_path = Path(args.data_root) / dataset / "segmentation_guide.npz"
    seg = util.build_segmentation(
        cube,
        int(ds_cfg["sp_seg"]),
        cache_path=None if args.no_cache else cache_path,
        guide_path=guide_path,
    )
    seg_flat = seg.reshape(-1).astype(np.int64)

    print(
        f"\n[RUN] SMVEL dataset={dataset} seed={seed} k={k} sp=ers:{ds_cfg['sp_seg']} rank={ds_cfg['rank']}",
        flush=True,
    )

    pca_dim = int(args.pca_dim) if int(args.pca_dim) > 0 else int(train_cfg["pca_dim"])
    coarse_groups = int(args.coarse_groups) if int(args.coarse_groups) > 0 else int(evidence_cfg.get("coarse_groups", 8))
    fine_groups = int(args.fine_groups) if int(args.fine_groups) > 0 else int(evidence_cfg.get("fine_groups", 16))
    views, teacher_x = util.build_views(cube, seg_flat, pca_dim=pca_dim, coarse_groups=coarse_groups, fine_groups=fine_groups)
    teacher = util.teacher_labels(teacher_x, seg_flat, k, int(ds_cfg["teacher_seed"]))
    teacher_met = util.evaluate(gt_flat, teacher, "teacher_")
    print(f"  [TEACHER] ACC={teacher_met['teacher_ACC']:.4f} NMI={teacher_met['teacher_NMI']:.4f}", flush=True)

    train_idx = util.choose_train_indices(len(gt_flat), int(train_cfg["train_samples"]), seed)
    rank = int(min(int(ds_cfg["rank"]), min(v.shape[1] for v in views)))
    basis = [util.pca_basis(v, rank, seed + 13 * i, int(train_cfg["basis_samples"])) for i, v in enumerate(views)]
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    model = SMVELNet(basis, rank=rank, n_clusters=k).to(device)
    xs_train = [torch.from_numpy(v[train_idx]).float().to(device) for v in views]
    xs_all = [torch.from_numpy(v).float().to(device) for v in views]
    sp_train = torch.from_numpy(seg_flat[train_idx]).long().to(device)
    teacher_train = torch.from_numpy(teacher[train_idx].astype(np.int64)).long().to(device)

    opt = torch.optim.Adam(model.parameters(), lr=float(train_cfg["lr"]), weight_decay=float(train_cfg["weight_decay"]))
    batch_size = int(train_cfg["batch_size"])
    print(f"  [TRAIN] samples={len(train_idx)} device={device} warmup={train_cfg['warmup_epochs']} main={train_cfg['main_epochs']}", flush=True)

    for ep in range(1, int(train_cfg["warmup_epochs"]) + 1):
        vals = []
        model.train()
        for batch in util.iter_batches(len(train_idx), batch_size, seed + 313, ep):
            out = model([x[batch] for x in xs_train])
            losses = warmup_loss(out, teacher_train[batch], float(loss_cfg["warmup_view_ce_weight"]))
            opt.zero_grad(set_to_none=True)
            losses["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(train_cfg["grad_clip"]))
            opt.step()
            vals.append(float(losses["loss"].detach().cpu()))
        if ep in {1, int(train_cfg["warmup_epochs"])}:
            print(f"    warmup={ep:04d} loss={np.mean(vals):.4f}", flush=True)

    with torch.no_grad():
        p0_train = torch.stack(model(xs_train)["probs"], dim=0).mean(dim=0).detach()
    warm_train_pred = util.predict_model(model, xs_train, int(train_cfg["eval_batch_size"]))
    warm_prob = warm_train_pred["fused_prob"]
    warm_usage = util.usage_entropy(warm_prob.argmax(axis=1), k)
    warm_conf = float(warm_prob.max(axis=1).mean())

    opt = torch.optim.Adam(model.parameters(), lr=float(train_cfg["lr"]), weight_decay=float(train_cfg["weight_decay"]))
    guard_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    guard_triggered = False
    guard_reason = ""
    for ep in range(1, int(train_cfg["main_epochs"]) + 1):
        vals = []
        model.train()
        for batch in util.iter_batches(len(train_idx), batch_size, seed, ep):
            out = model([x[batch] for x in xs_train])
            losses = main_loss(out, sp_train[batch], p0_train[batch], float(loss_cfg["anchor_weight"]), float(loss_cfg["regional_weight"]))
            opt.zero_grad(set_to_none=True)
            losses["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(train_cfg["grad_clip"]))
            opt.step()
            vals.append(float(losses["loss"].detach().cpu()))
        if ep == 1 or ep == int(train_cfg["main_epochs"]) or ep % int(train_cfg["eval_every"]) == 0:
            cur_train_pred = util.predict_model(model, xs_train, int(train_cfg["eval_batch_size"]))
            cur_prob = cur_train_pred["fused_prob"]
            usage = util.usage_entropy(cur_prob.argmax(axis=1), k)
            conf = float(cur_prob.max(axis=1).mean())
            min_usage = warm_usage * 0.9
            min_conf = min(warm_conf * 0.95, 0.56)
            if usage < min_usage:
                guard_triggered = True
                guard_reason = f"usage {usage:.4f} < {min_usage:.4f}"
            elif conf < min_conf:
                guard_triggered = True
                guard_reason = f"conf {conf:.4f} < {min_conf:.4f}"
            else:
                guard_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
            print(f"    epoch={ep:04d} loss={np.mean(vals):.4f} usage={usage:.4f} conf={conf:.4f}", flush=True)
            if guard_triggered:
                model.load_state_dict({name: value.to(device) for name, value in guard_state.items()})
                print(f"    guard stop epoch={ep:04d} reason={guard_reason}; restored last safe state", flush=True)
                break

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    pred = util.predict_model(model, xs_all, int(train_cfg["eval_batch_size"]))
    refine_cfg = cfg.get("spatial_refinement", {})
    refined = util.spatial_refine(
        pred["fused_prob"],
        cube,
        gt.shape,
        float(refine_cfg.get("lambda", 35.0)),
        int(refine_cfg.get("iterations", 10)),
    )
    final_label = refined.argmax(axis=1).astype(np.int32)
    final_met = util.evaluate(gt_flat, final_label, "SMVEL_")

    if bool(args.save_predictions):
        out_dir = Path(args.out_dir)
        pred_dir = out_dir / "predictions"
        pred_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            pred_dir / f"{dataset}_seed{seed}.npz",
            gt=gt.astype(np.int32),
            superpixels=seg.astype(np.int32),
            pred=final_label.reshape(gt.shape).astype(np.int32),
        )

    row: Dict[str, Any] = {
        "dataset": dataset,
        "seed": int(seed),
        "sp_method": "ers",
        "sp_seg": int(ds_cfg["sp_seg"]),
        "rank": int(rank),
        "teacher_seed": int(ds_cfg["teacher_seed"]),
        "pca_dim": int(pca_dim),
        "coarse_groups": int(coarse_groups),
        "fine_groups": int(fine_groups),
        "guard_triggered": int(guard_triggered),
        "guard_reason": guard_reason,
        "runtime_sec": float(time.time() - t0),
        **teacher_met,
        **final_met,
    }
    print(
        f"  [RESULT] SMVEL={row['SMVEL_ACC']:.4f}/{row['SMVEL_NMI']:.4f}/{row['SMVEL_ARI']:.4f}",
        flush=True,
    )
    return row


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    cols = ["SMVEL_ACC", "SMVEL_NMI", "SMVEL_ARI", "runtime_sec"]
    out = df.groupby("dataset", sort=False)[cols].mean().reset_index()
    out["runs"] = df.groupby("dataset", sort=False).size().to_numpy()
    mean = {c: float(out[c].mean()) for c in cols}
    mean.update({"dataset": "Mean", "runs": int(df.shape[0])})
    return pd.concat([out, pd.DataFrame([mean])], ignore_index=True)


def main() -> None:
    p = argparse.ArgumentParser(description="Run SMVEL on hyperspectral clustering datasets.")
    p.add_argument("--config", default=str(ROOT / "config.json"))
    p.add_argument("--datasets", default="", help="Comma-separated dataset names. Empty uses config.")
    p.add_argument("--seeds", default="", help="Comma-separated seeds. Empty uses config.")
    p.add_argument("--data-root", default="", help="Dataset directory. Empty uses config/data_root.")
    p.add_argument("--out-dir", default="", help="Output directory. Empty uses config/out_dir.")
    p.add_argument("--cache-dir", default="", help="Superpixel cache directory. Empty uses config/cache_dir.")
    p.add_argument("--device", default="")
    p.add_argument("--sp-seg", type=int, default=0, help="Override superpixel count for all datasets.")
    p.add_argument("--rank", type=int, default=0, help="Override rank for all datasets.")
    p.add_argument("--pca-dim", type=int, default=0, help="Override PCA dimension used by the evidence constructor.")
    p.add_argument("--coarse-groups", type=int, default=0, help="Override the coarse spectral grouping count.")
    p.add_argument("--fine-groups", type=int, default=0, help="Override the fine spectral grouping count.")
    p.add_argument("--no-cache", action="store_true")
    p.add_argument("--no-save-predictions", dest="save_predictions", action="store_false")
    p.set_defaults(save_predictions=True)
    args = p.parse_args()

    cfg = load_config(args.config)
    args.data_root = args.data_root or str(ROOT / cfg.get("data_root", "data"))
    args.out_dir = args.out_dir or str(ROOT / cfg.get("out_dir", "results"))
    args.cache_dir = args.cache_dir or str(ROOT / cfg.get("cache_dir", "cache"))
    datasets = parse_datasets(args.datasets) if args.datasets else list(cfg["datasets"])
    seeds = parse_ints(args.seeds) if args.seeds else [int(x) for x in cfg["seeds"]]

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, Any]] = []
    for dataset in datasets:
        for seed in seeds:
            rows.append(train_one(dataset, seed, cfg, args))
            pd.DataFrame(rows).to_csv(Path(args.out_dir) / "smvel_runs.csv", index=False, quoting=csv.QUOTE_MINIMAL)

    df = pd.DataFrame(rows)
    summary = summarize(df)
    summary.to_csv(Path(args.out_dir) / "smvel_summary.csv", index=False, quoting=csv.QUOTE_MINIMAL)
    print("\n[SMVEL SUMMARY]")
    print(summary.to_string(index=False))
    print(f"[SAVED] {Path(args.out_dir) / 'smvel_runs.csv'}")
    print(f"[SAVED] {Path(args.out_dir) / 'smvel_summary.csv'}")


if __name__ == "__main__":
    main()
