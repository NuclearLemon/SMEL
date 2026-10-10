from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class LowRankBranch(nn.Module):
    def __init__(self, basis_init: np.ndarray, rank: int, beta: float = 1e-2, residual_scale: float = 0.2) -> None:
        super().__init__()
        basis = torch.as_tensor(np.asarray(basis_init, dtype=np.float32))
        if basis.ndim != 2 or basis.shape[1] != int(rank):
            raise ValueError(f"Bad basis shape {tuple(basis.shape)} for rank={rank}")
        self.A = nn.Parameter(basis)
        self.beta = float(beta)
        self.residual_scale = float(residual_scale)
        self.residual = nn.Sequential(
            nn.Linear(int(rank), int(rank)),
            nn.GELU(),
            nn.Linear(int(rank), int(rank)),
        )
        self.norm = nn.LayerNorm(int(rank))
        nn.init.zeros_(self.residual[-1].weight)
        nn.init.zeros_(self.residual[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r = self.A.shape[1]
        eye = torch.eye(r, device=x.device, dtype=x.dtype)
        gram = self.A.T @ self.A + self.beta * eye
        z0 = x @ (self.A @ torch.linalg.inv(gram))
        return self.norm(z0 + self.residual_scale * self.residual(z0))


class SMVELNet(nn.Module):
    def __init__(self, basis_list: Sequence[np.ndarray], rank: int, n_clusters: int) -> None:
        super().__init__()
        self.rank = int(rank)
        self.n_clusters = int(n_clusters)
        self.branches = nn.ModuleList([LowRankBranch(b, self.rank) for b in basis_list])
        self.label_head = nn.Linear(self.rank, self.n_clusters)

    def forward(self, xs: Sequence[torch.Tensor]) -> Dict[str, List[torch.Tensor]]:
        zs: List[torch.Tensor] = []
        logits: List[torch.Tensor] = []
        probs: List[torch.Tensor] = []
        for branch, x in zip(self.branches, xs):
            z = branch(x)
            logit = self.label_head(z)
            zs.append(z)
            logits.append(logit)
            probs.append(F.softmax(logit, dim=1))
        return {"z": zs, "logits": logits, "probs": probs}


def fused_prob(probs: Sequence[torch.Tensor]) -> torch.Tensor:
    return torch.stack(list(probs), dim=0).mean(dim=0)


def sharpen(p: torch.Tensor, temp: float) -> torch.Tensor:
    q = p.clamp_min(1e-8) ** (1.0 / max(float(temp), 1e-6))
    return q / q.sum(dim=1, keepdim=True).clamp_min(1e-8)


def warmup_loss(out: Dict[str, List[torch.Tensor]], labels: torch.Tensor, weight: float = 1.0) -> Dict[str, torch.Tensor]:
    ce = torch.stack([F.cross_entropy(logit, labels) for logit in out["logits"]]).mean()
    return {"loss": float(weight) * ce, "ce": ce.detach()}


def anchor_loss(probs: Sequence[torch.Tensor], p0: torch.Tensor) -> torch.Tensor:
    fused = fused_prob(probs)
    conf = p0.max(dim=1).values
    weight = ((conf - 0.0) / 0.8).clamp(0.0, 1.0)
    if float(weight.sum().detach().cpu()) <= 1e-8:
        return fused.sum() * 0.0
    per = -(p0.detach() * fused.clamp_min(1e-8).log()).sum(dim=1)
    return (per * weight.detach()).sum() / weight.sum().clamp_min(1e-8)


def regional_loss(probs: Sequence[torch.Tensor], sp_ids: torch.Tensor, p0: torch.Tensor) -> torch.Tensor:
    current = fused_prob(probs).detach()
    uniq, inv = torch.unique(sp_ids, sorted=False, return_inverse=True)
    if uniq.numel() <= 1:
        return probs[0].sum() * 0.0
    sums0 = torch.zeros((uniq.numel(), p0.shape[1]), device=p0.device, dtype=p0.dtype)
    sumsc = torch.zeros_like(sums0)
    sums0.index_add_(0, inv, p0.detach())
    sumsc.index_add_(0, inv, current)
    cnt = torch.bincount(inv, minlength=uniq.numel()).to(p0.dtype).unsqueeze(1).clamp_min(1.0)
    q0 = sharpen(sums0 / cnt, 0.5)
    qc = sharpen(sumsc / cnt, 0.5)
    alpha = ((q0.max(dim=1).values.unsqueeze(1) - 0.55) / 0.35).clamp(0.0, 1.0)
    target = (alpha * q0 + (1.0 - alpha) * qc)[inv]
    target = target / target.sum(dim=1, keepdim=True).clamp_min(1e-8)
    losses = [-(target.detach() * p.clamp_min(1e-8).log()).sum(dim=1).mean() for p in probs]
    return torch.stack(losses).mean()


def main_loss(
    out: Dict[str, List[torch.Tensor]],
    sp_ids: torch.Tensor,
    p0: torch.Tensor,
    anchor_weight: float,
    regional_weight: float,
) -> Dict[str, torch.Tensor]:
    anc = anchor_loss(out["probs"], p0)
    reg = regional_loss(out["probs"], sp_ids, p0)
    return {"loss": float(anchor_weight) * anc + float(regional_weight) * reg, "anchor": anc.detach(), "regional": reg.detach()}

