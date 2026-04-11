"""Multi-task trunk shaping for the best hybrid oracle (Sub-Plan 05).

Trains a shared trunk representation with auxiliary regression, classification,
and ranking heads. The shaped trunk is then consumed by the SP4 oracle head
(DKL Ensemble) for uncertainty-aware prediction.

Usage:
    from bayesdiff.multi_task import MultiTaskTrunk, MultiTaskHybridOracle
    from bayesdiff.hybrid_oracle import DKLEnsembleOracle

    trunk = MultiTaskTrunk(input_dim=128, trunk_dim=128)
    oracle_head = DKLEnsembleOracle(input_dim=128, n_members=5)
    hybrid = MultiTaskHybridOracle(trunk, oracle_head)
"""

from __future__ import annotations

import logging
import random
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import ndcg_score

from bayesdiff.oracle_interface import OracleHead, OracleResult

logger = logging.getLogger(__name__)


# ============================================================================
# Trunk & Head Modules
# ============================================================================


class SharedTrunk(nn.Module):
    """Shared feature extractor for multi-task learning.

    Architecture: input_dim → hidden_dim → … → output_dim
    with ReLU activations, dropout, and optional residual connection.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        output_dim: int = 128,
        n_layers: int = 2,
        dropout: float = 0.1,
        residual: bool = True,
    ):
        super().__init__()
        layers: list[nn.Module] = []
        d_in = input_dim
        for i in range(n_layers):
            d_out = hidden_dim if i < n_layers - 1 else output_dim
            layers.append(nn.Linear(d_in, d_out))
            if i < n_layers - 1:
                layers.append(nn.ReLU(inplace=True))
                layers.append(nn.Dropout(dropout))
            d_in = d_out
        self.mlp = nn.Sequential(*layers)

        self.residual = residual
        if residual:
            self.proj = (
                nn.Linear(input_dim, output_dim, bias=False)
                if input_dim != output_dim
                else nn.Identity()
            )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """z: (B, input_dim) → h: (B, output_dim)"""
        h = self.mlp(z)
        if self.residual:
            h = h + self.proj(z)
        return h


class RegressionHead(nn.Module):
    """Single linear layer for pKd regression."""

    def __init__(self, input_dim: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.linear(h).squeeze(-1)


class ClassificationHead(nn.Module):
    """Single linear layer for active/inactive classification (outputs logits)."""

    def __init__(self, input_dim: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.linear(h).squeeze(-1)

    def predict_prob(self, h: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.forward(h))


class RankingHead(nn.Module):
    """Single linear layer for pairwise ranking scores."""

    def __init__(self, input_dim: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.linear(h).squeeze(-1)


# ============================================================================
# GroupedBatchSampler
# ============================================================================


class GroupedBatchSampler:
    """Yields batches filled by sampling entire protein clusters.

    Ensures each batch has multiple same-group samples for pair construction.
    """

    def __init__(
        self,
        group_ids: np.ndarray,
        batch_size: int = 128,
        min_group_size: int = 2,
        drop_last: bool = False,
    ):
        self.groups: dict[int, list[int]] = {}
        for i, g in enumerate(group_ids):
            self.groups.setdefault(int(g), []).append(i)
        # Filter groups with < min_group_size
        self.groups = {
            g: idx for g, idx in self.groups.items() if len(idx) >= min_group_size
        }
        self.batch_size = batch_size
        self.drop_last = drop_last

    def __iter__(self):
        group_keys = list(self.groups.keys())
        random.shuffle(group_keys)
        batch: list[int] = []
        for g in group_keys:
            batch.extend(self.groups[g])
            if len(batch) >= self.batch_size:
                yield batch[: self.batch_size]
                batch = batch[self.batch_size :]
        if batch and not self.drop_last:
            yield batch

    def __len__(self):
        total = sum(len(idx) for idx in self.groups.values())
        n = total // self.batch_size
        if not self.drop_last and total % self.batch_size:
            n += 1
        return n


# ============================================================================
# MultiTaskTrunk
# ============================================================================


class MultiTaskTrunk(nn.Module):
    """Multi-task trunk shaping module.

    Two phases:
      v1 (enable_ranking=False): L = λ₁ L_reg + λ₂ L_cls
      v2 (enable_ranking=True):  L = λ₁ L_reg + λ₂ L_cls + λ₃ L_rank
    """

    def __init__(
        self,
        input_dim: int = 128,
        hidden_dim: int = 256,
        trunk_dim: int = 128,
        n_layers: int = 2,
        dropout: float = 0.1,
        residual: bool = True,
        activity_threshold: float = 7.0,
        enable_ranking: bool = False,
        learned_weights: bool = False,
        cls_pos_weight: Optional[float] = None,
    ):
        super().__init__()
        self.trunk = SharedTrunk(
            input_dim, hidden_dim, trunk_dim, n_layers, dropout, residual
        )
        self.reg_head = RegressionHead(trunk_dim)
        self.cls_head = ClassificationHead(trunk_dim)
        self.rank_head = RankingHead(trunk_dim) if enable_ranking else None

        self.enable_ranking = enable_ranking
        self.threshold = activity_threshold

        self.learned_weights = learned_weights
        if learned_weights:
            n_tasks = 3 if enable_ranking else 2
            self.log_sigma = nn.Parameter(torch.zeros(n_tasks))

        self.cls_pos_weight = cls_pos_weight

    def forward(self, z: torch.Tensor):
        h = self.trunk(z)
        reg = self.reg_head(h)
        cls_logits = self.cls_head(h)
        rank = self.rank_head(h) if self.rank_head is not None else None
        return h, reg, cls_logits, rank

    def compute_loss(
        self,
        z: torch.Tensor,
        y: torch.Tensor,
        groups: Optional[torch.Tensor] = None,
        lambda_reg: float = 1.0,
        lambda_cls: float = 0.5,
        lambda_rank: float = 0.3,
        max_pairs_per_group: int = 50,
    ):
        h, reg_out, cls_logits, rank_out = self.forward(z)

        # Task 1: Regression (MSE)
        L_reg = F.mse_loss(reg_out, y)

        # Task 2: Classification (BCE with logits)
        c = (y >= self.threshold).float()
        pw = (
            torch.tensor([self.cls_pos_weight], device=z.device, dtype=z.dtype)
            if self.cls_pos_weight
            else None
        )
        L_cls = F.binary_cross_entropy_with_logits(cls_logits, c, pos_weight=pw)

        # Task 3: Ranking (BPR, optional)
        L_rank = torch.tensor(0.0, device=z.device)
        n_pairs = 0
        if self.enable_ranking and rank_out is not None:
            assert groups is not None, "groups required when ranking is enabled"
            pairs = self._make_grouped_pairs(y, groups, max_pairs_per_group)
            n_pairs = len(pairs)
            if n_pairs > 0:
                s_better = rank_out[pairs[:, 0]]
                s_worse = rank_out[pairs[:, 1]]
                L_rank = -F.logsigmoid(s_better - s_worse).mean()

        # Combine
        if self.learned_weights:
            precisions = torch.exp(-2 * self.log_sigma)
            losses = [L_reg, L_cls] + ([L_rank] if self.enable_ranking else [])
            total = sum(p * l for p, l in zip(precisions, losses))
            total = total + self.log_sigma.sum()
        else:
            total = lambda_reg * L_reg + lambda_cls * L_cls
            if self.enable_ranking:
                total = total + lambda_rank * L_rank

        loss_dict = {
            "L_reg": L_reg.item(),
            "L_cls": L_cls.item(),
            "L_rank": L_rank.item(),
            "L_total": total.item(),
            "n_pairs": n_pairs,
        }
        if self.learned_weights:
            for i, name in enumerate(
                ["sigma_reg", "sigma_cls", "sigma_rank"][: len(self.log_sigma)]
            ):
                loss_dict[name] = torch.exp(self.log_sigma[i]).item()

        return total, loss_dict

    def _make_grouped_pairs(
        self,
        y: torch.Tensor,
        groups: torch.Tensor,
        max_pairs_per_group: int = 50,
    ) -> torch.Tensor:
        device = y.device
        unique_groups = groups.unique()
        all_pairs: list[torch.Tensor] = []

        for g in unique_groups:
            mask = groups == g
            idx = mask.nonzero(as_tuple=True)[0]
            if len(idx) < 2:
                continue
            y_g = y[idx]
            n = len(idx)
            ii, jj = torch.meshgrid(
                torch.arange(n, device=device),
                torch.arange(n, device=device),
                indexing="ij",
            )
            valid = y_g[ii] > y_g[jj]
            local_pairs = torch.stack([ii[valid], jj[valid]], dim=1)
            global_pairs = torch.stack(
                [idx[local_pairs[:, 0]], idx[local_pairs[:, 1]]], dim=1
            )
            if len(global_pairs) > max_pairs_per_group:
                perm = torch.randperm(len(global_pairs), device=device)[
                    :max_pairs_per_group
                ]
                global_pairs = global_pairs[perm]
            all_pairs.append(global_pairs)

        if not all_pairs:
            return torch.zeros(0, 2, dtype=torch.long, device=device)
        return torch.cat(all_pairs, dim=0)

    def extract_trunk_features(self, z) -> np.ndarray:
        if isinstance(z, np.ndarray):
            z = torch.from_numpy(z).float()
        self.eval()
        with torch.no_grad():
            h = self.trunk(z.to(next(self.parameters()).device))
        return h.cpu().numpy()


# ============================================================================
# MultiTaskHybridOracle
# ============================================================================


class MultiTaskHybridOracle:
    """Multi-task shaped trunk + best SP4 oracle head.

    Two-stage pipeline:
      Stage 1: Train MultiTaskTrunk on frozen embeddings with multi-task loss
      Stage 2: Freeze trunk, extract h_trunk, fit SP4 OracleHead on top
    """

    def __init__(self, multi_task: MultiTaskTrunk, oracle_head: OracleHead):
        self.multi_task = multi_task
        self.oracle_head = oracle_head

    def train_trunk(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        groups_train: Optional[np.ndarray] = None,
        groups_val: Optional[np.ndarray] = None,
        n_epochs: int = 200,
        batch_size: int = 256,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        lambda_reg: float = 1.0,
        lambda_cls: float = 0.5,
        lambda_rank: float = 0.3,
        patience: int = 20,
        device: str = "cuda",
    ) -> dict:
        """Stage 1: Train multi-task trunk end-to-end."""
        self.multi_task = self.multi_task.to(device)
        optimizer = torch.optim.AdamW(
            self.multi_task.parameters(), lr=lr, weight_decay=weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=n_epochs
        )

        Xt = torch.from_numpy(X_train).float().to(device)
        yt = torch.from_numpy(y_train).float().to(device)
        Xv = torch.from_numpy(X_val).float().to(device)
        yv = torch.from_numpy(y_val).float().to(device)
        gt = (
            torch.from_numpy(groups_train).long().to(device)
            if groups_train is not None
            else None
        )
        gv = (
            torch.from_numpy(groups_val).long().to(device)
            if groups_val is not None
            else None
        )

        history: dict = {
            k: []
            for k in [
                "train_loss",
                "val_loss",
                "train_L_reg",
                "train_L_cls",
                "train_L_rank",
                "val_L_reg",
                "val_L_cls",
                "val_L_rank",
            ]
        }
        best_val_loss = float("inf")
        best_epoch = 0
        best_state: Optional[dict] = None

        N = len(Xt)
        for epoch in range(n_epochs):
            # Train
            self.multi_task.train()

            if self.multi_task.enable_ranking and groups_train is not None:
                sampler = GroupedBatchSampler(
                    groups_train, batch_size=batch_size, min_group_size=2
                )
                batches = list(sampler)
            else:
                perm = torch.randperm(N, device=device)
                batches = [
                    perm[start : start + batch_size]
                    for start in range(0, N, batch_size)
                ]

            epoch_losses: list[dict] = []
            for idx in batches:
                if isinstance(idx, list):
                    idx = torch.tensor(idx, dtype=torch.long, device=device)
                z_b, y_b = Xt[idx], yt[idx]
                g_b = gt[idx] if gt is not None else None
                loss, ld = self.multi_task.compute_loss(
                    z_b,
                    y_b,
                    groups=g_b,
                    lambda_reg=lambda_reg,
                    lambda_cls=lambda_cls,
                    lambda_rank=lambda_rank,
                )
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.multi_task.parameters(), 1.0)
                optimizer.step()
                epoch_losses.append(ld)
            scheduler.step()

            for key in ["L_reg", "L_cls", "L_rank", "L_total"]:
                avg = float(np.mean([d[key] for d in epoch_losses]))
                if key == "L_total":
                    history["train_loss"].append(avg)
                else:
                    history[f"train_{key}"].append(avg)

            # Validate
            self.multi_task.eval()
            with torch.no_grad():
                val_loss, vld = self.multi_task.compute_loss(
                    Xv,
                    yv,
                    groups=gv,
                    lambda_reg=lambda_reg,
                    lambda_cls=lambda_cls,
                    lambda_rank=lambda_rank,
                )
            history["val_loss"].append(vld["L_total"])
            history["val_L_reg"].append(vld["L_reg"])
            history["val_L_cls"].append(vld["L_cls"])
            history["val_L_rank"].append(vld["L_rank"])

            if vld["L_total"] < best_val_loss:
                best_val_loss = vld["L_total"]
                best_epoch = epoch
                best_state = {
                    k: v.cpu().clone()
                    for k, v in self.multi_task.state_dict().items()
                }
            elif epoch - best_epoch >= patience:
                logger.info(f"  Trunk early stopping at epoch {epoch + 1}")
                break

        if best_state is not None:
            self.multi_task.load_state_dict(best_state)
        self.multi_task = self.multi_task.to(device)
        history["best_epoch"] = best_epoch
        return history

    def train_oracle(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        **kwargs,
    ) -> dict:
        """Stage 2: Freeze trunk, extract h_trunk, fit SP4 oracle head."""
        self.multi_task.eval()
        h_train = self.multi_task.extract_trunk_features(X_train)
        h_val = self.multi_task.extract_trunk_features(X_val)
        return self.oracle_head.fit(h_train, y_train, h_val, y_val, **kwargs)

    def predict(self, X: np.ndarray) -> OracleResult:
        """Full inference: z → trunk → oracle_head → OracleResult."""
        self.multi_task.eval()
        h = self.multi_task.extract_trunk_features(X)
        result = self.oracle_head.predict(h)

        # Attach auxiliary head outputs
        z_t = torch.from_numpy(X).float()
        with torch.no_grad():
            device = next(self.multi_task.parameters()).device
            _, _, cls_logits, rank_out = self.multi_task(z_t.to(device))
            result.aux["cls_prob"] = torch.sigmoid(cls_logits).cpu().numpy()
            if rank_out is not None:
                result.aux["rank_score"] = rank_out.cpu().numpy()

        return result

    def predict_for_fusion(self, X: np.ndarray) -> OracleResult:
        """Expensive path with full Jacobian ∂μ/∂z for fusion.py.

        Uses two-stage Jacobian composition:
            J_fusion = ∂μ/∂z = (∂μ/∂h) · (∂h/∂z)

        The oracle's Jacobian (∂μ/∂h) comes from oracle_head.predict_for_fusion()
        as numpy. The trunk Jacobian (∂h/∂z) is computed via torch autograd.
        These are combined via einsum.

        Note: This is only called on the fusion path, never during standard
        evaluation. If profiling shows the O(d_h) backward passes are too slow,
        the fallback is to propagate Σ_gen into trunk space instead.
        """
        self.multi_task.eval()
        device = next(self.multi_task.parameters()).device
        z_t = torch.from_numpy(X).float().to(device).requires_grad_(True)

        # Forward through trunk (keep graph)
        h = self.multi_task.trunk(z_t)  # (N, trunk_dim), differentiable

        # Get oracle Jacobian ∂μ/∂h from oracle head
        h_np = h.detach().cpu().numpy()
        oracle_result = self.oracle_head.predict_for_fusion(h_np)
        J_oracle = oracle_result.jacobian  # (N, d_h) or None

        if J_oracle is not None:
            N, d_z = z_t.shape
            d_h = h.shape[1]
            J_trunk = torch.zeros(N, d_h, d_z, device=device)
            for k in range(d_h):
                grad_outputs = torch.zeros_like(h)
                grad_outputs[:, k] = 1.0
                g = torch.autograd.grad(
                    h,
                    z_t,
                    grad_outputs=grad_outputs,
                    retain_graph=True,
                    create_graph=False,
                )[0]
                J_trunk[:, k, :] = g

            J_oracle_t = torch.from_numpy(J_oracle).float().to(device)
            J_full = torch.einsum("nh,nhz->nz", J_oracle_t, J_trunk)
            oracle_result.jacobian = J_full.detach().cpu().numpy()

        return oracle_result

    def save(self, path: str):
        import os

        os.makedirs(path, exist_ok=True)
        torch.save(self.multi_task.state_dict(), os.path.join(path, "trunk.pt"))
        self.oracle_head.save(os.path.join(path, "oracle_head"))

    def load(self, path: str):
        import os

        self.multi_task.load_state_dict(
            torch.load(os.path.join(path, "trunk.pt"), weights_only=False)
        )
        self.oracle_head.load(os.path.join(path, "oracle_head"))


# ============================================================================
# Metrics
# ============================================================================


def within_group_ndcg(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    groups: np.ndarray,
    k: int = 10,
) -> dict:
    """Compute NDCG@k averaged over protein clusters.

    Returns dict with 'ndcg_mean', 'ndcg_std', 'n_groups_evaluated',
    'ndcg_per_group'.
    """
    unique_groups = np.unique(groups)
    ndcgs: dict[int, float] = {}
    for g in unique_groups:
        mask = groups == g
        if mask.sum() < 2:
            continue
        yt = y_true[mask]
        yp = y_pred[mask]
        cutoff = min(k, len(yt))
        ndcgs[int(g)] = float(
            ndcg_score(yt.reshape(1, -1), yp.reshape(1, -1), k=cutoff)
        )

    vals = list(ndcgs.values())
    return {
        "ndcg_mean": float(np.mean(vals)) if vals else 0.0,
        "ndcg_std": float(np.std(vals)) if vals else 0.0,
        "n_groups_evaluated": len(vals),
        "ndcg_per_group": ndcgs,
    }
