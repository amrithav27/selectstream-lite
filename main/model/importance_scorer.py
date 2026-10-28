"""Q1: causal per-frame importance scorer (docs/plan.md §3, Q1).

a_t = sigmoid(MLP(phi_t)), phi_t built only from frame t and the SAW running state, all detached:
[s_feat_t, sbar_t, W_g g_t (16-d)]. s_attn is not an input: no attention is computed (docs/pilot.md §7). Nothing depends on earlier cuts, so w_t is
a function of the frames alone (needed for offline threshold calibration).
Trained with weighted BCE against y_t = 1[t inside an answer window]; it never sees a query.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

NUM_SCALAR_FEATURES = 2


class ImportanceScorer(nn.Module):
    def __init__(self, hidden_size: int, proj_dim: int = 16, mlp_hidden: int = 64):
        super().__init__()
        self.g_proj = nn.Linear(hidden_size, proj_dim)
        self.mlp = nn.Sequential(
            nn.Linear(NUM_SCALAR_FEATURES + proj_dim, mlp_hidden),
            nn.GELU(),
            nn.Linear(mlp_hidden, 1),
        )

    def forward(self, scalars: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        """scalars: [2] = (s_feat, sbar), g: [D] -> logit []. Inputs are detached: the scorer's only gradient is its own BCE."""
        p = self.mlp[0].weight
        g = F.normalize(g.detach().to(dtype=p.dtype), dim=-1)
        x = torch.cat([scalars.detach().to(dtype=p.dtype), self.g_proj(g)], dim=0)
        return self.mlp(x).squeeze(-1)

    def forward_batch(self, scalars: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        """Same function over frames: scalars [T, 2], g [T, D] -> logits [T] (Stage 1 trains with this)."""
        p = self.mlp[0].weight
        g = F.normalize(g.detach().to(dtype=p.dtype), dim=-1)
        x = torch.cat([scalars.detach().to(dtype=p.dtype), self.g_proj(g)], dim=-1)
        return self.mlp(x).squeeze(-1)


def scorer_bce(logits: torch.Tensor, labels: torch.Tensor, pos_weight: float) -> torch.Tensor:
    if logits.numel() == 0:
        return logits.new_zeros(())
    return F.binary_cross_entropy_with_logits(
        logits.float(),
        labels.float(),
        pos_weight=torch.tensor(float(pos_weight), device=logits.device),
    )
