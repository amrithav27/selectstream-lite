"""Q1: causal per-frame importance scorer (docs/plan.md §3, Q1).

a_t = sigmoid(MLP(phi_t)), phi_t built only from frame t and the SAW running state, all detached:
[s_feat_t, sbar_t, W_g g_t (16-d)]. s_attn is not an input: no attention is computed (docs/pilot.md §7). Nothing depends on earlier cuts, so w_t is
a function of the frames alone (needed for offline threshold calibration).
Trained against y_t = 1[t inside an answer window] (streamlite/cli/train_scorer.py: weighted BCE, or a
within-stream objective); it never sees a query.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
NUM_SCALAR_FEATURES = 2
INPUT_MASKS = {"all": (1.0, 1.0), "g": (0.0, 1.0), "scalars": (1.0, 0.0)}  # (scalar gate, g gate); ablation AB-I1
class ImportanceScorer(nn.Module):
    def __init__(self, hidden_size: int, proj_dim: int = 16, mlp_hidden: int = 64, input_mask: str = "all"):
        super().__init__()
        self.g_proj = nn.Linear(hidden_size, proj_dim)
        self.mlp = nn.Sequential(
            nn.Linear(NUM_SCALAR_FEATURES + proj_dim, mlp_hidden),
            nn.GELU(),
            nn.Linear(mlp_hidden, 1),
        )
        # A buffer, so a masked scorer stays masked wherever its state_dict is loaded.
        self.register_buffer("input_gate", torch.tensor(INPUT_MASKS[input_mask]))
    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        # Scorers saved before the mask existed used every input.
        state_dict.setdefault(prefix + "input_gate", torch.ones(2))
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)
    def _features(self, scalars: torch.Tensor, g: torch.Tensor, dim: int) -> torch.Tensor:
        p = self.mlp[0].weight
        gate = self.input_gate.to(dtype=p.dtype)
        g = F.normalize(g.detach().to(dtype=p.dtype), dim=-1) * gate[1]
        return torch.cat([scalars.detach().to(dtype=p.dtype) * gate[0], self.g_proj(g)], dim=dim)
    def forward(self, scalars: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        """scalars: [2] = (s_feat, sbar), g: [D] -> logit []. Inputs are detached: the scorer's only gradient is its own loss."""
        return self.mlp(self._features(scalars, g, 0)).squeeze(-1)
    def forward_batch(self, scalars: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        """Same function over frames: scalars [T, 2], g [T, D] -> logits [T] (Stage 1 trains with this)."""
        return self.mlp(self._features(scalars, g, -1)).squeeze(-1)
def load_scorer(path: str, hidden_size: int = 3584) -> "ImportanceScorer":
    """A Stage-1 scorer.pt, rebuilt with the width and input mask it was trained with."""
    st = torch.load(path, map_location="cpu", weights_only=False)
    sc = ImportanceScorer(hidden_size, proj_dim=int(st.get("proj_dim", 16)), input_mask=st.get("input_mask", "all"))
    sc.load_state_dict(st["scorer"])
    return sc.eval()
def scorer_bce(logits: torch.Tensor, labels: torch.Tensor, pos_weight: float) -> torch.Tensor:
    if logits.numel() == 0:
        return logits.new_zeros(())
    return F.binary_cross_entropy_with_logits(
        logits.float(),
        labels.float(),
        pos_weight=torch.tensor(float(pos_weight), device=logits.device),
    )
