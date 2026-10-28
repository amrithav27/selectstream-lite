from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from main.model.importance_scorer import ImportanceScorer


@dataclass
class LVMWriteSummary:
    num_segments: int
    num_new: int
    num_update: int
    num_merge: int
    merge_penalty_total: float
    merge_penalty_similarity: float
    merge_penalty_surprise: float
    merge_penalty_access: float
    merge_penalty_recency: float
    usage_ratio: float


WRITE_SIGNALS = ("surprise", "learned", "oracle", "fixed")


@dataclass
class SAWStreamState:
    prev_feat: torch.Tensor
    prev_attn: torch.Tensor
    prev_ema: torch.Tensor
    # Running state of the write signal w_t (Q1): equals the surprise EMA in the "surprise" arm.
    # `recent_ema`, `energy` and the boundary rule (Eq. 3) all act on w_t.
    prev_w_ema: torch.Tensor
    recent_ema: torch.Tensor
    seg_sum: torch.Tensor
    seg_feats: torch.Tensor
    seg_surprise_sum: torch.Tensor
    seg_w_sum: torch.Tensor
    seg_count: torch.Tensor
    seg_start_step: torch.Tensor
    energy: torch.Tensor
    has_prev: torch.Tensor
    # Per memory slot: mean frozen feature g of the frames it holds, and how many frames. The
    # update-vs-new rule compares in this space (fixed during training), not z against slots h.
    node_g_mean: Optional[torch.Tensor] = None
    node_g_count: Optional[torch.Tensor] = None

    def to(self, device: torch.device, dtype: torch.dtype) -> "SAWStreamState":
        return SAWStreamState(
            prev_feat=self.prev_feat.to(device=device, dtype=dtype),
            prev_attn=self.prev_attn.to(device=device, dtype=dtype),
            prev_ema=self.prev_ema.to(device=device, dtype=dtype),
            prev_w_ema=self.prev_w_ema.to(device=device, dtype=dtype),
            recent_ema=self.recent_ema.to(device=device, dtype=dtype),
            seg_sum=self.seg_sum.to(device=device, dtype=dtype),
            seg_feats=self.seg_feats.to(device=device, dtype=dtype),
            seg_surprise_sum=self.seg_surprise_sum.to(device=device, dtype=dtype),
            seg_w_sum=self.seg_w_sum.to(device=device, dtype=dtype),
            seg_count=self.seg_count.to(device=device),
            seg_start_step=self.seg_start_step.to(device=device, dtype=dtype),
            energy=self.energy.to(device=device, dtype=dtype),
            has_prev=self.has_prev.to(device=device),
            node_g_mean=self.node_g_mean.to(device=device, dtype=dtype) if self.node_g_mean is not None else None,
            node_g_count=self.node_g_count.to(device=device, dtype=dtype) if self.node_g_count is not None else None,
        )


@dataclass
class LVMMemoryState:
    slots: torch.Tensor
    active: torch.Tensor
    age: torch.Tensor
    time_start: torch.Tensor
    time_end: torch.Tensor
    avg_surprise: torch.Tensor
    avg_importance: torch.Tensor
    read_count: torch.Tensor
    write_count: torch.Tensor
    merge_count: torch.Tensor
    temporal_graph: torch.Tensor
    similarity_graph: torch.Tensor
    last_graph_slot: torch.Tensor
    next_step: float = 0.0
    saw_state: Optional[SAWStreamState] = None

    def to(self, device: torch.device, dtype: torch.dtype) -> "LVMMemoryState":
        return LVMMemoryState(
            slots=self.slots.to(device=device, dtype=dtype),
            active=self.active.to(device=device),
            age=self.age.to(device=device),
            time_start=self.time_start.to(device=device, dtype=dtype),
            time_end=self.time_end.to(device=device, dtype=dtype),
            avg_surprise=self.avg_surprise.to(device=device, dtype=dtype),
            avg_importance=self.avg_importance.to(device=device, dtype=dtype),
            read_count=self.read_count.to(device=device, dtype=dtype),
            write_count=self.write_count.to(device=device, dtype=dtype),
            merge_count=self.merge_count.to(device=device, dtype=dtype),
            temporal_graph=self.temporal_graph.to(device=device, dtype=dtype),
            similarity_graph=self.similarity_graph.to(device=device, dtype=dtype),
            last_graph_slot=self.last_graph_slot.to(device=device),
            next_step=float(self.next_step),
            saw_state=self.saw_state.to(device=device, dtype=dtype) if self.saw_state is not None else None,
        )


class LVMMemoryFormer(nn.Module):
    """
    Trainable latent visual memory writer for SFT.

    Input:
        H: [B, T, D] visual/text hidden states
    Output:
        M: [B, mem_len, D] memory tokens
    """

    def __init__(
        self,
        hidden_size: int,
        mem_len: int,
        num_slots: int = 256,
        segment_len: int = 32,
        tau_r: float = 0.75,
        tau_s: float = 0.35,
        gate_hidden: int = 256,
        normalize_slot: bool = True,
        eviction_policy: str = "fifo",
        readout_policy: str = "recent",
        delta_t_scale: float = 32.0,
        enable_graph_merge: bool = True,
        merge_similarity_weight: float = 1.0,
        merge_surprise_weight: float = 0.5,
        merge_access_weight: float = 0.25,
        merge_recency_weight: float = 0.25,
        segment_encoder_layers: int = 1,
        segment_encoder_heads: int = 4,
        segment_time_pos: bool = True,
        use_saw: bool = False,
        saw_lambda_attn: float = 0.5,
        saw_ema_decay: float = 0.9,
        saw_l_min: int = 8,
        saw_l_max: int = 64,
        saw_energy_budget: float = 8.0,
        saw_recent_window: int = 64,
        saw_quantile: float = 0.9,
        saw_attn_groups: int = 32,
        saw_use_attn_proxy: bool = True,
        graph_temporal_weight_c: float = 1.0,
        graph_sim_topk: int = 4,
        write_signal: str = "surprise",
        fixed_segment_len: int = 16,
        use_importance_scorer: bool = False,
        scorer_proj_dim: int = 16,
    ):
        super().__init__()
        if write_signal not in WRITE_SIGNALS:
            raise ValueError(f"`write_signal` must be one of {WRITE_SIGNALS}, got {write_signal!r}.")
        if write_signal == "learned" and not use_importance_scorer:
            raise ValueError("write_signal='learned' needs use_importance_scorer=True.")
        if not (1 <= fixed_segment_len <= saw_l_max):
            raise ValueError(f"`fixed_segment_len` must be in [1, saw_l_max], got {fixed_segment_len}.")
        if num_slots <= 0:
            raise ValueError(f"`num_slots` must be > 0, got {num_slots}.")
        if mem_len <= 0:
            raise ValueError(f"`mem_len` must be > 0, got {mem_len}.")
        if segment_len <= 0:
            raise ValueError(f"`segment_len` must be > 0, got {segment_len}.")
        if eviction_policy not in {"fifo", "low_surprise", "low_access"}:
            raise ValueError(f"Unsupported eviction policy: {eviction_policy}")
        if readout_policy not in {"recent", "surprise", "hybrid"}:
            raise ValueError(f"Unsupported readout policy: {readout_policy}")
        if delta_t_scale <= 0:
            raise ValueError(f"`delta_t_scale` must be > 0, got {delta_t_scale}.")
        if segment_encoder_layers <= 0:
            raise ValueError(f"`segment_encoder_layers` must be > 0, got {segment_encoder_layers}.")
        if segment_encoder_heads <= 0:
            raise ValueError(f"`segment_encoder_heads` must be > 0, got {segment_encoder_heads}.")
        if not (0.0 <= saw_lambda_attn <= 1.0):
            raise ValueError(f"`saw_lambda_attn` must be in [0,1], got {saw_lambda_attn}.")
        if not (0.0 <= saw_ema_decay < 1.0):
            raise ValueError(f"`saw_ema_decay` must be in [0,1), got {saw_ema_decay}.")
        if saw_l_min <= 0:
            raise ValueError(f"`saw_l_min` must be > 0, got {saw_l_min}.")
        if saw_l_max < saw_l_min:
            raise ValueError(f"`saw_l_max` must be >= `saw_l_min`, got {saw_l_max} < {saw_l_min}.")
        if saw_energy_budget <= 0:
            raise ValueError(f"`saw_energy_budget` must be > 0, got {saw_energy_budget}.")
        if saw_recent_window <= 0:
            raise ValueError(f"`saw_recent_window` must be > 0, got {saw_recent_window}.")
        if not (0.0 <= saw_quantile <= 1.0):
            raise ValueError(f"`saw_quantile` must be in [0,1], got {saw_quantile}.")
        if saw_attn_groups <= 0:
            raise ValueError(f"`saw_attn_groups` must be > 0, got {saw_attn_groups}.")
        if graph_sim_topk <= 0:
            raise ValueError(f"`graph_sim_topk` must be > 0, got {graph_sim_topk}.")

        self.hidden_size = hidden_size
        self.mem_len = mem_len
        self.num_slots = num_slots
        self.segment_len = segment_len
        self.tau_r = tau_r
        self.tau_s = tau_s
        self.normalize_slot = normalize_slot
        self.eviction_policy = eviction_policy
        self.readout_policy = readout_policy
        self.delta_t_scale = float(delta_t_scale)
        self.enable_graph_merge = bool(enable_graph_merge)
        self.merge_similarity_weight = float(merge_similarity_weight)
        self.merge_surprise_weight = float(merge_surprise_weight)
        self.merge_access_weight = float(merge_access_weight)
        self.merge_recency_weight = float(merge_recency_weight)
        self.segment_encoder_layers = int(segment_encoder_layers)
        self.segment_encoder_heads = int(segment_encoder_heads)
        self.segment_time_pos = bool(segment_time_pos)
        self.use_saw = use_saw
        self.saw_lambda_attn = float(saw_lambda_attn)
        self.saw_ema_decay = float(saw_ema_decay)
        self.saw_l_min = int(saw_l_min)
        self.saw_l_max = int(saw_l_max)
        self.saw_energy_budget = float(saw_energy_budget)
        self.saw_recent_window = int(saw_recent_window)
        self.saw_quantile = float(saw_quantile)
        self.saw_attn_groups = int(saw_attn_groups)
        self.saw_use_attn_proxy = bool(saw_use_attn_proxy)
        self.graph_temporal_weight_c = float(graph_temporal_weight_c)
        self.graph_sim_topk = int(graph_sim_topk)
        self.write_signal = write_signal
        self.fixed_segment_len = int(fixed_segment_len)
        self.importance_scorer = ImportanceScorer(hidden_size, proj_dim=scorer_proj_dim) if use_importance_scorer else None
        self.last_summary: Optional[LVMWriteSummary] = None
        self.last_states: Optional[list[Dict[str, torch.Tensor]]] = None

        # Learnable initial slot bank. The runtime state holds N + 1 slots: a new
        # node is written first and consolidation then restores |M| <= N
        # (Algorithm 1), so one extra slot is needed as write headroom.
        self.slot_init = nn.Parameter(torch.randn(num_slots, hidden_size) * 0.02)
        self.slot_overflow_init = nn.Parameter(torch.randn(1, hidden_size) * 0.02)

        # f_write(z_j, h_i*)
        self.write_mlp = nn.Sequential(
            nn.Linear(hidden_size * 2, gate_hidden),
            nn.GELU(),
            nn.Linear(gate_hidden, hidden_size),
        )

        # alpha = sigma(MLP([z_j; h_i*; surprise_j; delta_t]))
        self.gate_mlp = nn.Sequential(
            nn.Linear(hidden_size * 2 + 2, gate_hidden),
            nn.GELU(),
            nn.Linear(gate_hidden, 1),
        )
        seg_encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=self.segment_encoder_heads,
            dim_feedforward=hidden_size * 4,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.segment_encoder = nn.TransformerEncoder(seg_encoder_layer, num_layers=self.segment_encoder_layers)
        self.segment_query = nn.Parameter(torch.randn(1, 1, hidden_size) * 0.02)
        self.segment_ln = nn.LayerNorm(hidden_size)

    def _segment_relative_pos(self, length: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if length <= 0:
            return torch.zeros(1, 0, self.hidden_size, device=device, dtype=dtype)
        if length == 1:
            rel = torch.zeros(1, device=device, dtype=dtype)
        else:
            rel = torch.linspace(0.0, 1.0, steps=length, device=device, dtype=dtype)
        freqs = torch.arange(1, self.hidden_size + 1, device=device, dtype=dtype)
        freqs = freqs / float(max(1, self.hidden_size))
        phase = rel.unsqueeze(1) * freqs.unsqueeze(0) * 3.141592653589793
        return torch.sin(phase).unsqueeze(0)

    def _encode_segment_sequence(self, seq: torch.Tensor) -> torch.Tensor:
        # seq: [L, D]
        if seq.dim() != 2:
            raise ValueError(f"`seq` must be [L,D], got {tuple(seq.shape)}")
        if seq.size(0) == 0:
            return seq.new_zeros(self.hidden_size)

        x = seq.unsqueeze(0)
        if self.segment_time_pos:
            x = x + self._segment_relative_pos(seq.size(0), device=seq.device, dtype=seq.dtype)

        q = self.segment_query.to(device=seq.device, dtype=seq.dtype)
        k = self.segment_encoder(x)
        attn = torch.softmax((q @ k.transpose(-1, -2)) / (self.hidden_size ** 0.5), dim=-1)
        pooled = attn @ k
        pooled = self.segment_ln(pooled).squeeze(0).squeeze(0)
        return pooled

    def _fresh_saw_state(self, device: torch.device, dtype: torch.dtype) -> SAWStreamState:
        return SAWStreamState(
            prev_feat=torch.zeros(self.hidden_size, device=device, dtype=dtype),
            prev_attn=torch.zeros(self.saw_attn_groups, device=device, dtype=dtype),
            prev_ema=torch.zeros((), device=device, dtype=dtype),
            prev_w_ema=torch.zeros((), device=device, dtype=dtype),
            recent_ema=torch.empty(0, device=device, dtype=dtype),
            seg_sum=torch.zeros(self.hidden_size, device=device, dtype=dtype),
            seg_feats=torch.empty(0, self.hidden_size, device=device, dtype=dtype),
            seg_surprise_sum=torch.zeros((), device=device, dtype=dtype),
            seg_w_sum=torch.zeros((), device=device, dtype=dtype),
            seg_count=torch.zeros((), device=device, dtype=torch.long),
            seg_start_step=torch.full((), -1.0, device=device, dtype=dtype),
            energy=torch.zeros((), device=device, dtype=dtype),
            has_prev=torch.zeros((), device=device, dtype=torch.bool),
            node_g_mean=torch.zeros(self.capacity, self.hidden_size, device=device, dtype=dtype),
            node_g_count=torch.zeros(self.capacity, device=device, dtype=dtype),
        )

    @property
    def capacity(self) -> int:
        return self.num_slots + 1

    def _initial_slots(self) -> torch.Tensor:
        return torch.cat([self.slot_init, self.slot_overflow_init], dim=0)

    def _fresh_state(self, device: torch.device, dtype: torch.dtype) -> LVMMemoryState:
        slots = self._initial_slots().clone().to(device=device, dtype=dtype)
        if self.normalize_slot:
            slots = F.normalize(slots, dim=-1)
        cap = self.capacity
        return LVMMemoryState(
            slots=slots,
            active=torch.zeros(cap, device=device, dtype=torch.bool),
            age=torch.full((cap,), -1.0, device=device, dtype=dtype),
            time_start=torch.full((cap,), -1.0, device=device, dtype=dtype),
            time_end=torch.full((cap,), -1.0, device=device, dtype=dtype),
            avg_surprise=torch.zeros(cap, device=device, dtype=dtype),
            avg_importance=torch.zeros(cap, device=device, dtype=dtype),
            read_count=torch.zeros(cap, device=device, dtype=dtype),
            write_count=torch.zeros(cap, device=device, dtype=dtype),
            merge_count=torch.zeros(cap, device=device, dtype=dtype),
            temporal_graph=torch.zeros(cap, cap, device=device, dtype=dtype),
            similarity_graph=torch.zeros(cap, cap, device=device, dtype=dtype),
            last_graph_slot=torch.full((), -1, device=device, dtype=torch.long),
            next_step=0.0,
            saw_state=self._fresh_saw_state(device=device, dtype=dtype) if self.use_saw else None,
        )

    def init_memory_state(
        self,
        batch_size: int = 1,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> List[LVMMemoryState]:
        if batch_size <= 0:
            raise ValueError(f"`batch_size` must be > 0, got {batch_size}.")
        device = device or self.slot_init.device
        dtype = dtype or self.slot_init.dtype
        return [self._fresh_state(device=device, dtype=dtype) for _ in range(batch_size)]

    def _prepare_memory_state(
        self,
        memory_state: Optional[Sequence[LVMMemoryState] | LVMMemoryState],
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> List[LVMMemoryState]:
        if memory_state is None:
            return self.init_memory_state(batch_size=batch_size, device=device, dtype=dtype)
        if isinstance(memory_state, LVMMemoryState):
            states = [memory_state]
        else:
            states = list(memory_state)
        if len(states) != batch_size:
            raise ValueError(f"`memory_state` length must match batch size {batch_size}, got {len(states)}.")
        return [state.to(device=device, dtype=dtype) for state in states]

    def _attention_proxy_js(self, H: torch.Tensor) -> torch.Tensor:
        # H: [T, D] -> [T], JS between adjacent channel-attention proxies.
        t, d = H.shape
        if t <= 1:
            return H.new_zeros(t)

        groups = min(self.saw_attn_groups, d)
        chunk = max(d // groups, 1)
        groups = max(d // chunk, 1)
        used = groups * chunk

        # Groupwise absolute activation as a lightweight attention proxy.
        x = H[:, :used].abs().reshape(t, groups, chunk).mean(dim=-1)
        p = F.softmax(x[1:], dim=-1)
        q = F.softmax(x[:-1], dim=-1)
        m = 0.5 * (p + q)
        eps = 1e-6
        js = 0.5 * (p * (torch.log(p + eps) - torch.log(m + eps))).sum(dim=-1)
        js = js + 0.5 * (q * (torch.log(q + eps) - torch.log(m + eps))).sum(dim=-1)

        out = H.new_zeros(t)
        out[1:] = js
        return out

    def _compute_token_surprise(self, H: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # H: [T, D] -> (raw surprise [T], EMA surprise [T])
        t = H.size(0)
        if t <= 0:
            return H.new_zeros(1), H.new_zeros(1)

        s_feat = H.new_zeros(t)
        if t > 1:
            s_feat[1:] = 1.0 - F.cosine_similarity(H[1:], H[:-1], dim=-1)

        if self.saw_use_attn_proxy:
            s_attn = self._attention_proxy_js(H)
        else:
            s_attn = H.new_zeros(t)

        s = self.saw_lambda_attn * s_attn + (1.0 - self.saw_lambda_attn) * s_feat

        s_ema = H.new_zeros(t)
        s_ema[0] = s[0]
        for i in range(1, t):
            s_ema[i] = self.saw_ema_decay * s_ema[i - 1] + (1.0 - self.saw_ema_decay) * s[i]
        return s, s_ema

    def _chunk_attention_proxy(self, H: torch.Tensor) -> torch.Tensor:
        # H: [T, D] -> [G], a chunk-level proxy for CLS->patch attention.
        if H.dim() != 2:
            raise ValueError(f"`H` must be [T,D], got shape={tuple(H.shape)}.")
        t, d = H.shape
        if t <= 0 or d <= 0:
            return H.new_full((self.saw_attn_groups,), 1.0 / float(self.saw_attn_groups))

        groups = self.saw_attn_groups
        chunk = max((d + groups - 1) // groups, 1)
        used = groups * chunk

        pooled = H.abs().mean(dim=0)
        if pooled.numel() < used:
            pooled = F.pad(pooled, (0, used - pooled.numel()))
        else:
            pooled = pooled[:used]

        proxy = pooled.reshape(groups, chunk).mean(dim=-1)
        return F.softmax(proxy, dim=-1)

    @staticmethod
    def _js_divergence(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        eps = 1e-6
        m = 0.5 * (p + q)
        return 0.5 * (p * (torch.log(p + eps) - torch.log(m + eps))).sum() + 0.5 * (
            q * (torch.log(q + eps) - torch.log(m + eps))
        ).sum()

    def _compute_chunk_surprise(
        self,
        chunk_feat: torch.Tensor,
        chunk_attn: torch.Tensor,
        saw_state: SAWStreamState,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Eq. 2. Returns (s_feat, s_attn, s, s_ema)."""
        if not bool(saw_state.has_prev.item()):
            zero = chunk_feat.new_zeros(())
            return zero, zero, zero, zero

        s_feat = 1.0 - F.cosine_similarity(
            chunk_feat.unsqueeze(0),
            saw_state.prev_feat.unsqueeze(0),
            dim=-1,
        ).squeeze(0)
        if self.saw_use_attn_proxy:
            s_attn = self._js_divergence(chunk_attn, saw_state.prev_attn)
            s = self.saw_lambda_attn * s_attn + (1.0 - self.saw_lambda_attn) * s_feat
        else:
            # No attention term: s_t = s_feat, not (1 - lambda) * s_feat.
            s_attn = chunk_feat.new_zeros(())
            s = s_feat
        s_ema = self.saw_ema_decay * saw_state.prev_ema + (1.0 - self.saw_ema_decay) * s
        return s_feat, s_attn, s, s_ema

    def _build_segments_fixed(self, H: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Fixed-length fallback segmentation.
        t, d = H.shape
        _, s_ema = self._compute_token_surprise(H)
        segs = []
        seg_sur = []
        for i in range(0, t, self.segment_len):
            seg = H[i : i + self.segment_len]
            segs.append(self._encode_segment_sequence(seg))
            seg_sur.append(s_ema[i : i + self.segment_len].mean())
        if not segs:
            return H.new_zeros(1, d), H.new_zeros(1)
        return torch.stack(segs, dim=0), torch.stack(seg_sur, dim=0)

    def _build_segments_saw(self, H: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Surprise-driven Adaptive Windowing (SAW)
        t, d = H.shape
        _, s_ema = self._compute_token_surprise(H)

        segs = []
        seg_sur = []
        start = 0
        energy = 0.0

        for idx in range(t):
            energy += float(s_ema[idx].item())
            seg_len = idx - start + 1

            recent_start = max(0, idx - self.saw_recent_window + 1)
            recent = s_ema[recent_start : idx + 1]
            theta_high = float(torch.quantile(recent.float(), self.saw_quantile).item()) if recent.numel() > 0 else 0.0

            cond_jump = float(s_ema[idx].item()) > theta_high
            cond_energy = energy > self.saw_energy_budget
            cond_force = seg_len >= self.saw_l_max

            if seg_len >= self.saw_l_min and (cond_jump or cond_energy or cond_force):
                seg = H[start : idx + 1]
                segs.append(self._encode_segment_sequence(seg))
                seg_sur.append(s_ema[start : idx + 1].mean())
                start = idx + 1
                energy = 0.0

        if start < t:
            seg = H[start:t]
            segs.append(self._encode_segment_sequence(seg))
            seg_sur.append(s_ema[start:t].mean())

        if not segs:
            return H.new_zeros(1, d), H.new_zeros(1)
        return torch.stack(segs, dim=0), torch.stack(seg_sur, dim=0)

    def _build_segments(self, H: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.use_saw:
            return self._build_segments_saw(H)
        return self._build_segments_fixed(H)

    def _evict_index(
        self,
        active: torch.Tensor,
        age: torch.Tensor,
        avg_surprise: torch.Tensor,
        read_count: torch.Tensor,
    ) -> int:
        active_idx = active.nonzero(as_tuple=False).squeeze(-1)
        if active_idx.numel() == 0:
            return 0

        if self.eviction_policy == "fifo":
            local = torch.argmin(age[active_idx])
            return int(active_idx[local].item())
        if self.eviction_policy == "low_surprise":
            local = torch.argmin(avg_surprise[active_idx])
            return int(active_idx[local].item())
        # low_access
        local = torch.argmin(read_count[active_idx])
        return int(active_idx[local].item())

    def _merge_pair_penalty_components(
        self,
        sim: torch.Tensor,
        surprise_i: torch.Tensor,
        surprise_j: torch.Tensor,
        access_i: torch.Tensor,
        access_j: torch.Tensor,
        recency_i: torch.Tensor,
        recency_j: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # Eqs. (15)-(20): pi_uv = l_sim * p_sim + l_sup * p_sup + l_acc * p_acc + l_rec * p_rec.
        sim_penalty = 1.0 - 0.5 * (sim + 1.0)
        surprise_penalty = 0.5 * (surprise_i + surprise_j)
        access_penalty = 0.5 * (access_i + access_j)
        recency_penalty = 0.5 * (recency_i + recency_j)
        total_penalty = (
            self.merge_similarity_weight * sim_penalty
            + self.merge_surprise_weight * surprise_penalty
            + self.merge_access_weight * access_penalty
            + self.merge_recency_weight * recency_penalty
        )
        return total_penalty, sim_penalty, surprise_penalty, access_penalty, recency_penalty

    def _consolidate(
        self,
        slots: torch.Tensor,
        active: torch.Tensor,
        age: torch.Tensor,
        time_start: torch.Tensor,
        time_end: torch.Tensor,
        avg_surprise: torch.Tensor,
        read_count: torch.Tensor,
        write_count: torch.Tensor,
        merge_count: torch.Tensor,
        temporal_graph: torch.Tensor,
        similarity_graph: torch.Tensor,
        last_graph_slot: torch.Tensor,
        avg_importance: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        bool,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        # Called once a new node has pushed |M| to N + 1 (Algorithm 1, lines 13-15):
        # merge the pair with minimum penalty pi_uv and release one slot.
        active_idx = active.nonzero(as_tuple=False).squeeze(-1)
        merge_metrics = slots.new_zeros(5)
        merge_pair = torch.full((2,), -1, device=slots.device, dtype=torch.long)
        merged = self.enable_graph_merge and active_idx.numel() >= 2

        if merged:
            active_slots = F.normalize(slots[active_idx], dim=-1)
            sim = active_slots @ active_slots.t()
            upper = torch.triu(torch.ones_like(sim, dtype=torch.bool), diagonal=1)

            # Each statistic is divided by its maximum over the active nodes (Appendix B.2).
            # p_sup (Eq. 16) reads the arm's write signal; it equals avg_surprise in the surprise arm.
            surprise_norm = avg_importance[active_idx]
            if float(surprise_norm.max().item()) > 0:
                surprise_norm = surprise_norm / surprise_norm.max().clamp(min=1e-6)

            access_norm = read_count[active_idx]
            if float(access_norm.max().item()) > 0:
                access_norm = access_norm / access_norm.max().clamp(min=1e-6)

            recency_norm = time_end[active_idx].float()
            valid_recency = recency_norm >= 0
            if bool(valid_recency.any().item()):
                recency_shifted = recency_norm.clone()
                recency_shifted[~valid_recency] = recency_shifted[valid_recency].min()
                recency_shifted = recency_shifted - recency_shifted.min()
                if float(recency_shifted.max().item()) > 0:
                    recency_norm = recency_shifted / recency_shifted.max().clamp(min=1e-6)
                else:
                    recency_norm = recency_shifted
            else:
                recency_norm = torch.zeros_like(recency_norm)

            total_penalty, sim_penalty, surprise_penalty, access_penalty, recency_penalty = self._merge_pair_penalty_components(
                sim=sim,
                surprise_i=surprise_norm.unsqueeze(1),
                surprise_j=surprise_norm.unsqueeze(0),
                access_i=access_norm.unsqueeze(1),
                access_j=access_norm.unsqueeze(0),
                recency_i=recency_norm.unsqueeze(1),
                recency_j=recency_norm.unsqueeze(0),
            )
            penalty = total_penalty.masked_fill(~upper, float("inf"))
            flat_idx = torch.argmin(penalty)
            num_active = int(active_idx.numel())
            keep_local = int(flat_idx.item() // num_active)
            free_local = int(flat_idx.item() % num_active)
            keep_idx = int(active_idx[keep_local].item())
            free_idx = int(active_idx[free_local].item())
            merge_metrics = torch.stack(
                [
                    sim_penalty[keep_local, free_local],
                    surprise_penalty[keep_local, free_local],
                    access_penalty[keep_local, free_local],
                    recency_penalty[keep_local, free_local],
                    total_penalty[keep_local, free_local],
                ],
                dim=0,
            )

            keep_priority = read_count[keep_idx] + write_count[keep_idx]
            free_priority = read_count[free_idx] + write_count[free_idx]
            if float(free_priority.item()) > float(keep_priority.item()):
                keep_idx, free_idx = free_idx, keep_idx
            merge_pair = torch.tensor([keep_idx, free_idx], device=slots.device, dtype=torch.long)
        else:
            # Non-merging ablations (e.g. FIFO consolidation) evict one node instead.
            keep_idx = -1
            free_idx = self._evict_index(active, age, avg_surprise, read_count)

        slots = slots.clone()
        active = active.clone()
        age = age.clone()
        time_start = time_start.clone()
        time_end = time_end.clone()
        avg_surprise = avg_surprise.clone()
        avg_importance = avg_importance.clone()
        read_count = read_count.clone()
        write_count = write_count.clone()
        merge_count = merge_count.clone()

        if merged:
            # Eq. (21): write-count weighted centroid; Eqs. (22)-(27): metadata.
            keep_weight = write_count[keep_idx].clamp(min=1.0)
            free_weight = write_count[free_idx].clamp(min=1.0)
            merged_slot = (keep_weight * slots[keep_idx] + free_weight * slots[free_idx]) / (keep_weight + free_weight)
            if self.normalize_slot:
                merged_slot = F.normalize(merged_slot, dim=-1)
            slots[keep_idx] = merged_slot

            keep_start = float(time_start[keep_idx].item()) if float(time_start[keep_idx].item()) >= 0 else float(time_start[free_idx].item())
            free_start = float(time_start[free_idx].item()) if float(time_start[free_idx].item()) >= 0 else keep_start
            keep_end = float(time_end[keep_idx].item()) if float(time_end[keep_idx].item()) >= 0 else float(time_end[free_idx].item())
            free_end = float(time_end[free_idx].item()) if float(time_end[free_idx].item()) >= 0 else keep_end
            time_start[keep_idx] = min(keep_start, free_start)
            time_end[keep_idx] = max(keep_end, free_end)
            age[keep_idx] = max(age[keep_idx], age[free_idx])
            avg_surprise[keep_idx] = (keep_weight * avg_surprise[keep_idx] + free_weight * avg_surprise[free_idx]) / (keep_weight + free_weight)
            avg_importance[keep_idx] = (keep_weight * avg_importance[keep_idx] + free_weight * avg_importance[free_idx]) / (keep_weight + free_weight)
            read_count[keep_idx] = read_count[keep_idx] + read_count[free_idx]
            write_count[keep_idx] = write_count[keep_idx] + write_count[free_idx]
            merge_count[keep_idx] = merge_count[keep_idx] + merge_count[free_idx] + 1.0

            # Temporal edges are inherited with max aggregation.
            in_edges = torch.maximum(temporal_graph[:, keep_idx], temporal_graph[:, free_idx])
            out_edges = torch.maximum(temporal_graph[keep_idx], temporal_graph[free_idx])
            temporal_graph, similarity_graph = self._clear_graph_slot(
                temporal_graph=temporal_graph,
                similarity_graph=similarity_graph,
                slot_idx=keep_idx,
            )
            temporal_graph, similarity_graph = self._clear_graph_slot(
                temporal_graph=temporal_graph,
                similarity_graph=similarity_graph,
                slot_idx=free_idx,
            )
            temporal_graph[:, keep_idx] = in_edges
            temporal_graph[keep_idx] = out_edges
            temporal_graph[keep_idx, keep_idx] = 0
            temporal_graph[:, free_idx] = 0
            temporal_graph[free_idx] = 0
        else:
            temporal_graph, similarity_graph = self._clear_graph_slot(
                temporal_graph=temporal_graph,
                similarity_graph=similarity_graph,
                slot_idx=free_idx,
            )

        active[free_idx] = False
        age[free_idx] = -1.0
        time_start[free_idx] = -1.0
        time_end[free_idx] = -1.0
        avg_surprise[free_idx] = 0.0
        avg_importance[free_idx] = 0.0
        read_count[free_idx] = 0.0
        write_count[free_idx] = 0.0
        merge_count[free_idx] = 0.0
        reset_slot = self._initial_slots()[free_idx].to(device=slots.device, dtype=slots.dtype)
        if self.normalize_slot:
            reset_slot = F.normalize(reset_slot, dim=-1)
        slots[free_idx] = reset_slot

        if merged:
            # Similarity edges are recomputed from the merged latent state.
            active_after = active.nonzero(as_tuple=False).squeeze(-1)
            others = active_after[active_after != keep_idx]
            if others.numel() > 0:
                sims = F.cosine_similarity(slots[keep_idx].unsqueeze(0), slots[others], dim=-1)
                similarity_graph[keep_idx, others] = sims
                similarity_graph[others, keep_idx] = sims

        last_graph_slot = last_graph_slot.clone()
        if int(last_graph_slot.item()) == free_idx:
            last_graph_slot.fill_(keep_idx)

        return (
            slots,
            active,
            age,
            time_start,
            time_end,
            avg_surprise,
            read_count,
            write_count,
            merge_count,
            temporal_graph,
            similarity_graph,
            last_graph_slot,
            merged,
            merge_metrics,
            merge_pair,
            avg_importance,
        )

    def _clear_graph_slot(
        self,
        temporal_graph: torch.Tensor,
        similarity_graph: torch.Tensor,
        slot_idx: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        temporal_graph = temporal_graph.clone()
        similarity_graph = similarity_graph.clone()
        temporal_graph[slot_idx] = 0
        temporal_graph[:, slot_idx] = 0
        similarity_graph[slot_idx] = 0
        similarity_graph[:, slot_idx] = 0
        return temporal_graph, similarity_graph

    def _update_memory_graph(
        self,
        slots: torch.Tensor,
        active: torch.Tensor,
        temporal_graph: torch.Tensor,
        similarity_graph: torch.Tensor,
        last_graph_slot: torch.Tensor,
        target_idx: int,
        surprise: torch.Tensor,
        action: str,
        was_active: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if action == "new":
            temporal_graph, similarity_graph = self._clear_graph_slot(
                temporal_graph=temporal_graph,
                similarity_graph=similarity_graph,
                slot_idx=target_idx,
            )
        else:
            temporal_graph = temporal_graph.clone()
            similarity_graph = similarity_graph.clone()

        similarity_graph[target_idx] = 0
        similarity_graph[:, target_idx] = 0
        active_idx = active.nonzero(as_tuple=False).squeeze(-1)
        others = active_idx[active_idx != target_idx]
        if others.numel() > 0:
            sims = F.cosine_similarity(slots[target_idx].unsqueeze(0), slots[others], dim=-1)
            similarity_graph[target_idx, others] = sims
            similarity_graph[others, target_idx] = sims

        prev_slot = int(last_graph_slot.item())
        if prev_slot >= 0 and prev_slot != target_idx and bool(active[prev_slot].item()):
            temporal_graph[prev_slot, target_idx] = torch.exp(
                -self.graph_temporal_weight_c * surprise.to(dtype=temporal_graph.dtype)
            )

        last_graph_slot = last_graph_slot.clone()
        last_graph_slot.fill_(int(target_idx))
        return temporal_graph, similarity_graph, last_graph_slot

    def _export_active_graph(
        self,
        temporal_graph: torch.Tensor,
        similarity_graph: torch.Tensor,
        active_idx: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        device = temporal_graph.device
        num_active = int(active_idx.numel())
        if num_active == 0:
            return (
                torch.empty(2, 0, device=device, dtype=torch.long),
                torch.empty(0, device=device, dtype=torch.long),
                torch.empty(0, device=device, dtype=temporal_graph.dtype),
            )

        temporal_sub = temporal_graph[active_idx][:, active_idx]
        temporal_pairs = (temporal_sub > 0).nonzero(as_tuple=False)
        if temporal_pairs.numel() > 0:
            temporal_edge_index = temporal_pairs.t().contiguous()
            temporal_weight = temporal_sub[temporal_pairs[:, 0], temporal_pairs[:, 1]]
            temporal_type = torch.zeros(temporal_pairs.size(0), device=device, dtype=torch.long)
        else:
            temporal_edge_index = torch.empty(2, 0, device=device, dtype=torch.long)
            temporal_weight = torch.empty(0, device=device, dtype=temporal_graph.dtype)
            temporal_type = torch.empty(0, device=device, dtype=torch.long)

        if num_active <= 1:
            return temporal_edge_index, temporal_type, temporal_weight

        similarity_sub = similarity_graph[active_idx][:, active_idx].clone()
        similarity_sub.fill_diagonal_(-1e4)
        k = min(self.graph_sim_topk, max(1, num_active - 1))
        topv, topi = torch.topk(similarity_sub, k=k, dim=-1)
        sim_src = torch.arange(num_active, device=device).unsqueeze(1).expand(num_active, k).reshape(-1)
        sim_dst = topi.reshape(-1)
        sim_weight = topv.reshape(-1)
        sim_edge_index = torch.stack([sim_src, sim_dst], dim=0)
        sim_type = torch.ones(sim_edge_index.size(1), device=device, dtype=torch.long)

        edge_index = torch.cat([temporal_edge_index, sim_edge_index], dim=1)
        edge_type = torch.cat([temporal_type, sim_type], dim=0)
        edge_weight = torch.cat([temporal_weight, sim_weight], dim=0)
        return edge_index, edge_type, edge_weight

    def _write_one(
        self,
        slots: torch.Tensor,
        active: torch.Tensor,
        age: torch.Tensor,
        time_start: torch.Tensor,
        time_end: torch.Tensor,
        avg_surprise: torch.Tensor,
        read_count: torch.Tensor,
        write_count: torch.Tensor,
        merge_count: torch.Tensor,
        temporal_graph: torch.Tensor,
        similarity_graph: torch.Tensor,
        last_graph_slot: torch.Tensor,
        z: torch.Tensor,
        surprise: torch.Tensor,
        step: float,
        segment_start: Optional[float] = None,
        segment_end: Optional[float] = None,
        avg_importance: Optional[torch.Tensor] = None,
        write_value: Optional[torch.Tensor] = None,
        match_query: Optional[torch.Tensor] = None,
        match_bank: Optional[torch.Tensor] = None,
    ) -> tuple:
        """One segment write (Eq. 4) + edges + consolidation (Eq. 5).

        match_query / match_bank: the space the update-vs-new rule compares in (segment mean g vs each
        node's mean g). Without them the rule compares z with the slots h, which live in different
        spaces (pilot 2: cosine ~ 0, so no update could ever happen).

        `surprise` (segment mean of sbar) feeds the node's avg_surprise and its temporal edges, which
        the read path uses. `write_value` (segment mean of the arm's write signal w) drives the
        update-vs-create rule, the gate input and avg_importance (merge penalty). They are equal in
        the surprise arm. Returns the 16-tuple of `forward` state followed by avg_importance.
        """
        if avg_importance is None:
            avg_importance = avg_surprise
        if write_value is None:
            write_value = surprise
        active_idx = active.nonzero(as_tuple=False).squeeze(-1)
        action = "new"
        self._last_write = {"best_sim": None, "write_value": float(write_value), "action": "new"}
        seg_start = float(step if segment_start is None else segment_start)
        seg_end = float(step if segment_end is None else segment_end)
        if seg_end < seg_start:
            seg_start, seg_end = seg_end, seg_start

        target_idx = -1
        if active_idx.numel() > 0:
            if match_query is not None and match_bank is not None:
                sims = F.cosine_similarity(match_query.unsqueeze(0).to(match_bank.dtype), match_bank[active_idx], dim=-1)
            else:
                sims = F.cosine_similarity(z.unsqueeze(0), slots[active_idx], dim=-1)
            best_local = torch.argmax(sims)
            best_sim = sims[best_local]
            self._last_write["best_sim"] = float(best_sim)
            if bool((best_sim > self.tau_r) and (write_value < self.tau_s)):
                target_idx = int(active_idx[best_local].item())
                action = "update"
                self._last_write["action"] = "update"
        if action == "new":
            # |M| <= N before the write, so the N + 1 slot buffer always has a free slot.
            target_idx = int((~active).nonzero(as_tuple=False)[0].item())
        self._last_write["target"] = int(target_idx)

        if age[target_idx] < 0:
            delta_t = z.new_tensor(0.0)
        else:
            raw_delta_t = float(step - float(age[target_idx].item()))
            delta_t = z.new_tensor(raw_delta_t / self.delta_t_scale)
        was_active_target = bool(active[target_idx].item())

        # Eq. (4): gated write.
        h_old = slots[target_idx]
        gate_in = torch.cat([z, h_old, write_value.to(dtype=z.dtype).view(1), delta_t.view(1)], dim=0)
        alpha = torch.sigmoid(self.gate_mlp(gate_in)).squeeze(0)
        h_write = self.write_mlp(torch.cat([z, h_old], dim=0))
        h_new = (1.0 - alpha) * h_old + alpha * h_write
        if self.normalize_slot:
            h_new = F.normalize(h_new, dim=-1)

        slots = slots.clone()
        active = active.clone()
        age = age.clone()
        time_start = time_start.clone()
        time_end = time_end.clone()
        write_count = write_count.clone()
        merge_count = merge_count.clone()
        avg_surprise = avg_surprise.clone()
        avg_importance = avg_importance.clone()

        slots[target_idx] = h_new
        active[target_idx] = True
        age[target_idx] = step
        if action == "update" and time_start[target_idx] >= 0 and time_end[target_idx] >= 0:
            time_start[target_idx] = min(float(time_start[target_idx].item()), seg_start)
            time_end[target_idx] = max(float(time_end[target_idx].item()), seg_end)
        else:
            time_start[target_idx] = seg_start
            time_end[target_idx] = seg_end
            merge_count[target_idx] = 0.0
        write_count[target_idx] = write_count[target_idx] + 1.0
        wc = write_count[target_idx]
        avg_surprise[target_idx] = avg_surprise[target_idx] + (surprise - avg_surprise[target_idx]) / wc
        avg_importance[target_idx] = avg_importance[target_idx] + (write_value.to(dtype=avg_importance.dtype) - avg_importance[target_idx]) / wc
        temporal_graph, similarity_graph, last_graph_slot = self._update_memory_graph(
            slots=slots,
            active=active,
            temporal_graph=temporal_graph,
            similarity_graph=similarity_graph,
            last_graph_slot=last_graph_slot,
            target_idx=target_idx,
            surprise=surprise,
            action=action,
            was_active=was_active_target,
        )

        num_merge = 0
        merge_metrics = z.new_zeros(5)
        merge_pair = torch.full((2,), -1, device=z.device, dtype=torch.long)
        if int(active.sum().item()) > self.num_slots:
            (
                slots,
                active,
                age,
                time_start,
                time_end,
                avg_surprise,
                read_count,
                write_count,
                merge_count,
                temporal_graph,
                similarity_graph,
                last_graph_slot,
                merged,
                merge_metrics,
                merge_pair,
                avg_importance,
            ) = self._consolidate(
                slots=slots,
                active=active,
                age=age,
                time_start=time_start,
                time_end=time_end,
                avg_surprise=avg_surprise,
                read_count=read_count,
                write_count=write_count,
                merge_count=merge_count,
                temporal_graph=temporal_graph,
                similarity_graph=similarity_graph,
                last_graph_slot=last_graph_slot,
                avg_importance=avg_importance,
            )
            num_merge = 1 if merged else 0

        return (
            slots,
            active,
            age,
            time_start,
            time_end,
            avg_surprise,
            read_count,
            write_count,
            merge_count,
            temporal_graph,
            similarity_graph,
            last_graph_slot,
            action,
            num_merge,
            merge_metrics,
            merge_pair,
            avg_importance,
        )

    @staticmethod
    def _update_node_g(node_g_mean, node_g_count, target, action, seg_sum, seg_count, merge_pair):
        """Frame-weighted mean of g per node: set on new, accumulate on update, combine on merge."""
        g_mean, g_count = node_g_mean.detach().clone(), node_g_count.detach().clone()
        n = seg_count.to(dtype=g_count.dtype)
        seg_mean = (seg_sum.detach() / n.clamp(min=1.0)).to(g_mean.dtype)
        if action == "new":
            g_mean[target], g_count[target] = seg_mean, n
        else:
            c = g_count[target]
            g_mean[target] = (c * g_mean[target] + n * seg_mean) / (c + n).clamp(min=1.0)
            g_count[target] = c + n
        keep, free = int(merge_pair[0].item()), int(merge_pair[1].item())
        if keep >= 0 and free >= 0:
            ck, cf = g_count[keep], g_count[free]
            g_mean[keep] = (ck * g_mean[keep] + cf * g_mean[free]) / (ck + cf).clamp(min=1.0)
            g_count[keep], g_count[free] = ck + cf, 0.0
            g_mean[free] = 0.0
        return g_mean, g_count

    def _write_signal_step(
        self,
        saw_state: SAWStreamState,
        s_feat: torch.Tensor,
        s_attn: torch.Tensor,
        s_ema: torch.Tensor,
        chunk_feat: torch.Tensor,
        frame_signal: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Write signal w_t for this frame (Q1) and the scorer logit (always computed when a scorer exists).

        surprise: w_t = sbar_t (Eq. 2), so the paper is reproduced exactly.
        learned:  w_t = EMA_rho(sigmoid(scorer(phi_t))), scorer output detached.
        oracle:   w_t = EMA_rho(y_t), y_t passed in as `frame_signal`.
        fixed:    w_t = 0; cuts come from `fixed_segment_len` only.
        """
        logit = None
        if self.importance_scorer is not None:
            # No cut-dependent feature: w_t must depend on the frames only, so thresholds can be
            # calibrated by replaying Eq. 3 offline on a recorded w_t sequence.
            scalars = torch.stack([s_feat, s_ema]).to(dtype=chunk_feat.dtype)
            logit = self.importance_scorer(scalars, chunk_feat)

        if self.write_signal == "surprise":
            return s_ema, logit
        if self.write_signal == "fixed":
            return s_ema.new_zeros(()), logit
        if self.write_signal == "learned":
            raw = torch.sigmoid(logit.detach()).to(dtype=s_ema.dtype)
        else:
            if frame_signal is None:
                raise ValueError("write_signal='oracle' needs `frame_signal` (y_t) for every frame.")
            raw = frame_signal.detach().to(device=s_ema.device, dtype=s_ema.dtype).reshape(())
        w = self.saw_ema_decay * saw_state.prev_w_ema + (1.0 - self.saw_ema_decay) * raw
        return w, logit

    def _update_temporal_saw(
        self,
        H: torch.Tensor,
        saw_state: SAWStreamState,
        chunk_attn: Optional[torch.Tensor],
        slots: torch.Tensor,
        active: torch.Tensor,
        age: torch.Tensor,
        time_start: torch.Tensor,
        time_end: torch.Tensor,
        avg_surprise: torch.Tensor,
        avg_importance: torch.Tensor,
        read_count: torch.Tensor,
        write_count: torch.Tensor,
        merge_count: torch.Tensor,
        temporal_graph: torch.Tensor,
        similarity_graph: torch.Tensor,
        last_graph_slot: torch.Tensor,
        chunk_start: float,
        step: float,
        flush_saw: bool,
        frame_signal: Optional[torch.Tensor] = None,
    ) -> tuple:
        """One frame of Algorithm 1. Returns (next_saw_state, 13 memory tensors incl. avg_importance,
        num_segments, num_new, num_update, num_merge, merge_metrics, merge_pair, frame_info)."""
        chunk_feat = H.mean(dim=0)  # g_t = mean-pool of Phi_v(x_t)
        if chunk_attn is None:
            # Proxy off (default): no attention term at all, s_t = s_feat (docs/pilot.md §7).
            chunk_attn = self._chunk_attention_proxy(H) if self.saw_use_attn_proxy else H.new_zeros(self.saw_attn_groups)
        s_feat, s_attn, _, s_ema = self._compute_chunk_surprise(chunk_feat=chunk_feat, chunk_attn=chunk_attn, saw_state=saw_state)
        w, logit = self._write_signal_step(saw_state, s_feat, s_attn, s_ema, chunk_feat, frame_signal)

        recent_ema = torch.cat([saw_state.recent_ema, w.detach().reshape(1)], dim=0)
        if recent_ema.numel() > self.saw_recent_window:
            recent_ema = recent_ema[-self.saw_recent_window :]

        seg_sum = saw_state.seg_sum + chunk_feat
        seg_feats = torch.cat([saw_state.seg_feats, chunk_feat.unsqueeze(0)], dim=0)
        seg_surprise_sum = saw_state.seg_surprise_sum + s_ema
        seg_w_sum = saw_state.seg_w_sum + w
        seg_count = saw_state.seg_count + 1
        seg_start_step = saw_state.seg_start_step
        if int(saw_state.seg_count.item()) == 0:
            seg_start_step = chunk_feat.new_tensor(float(chunk_start))
        energy = saw_state.energy + w
        node_g_mean = saw_state.node_g_mean
        node_g_count = saw_state.node_g_count
        if node_g_mean is None:  # state from before this field existed
            node_g_mean = chunk_feat.new_zeros(self.capacity, self.hidden_size)
            node_g_count = chunk_feat.new_zeros(self.capacity)

        # Eq. 3 on the write signal w_t (identical to the paper when w_t = sbar_t).
        seg_len = int(seg_count.item())
        theta_high = torch.quantile(recent_ema.float(), self.saw_quantile).to(dtype=w.dtype)
        if self.write_signal == "fixed":
            cond_jump = cond_energy = False
            cond_force = seg_len >= self.fixed_segment_len
            should_cut = cond_force
        else:
            cond_jump = bool((w > theta_high).item())
            cond_energy = bool((energy > self.saw_energy_budget).item())
            cond_force = seg_len >= self.saw_l_max
            should_cut = seg_len >= self.saw_l_min and (cond_jump or cond_energy or cond_force)
        if flush_saw and seg_len > 0:
            should_cut = True

        num_segments = 0
        num_new = 0
        num_update = 0
        num_merge = 0
        merge_metrics = chunk_feat.new_zeros(5)
        merge_pair = torch.full((2,), -1, device=chunk_feat.device, dtype=torch.long)
        if should_cut:
            z = self._encode_segment_sequence(seg_feats)
            denom = seg_count.to(dtype=chunk_feat.dtype).clamp(min=1.0)
            (
                slots,
                active,
                age,
                time_start,
                time_end,
                avg_surprise,
                read_count,
                write_count,
                merge_count,
                temporal_graph,
                similarity_graph,
                last_graph_slot,
                action,
                num_merge,
                merge_metrics,
                merge_pair,
                avg_importance,
            ) = self._write_one(
                slots=slots,
                active=active,
                age=age,
                time_start=time_start,
                time_end=time_end,
                avg_surprise=avg_surprise,
                read_count=read_count,
                write_count=write_count,
                merge_count=merge_count,
                temporal_graph=temporal_graph,
                similarity_graph=similarity_graph,
                last_graph_slot=last_graph_slot,
                z=z,
                surprise=seg_surprise_sum / denom,
                step=step,
                segment_start=float(seg_start_step.item()) if float(seg_start_step.item()) >= 0 else step,
                segment_end=step,
                avg_importance=avg_importance,
                write_value=seg_w_sum / denom,
                match_query=seg_sum / denom,
                match_bank=node_g_mean,
            )
            node_g_mean, node_g_count = self._update_node_g(
                node_g_mean, node_g_count, self._last_write["target"], action, seg_sum, seg_count, merge_pair
            )
            num_segments = 1
            if action == "new":
                num_new = 1
            else:
                num_update = 1
            seg_sum = torch.zeros_like(seg_sum)
            seg_feats = seg_feats[:0]
            seg_surprise_sum = torch.zeros_like(seg_surprise_sum)
            seg_w_sum = torch.zeros_like(seg_w_sum)
            seg_count = torch.zeros_like(seg_count)
            seg_start_step = chunk_feat.new_tensor(-1.0)
            energy = torch.zeros_like(energy)

        next_saw_state = SAWStreamState(
            prev_feat=chunk_feat.detach(),
            prev_attn=chunk_attn.detach(),
            prev_ema=s_ema.detach(),
            prev_w_ema=w.detach(),
            recent_ema=recent_ema,
            seg_sum=seg_sum,
            seg_feats=seg_feats,
            seg_surprise_sum=seg_surprise_sum.detach(),
            seg_w_sum=seg_w_sum.detach(),
            seg_count=seg_count,
            seg_start_step=seg_start_step,
            energy=energy.detach(),
            has_prev=torch.ones_like(saw_state.has_prev),
            node_g_mean=node_g_mean.detach(),
            node_g_count=node_g_count.detach(),
        )
        frame_info = {
            "t": float(step),
            "s_feat": float(s_feat),
            "s_attn": float(s_attn),
            "s_ema": float(s_ema),
            "w": float(w),
            "theta_high": float(theta_high),
            "energy": float(saw_state.energy + w),
            "cut": bool(should_cut),
            # On a cut: cosine to the nearest node and update/new (tau_r, tau_s diagnostics, pilot §7).
            "write": dict(self._last_write) if should_cut else None,
            "num_merge": int(num_merge),
            "importance_logit": logit,
            "attn_proxy": chunk_attn.detach().float().cpu() if self.saw_use_attn_proxy else None,
        }
        return (
            next_saw_state,
            slots,
            active,
            age,
            time_start,
            time_end,
            avg_surprise,
            avg_importance,
            read_count,
            write_count,
            merge_count,
            temporal_graph,
            similarity_graph,
            last_graph_slot,
            num_segments,
            num_new,
            num_update,
            num_merge,
            merge_metrics,
            merge_pair,
            frame_info,
        )

    def _select_memory(
        self,
        slots: torch.Tensor,
        active: torch.Tensor,
        age: torch.Tensor,
        avg_surprise: torch.Tensor,
        read_count: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        active_idx = active.nonzero(as_tuple=False).squeeze(-1)
        if active_idx.numel() == 0:
            return (
                slots.new_zeros(self.mem_len, self.hidden_size),
                torch.empty(0, dtype=torch.long, device=slots.device),
            )

        if self.readout_policy == "recent":
            score = age[active_idx].float()
        elif self.readout_policy == "surprise":
            score = avg_surprise[active_idx]
        else:
            # hybrid: prefer recently updated + high-surprise slots.
            age_score = age[active_idx].float()
            age_score = age_score / age_score.max().clamp(min=1.0)
            surprise_score = avg_surprise[active_idx]
            read_penalty = read_count[active_idx] / read_count[active_idx].max().clamp(min=1.0)
            score = 0.5 * age_score + 0.5 * surprise_score - 0.2 * read_penalty

        order = torch.argsort(score, descending=True)
        picked = active_idx[order[: self.mem_len]]
        out = slots[picked]
        if out.size(0) < self.mem_len:
            pad = slots.new_zeros(self.mem_len - out.size(0), self.hidden_size)
            out = torch.cat([out, pad], dim=0)
        return out, picked

    def forward(
        self,
        H: torch.Tensor,
        Q: Optional[torch.Tensor] = None,
        memory_state: Optional[Sequence[LVMMemoryState] | LVMMemoryState] = None,
        return_memory_state: bool = False,
        step_stride: float = 1.0,
        flush_saw: bool = True,
        chunk_attention: Optional[torch.Tensor] = None,
        chunk_time_spans: Optional[torch.Tensor] = None,
        frame_signal: Optional[torch.Tensor] = None,
    ) -> torch.Tensor | tuple[torch.Tensor, List[LVMMemoryState]]:
        # Q is unused, kept for compatibility with former call signatures.
        # frame_signal: [B] oracle y_t for this frame (write_signal='oracle' only).
        if H.dim() != 3:
            raise ValueError(f"`H` must be [B,T,D], got shape={tuple(H.shape)}.")
        bsz, _, d = H.shape
        if d != self.hidden_size:
            raise ValueError(f"Hidden size mismatch: expected {self.hidden_size}, got {d}.")
        if step_stride <= 0:
            raise ValueError(f"`step_stride` must be > 0, got {step_stride}.")

        use_temporal_saw = self.use_saw and memory_state is not None
        runtime_states = self._prepare_memory_state(
            memory_state=memory_state,
            batch_size=bsz,
            device=H.device,
            dtype=H.dtype,
        )
        out_all = []
        summaries = []
        batch_states = []
        next_states = []
        for b in range(bsz):
            state = runtime_states[b]
            slots = state.slots
            active = state.active
            age = state.age
            time_start = state.time_start
            time_end = state.time_end
            avg_surprise = state.avg_surprise
            avg_importance = state.avg_importance
            read_count = state.read_count
            write_count = state.write_count
            merge_count = state.merge_count
            temporal_graph = state.temporal_graph
            similarity_graph = state.similarity_graph
            last_graph_slot = state.last_graph_slot
            start_step = float(state.next_step)
            frame_info = None
            num_new = 0
            num_update = 0
            num_segments = 0
            num_merge = 0
            merge_metric_sums = slots.new_zeros(5)
            last_merge_pair = torch.full((2,), -1, device=H.device, dtype=torch.long)
            saw_state = state.saw_state

            if use_temporal_saw and saw_state is not None:
                chunk_attn_b = None
                if chunk_attention is not None:
                    chunk_attn_b = chunk_attention[b]
                chunk_start = start_step
                chunk_end = start_step
                if chunk_time_spans is not None:
                    chunk_start = float(chunk_time_spans[b, 0].item())
                    chunk_end = float(chunk_time_spans[b, 1].item())
                (
                    saw_state,
                    slots,
                    active,
                    age,
                    time_start,
                    time_end,
                    avg_surprise,
                    avg_importance,
                    read_count,
                    write_count,
                    merge_count,
                    temporal_graph,
                    similarity_graph,
                    last_graph_slot,
                    num_segments,
                    num_new,
                    num_update,
                    num_merge,
                    merge_metrics,
                    merge_pair,
                    frame_info,
                ) = self._update_temporal_saw(
                    H=H[b],
                    saw_state=saw_state,
                    chunk_attn=chunk_attn_b,
                    slots=slots,
                    active=active,
                    age=age,
                    time_start=time_start,
                    time_end=time_end,
                    avg_surprise=avg_surprise,
                    avg_importance=avg_importance,
                    read_count=read_count,
                    write_count=write_count,
                    merge_count=merge_count,
                    temporal_graph=temporal_graph,
                    similarity_graph=similarity_graph,
                    last_graph_slot=last_graph_slot,
                    chunk_start=chunk_start,
                    step=chunk_end,
                    flush_saw=flush_saw,
                    frame_signal=frame_signal[b] if frame_signal is not None else None,
                )
                merge_metric_sums = merge_metric_sums + merge_metrics
                if num_merge > 0:
                    last_merge_pair = merge_pair
                if chunk_time_spans is not None:
                    next_step = chunk_end
                else:
                    next_step = start_step + float(step_stride)
            else:
                Z, S = self._build_segments(H[b])  # [S,D], [S]
                num_segments = int(Z.size(0))
                for local_step in range(Z.size(0)):
                    if chunk_time_spans is not None:
                        step = float(chunk_time_spans[b, 1].item())
                        seg_start = float(chunk_time_spans[b, 0].item())
                        seg_end = float(chunk_time_spans[b, 1].item())
                    else:
                        step = start_step + float(local_step) * float(step_stride)
                        seg_start = step
                        seg_end = step
                    (
                        slots,
                        active,
                        age,
                        time_start,
                        time_end,
                        avg_surprise,
                        read_count,
                        write_count,
                        merge_count,
                        temporal_graph,
                        similarity_graph,
                        last_graph_slot,
                        action,
                        merged_now,
                        merge_metrics,
                        merge_pair,
                        avg_importance,
                    ) = self._write_one(
                        slots=slots,
                        active=active,
                        age=age,
                        time_start=time_start,
                        time_end=time_end,
                        avg_surprise=avg_surprise,
                        read_count=read_count,
                        write_count=write_count,
                        merge_count=merge_count,
                        temporal_graph=temporal_graph,
                        similarity_graph=similarity_graph,
                        last_graph_slot=last_graph_slot,
                        z=Z[local_step],
                        surprise=S[local_step],
                        step=step,
                        segment_start=seg_start,
                        segment_end=seg_end,
                        avg_importance=avg_importance,
                    )
                    if action == "new":
                        num_new += 1
                    else:
                        num_update += 1
                    num_merge += merged_now
                    merge_metric_sums = merge_metric_sums + merge_metrics
                    if merged_now > 0:
                        last_merge_pair = merge_pair
                if chunk_time_spans is not None:
                    next_step = float(chunk_time_spans[b, 1].item())
                else:
                    next_step = start_step + float(num_segments) * float(step_stride)

            # Read counts are updated only when nodes are actually read at query
            # time (see `mark_read`), not by this legacy readout.
            mem_tokens, picked = self._select_memory(slots, active, age, avg_surprise, read_count)
            out_all.append(mem_tokens)
            new_state = LVMMemoryState(
                slots=slots,
                active=active,
                age=age,
                time_start=time_start,
                time_end=time_end,
                avg_surprise=avg_surprise,
                avg_importance=avg_importance,
                read_count=read_count,
                write_count=write_count,
                merge_count=merge_count,
                temporal_graph=temporal_graph,
                similarity_graph=similarity_graph,
                last_graph_slot=last_graph_slot,
                next_step=next_step,
                saw_state=saw_state,
            )
            next_states.append(new_state)

            batch_states.append(
                {
                    **self.state_view(new_state),
                    "frame_info": frame_info,
                    "selected_indices": picked,
                    "selected_slots": mem_tokens[: picked.size(0)] if picked.numel() > 0 else mem_tokens[:0],
                    "merge_count": torch.tensor(float(num_merge), device=H.device, dtype=H.dtype),
                    "merge_penalty_similarity": merge_metric_sums[0].clone(),
                    "merge_penalty_surprise": merge_metric_sums[1].clone(),
                    "merge_penalty_access": merge_metric_sums[2].clone(),
                    "merge_penalty_recency": merge_metric_sums[3].clone(),
                    "merge_penalty_total": merge_metric_sums[4].clone(),
                    "last_merge_pair": last_merge_pair.clone(),
                    "open_segment_len": saw_state.seg_count.clone() if saw_state is not None else torch.zeros((), device=H.device, dtype=torch.long),
                    "open_segment_start": saw_state.seg_start_step.clone() if saw_state is not None else torch.full((), -1.0, device=H.device, dtype=H.dtype),
                }
            )

            usage_ratio = float(active.float().sum().item()) / float(self.num_slots)
            merge_denom = max(1, num_merge)
            summaries.append(
                LVMWriteSummary(
                    num_segments=num_segments,
                    num_new=num_new,
                    num_update=num_update,
                    num_merge=num_merge,
                    merge_penalty_total=float((merge_metric_sums[4] / merge_denom).item()),
                    merge_penalty_similarity=float((merge_metric_sums[0] / merge_denom).item()),
                    merge_penalty_surprise=float((merge_metric_sums[1] / merge_denom).item()),
                    merge_penalty_access=float((merge_metric_sums[2] / merge_denom).item()),
                    merge_penalty_recency=float((merge_metric_sums[3] / merge_denom).item()),
                    usage_ratio=usage_ratio,
                )
            )

        if summaries:
            # Save average stats for logging in trainer.
            avg_seg = int(sum(s.num_segments for s in summaries) / len(summaries))
            avg_new = int(sum(s.num_new for s in summaries) / len(summaries))
            avg_update = int(sum(s.num_update for s in summaries) / len(summaries))
            avg_merge = int(sum(s.num_merge for s in summaries) / len(summaries))
            avg_merge_penalty_total = float(sum(s.merge_penalty_total for s in summaries) / len(summaries))
            avg_merge_penalty_similarity = float(sum(s.merge_penalty_similarity for s in summaries) / len(summaries))
            avg_merge_penalty_surprise = float(sum(s.merge_penalty_surprise for s in summaries) / len(summaries))
            avg_merge_penalty_access = float(sum(s.merge_penalty_access for s in summaries) / len(summaries))
            avg_merge_penalty_recency = float(sum(s.merge_penalty_recency for s in summaries) / len(summaries))
            avg_usage = float(sum(s.usage_ratio for s in summaries) / len(summaries))
            self.last_summary = LVMWriteSummary(
                num_segments=avg_seg,
                num_new=avg_new,
                num_update=avg_update,
                num_merge=avg_merge,
                merge_penalty_total=avg_merge_penalty_total,
                merge_penalty_similarity=avg_merge_penalty_similarity,
                merge_penalty_surprise=avg_merge_penalty_surprise,
                merge_penalty_access=avg_merge_penalty_access,
                merge_penalty_recency=avg_merge_penalty_recency,
                usage_ratio=avg_usage,
            )
        self.last_states = batch_states

        mem_tokens = torch.stack(out_all, dim=0)  # [B, mem_len, D]
        if return_memory_state:
            return mem_tokens, next_states
        return mem_tokens

    def _open_segment_node(self, state: LVMMemoryState) -> Optional[Dict[str, torch.Tensor]]:
        """The open (not yet committed) segment as the node a commit would create now: same segment
        encoder, gate and write MLP as a 'new' write (Eq. 4) into the first free slot. Nothing is
        stored. Lets a query asked while its evidence is still in the open segment read it."""
        saw = state.saw_state
        if saw is None or int(saw.seg_count.item()) == 0:
            return None
        z = self._encode_segment_sequence(saw.seg_feats)
        denom = saw.seg_count.to(dtype=z.dtype).clamp(min=1.0)
        write_value = (saw.seg_w_sum / denom).to(dtype=z.dtype)
        free = (~state.active).nonzero(as_tuple=False)
        h_old = state.slots[int(free[0].item())] if free.numel() > 0 else state.slots.new_zeros(state.slots.size(-1))
        gate_in = torch.cat([z, h_old, write_value.view(1), z.new_zeros(1)], dim=0)  # delta_t = 0, as for a fresh slot
        alpha = torch.sigmoid(self.gate_mlp(gate_in)).squeeze(0)
        h = (1.0 - alpha) * h_old + alpha * self.write_mlp(torch.cat([z, h_old], dim=0))
        if self.normalize_slot:
            h = F.normalize(h, dim=-1)
        return {
            "slot": h,
            "time_start": saw.seg_start_step.to(dtype=state.time_start.dtype),
            "surprise": (saw.seg_surprise_sum / denom).to(dtype=state.avg_surprise.dtype),
            "importance": write_value.to(dtype=state.avg_importance.dtype),
        }

    def state_view(self, state: LVMMemoryState, include_open: bool = False, t_now: Optional[float] = None) -> Dict[str, torch.Tensor]:
        """Nodes and edges of `state`, as consumed by the read path. Never writes.

        include_open: also expose the open segment as a provisional node spanning [segment start, t_now],
        index -1 (never marked read), with the edges a commit would add: a temporal edge from the last
        committed node and similarity edges to its top-k most similar nodes."""
        active_idx = state.active.nonzero(as_tuple=False).squeeze(-1)
        edge_index, edge_type, edge_weight = self._export_active_graph(
            temporal_graph=state.temporal_graph,
            similarity_graph=state.similarity_graph,
            active_idx=active_idx,
        )
        view = self._committed_view(state, active_idx, edge_index, edge_type, edge_weight)
        view["has_open_node"] = False
        if not include_open:
            return view
        node = self._open_segment_node(state)
        if node is None:
            return view
        if t_now is None:
            raise ValueError("include_open needs t_now (the current stream time) for the open node's end.")
        n = int(active_idx.numel())
        slots = view["active_slots"]
        new_edges, new_types, new_w = [], [], []
        last = int(state.last_graph_slot.item())
        if last >= 0 and bool(state.active[last].item()):
            src = int((active_idx == last).nonzero(as_tuple=False)[0].item())
            new_edges.append(torch.tensor([[src], [n]], device=edge_index.device))
            new_types.append(torch.zeros(1, dtype=torch.long, device=edge_index.device))
            new_w.append(torch.exp(-self.graph_temporal_weight_c * node["surprise"].detach()).reshape(1).to(edge_weight.dtype))
        if n > 0:
            sims = F.cosine_similarity(node["slot"].detach().unsqueeze(0), slots.detach(), dim=-1)
            k = min(self.graph_sim_topk, n)
            topv, topi = torch.topk(sims, k=k)
            open_col = torch.full_like(topi, n)
            new_edges.append(torch.stack([torch.cat([open_col, topi]), torch.cat([topi, open_col])], dim=0))
            new_types.append(torch.ones(2 * k, dtype=torch.long, device=edge_index.device))
            new_w.append(torch.cat([topv, topv]).to(edge_weight.dtype))
        dev, dt = state.time_start.device, state.time_start.dtype
        view.update({
            "active_indices": torch.cat([active_idx, active_idx.new_full((1,), -1)]),
            "active_slots": torch.cat([slots, node["slot"].unsqueeze(0).to(slots.dtype)], dim=0),
            "active_times": torch.cat([view["active_times"], torch.tensor([float(t_now)], device=dev, dtype=view["active_times"].dtype)]),
            "active_time_starts": torch.cat([view["active_time_starts"], node["time_start"].reshape(1)]),
            "active_time_ends": torch.cat([view["active_time_ends"], torch.tensor([float(t_now)], device=dev, dtype=dt)]),
            "active_surprise": torch.cat([view["active_surprise"], node["surprise"].reshape(1)]),
            "active_importance": torch.cat([view["active_importance"], node["importance"].reshape(1)]),
            "active_merge_count": torch.cat([view["active_merge_count"], view["active_merge_count"].new_zeros(1)]),
            "active_read_count": torch.cat([view["active_read_count"], view["active_read_count"].new_zeros(1)]),
            "graph_edge_index": torch.cat([edge_index] + new_edges, dim=1),
            "graph_edge_type": torch.cat([edge_type] + new_types),
            "graph_edge_weight": torch.cat([edge_weight] + new_w),
            "has_open_node": True,
        })
        return view

    def _committed_view(self, state, active_idx, edge_index, edge_type, edge_weight) -> Dict[str, torch.Tensor]:
        return {
            "active_indices": active_idx,
            "active_slots": state.slots[active_idx],
            "active_times": state.age[active_idx],
            "active_time_starts": state.time_start[active_idx],
            "active_time_ends": state.time_end[active_idx],
            "active_surprise": state.avg_surprise[active_idx],
            "active_importance": state.avg_importance[active_idx],
            "active_merge_count": state.merge_count[active_idx],
            "active_read_count": state.read_count[active_idx],
            "graph_edge_index": edge_index,
            "graph_edge_type": edge_type,
            "graph_edge_weight": edge_weight,
        }

    @staticmethod
    def mark_read(state: LVMMemoryState, slot_indices: torch.Tensor) -> None:
        """Increment read counts of nodes exposed as evidence for a query (Appendix B.2)."""
        slot_indices = slot_indices[slot_indices >= 0]  # -1 = open-segment node, not stored
        if slot_indices.numel() == 0:
            return
        read_count = state.read_count.clone()
        idx = slot_indices.to(device=read_count.device, dtype=torch.long)
        read_count[idx] = read_count[idx] + 1.0
        state.read_count = read_count

    def get_last_summary(self) -> Optional[Dict[str, float]]:
        if self.last_summary is None:
            return None
        return {
            "num_segments": float(self.last_summary.num_segments),
            "num_new": float(self.last_summary.num_new),
            "num_update": float(self.last_summary.num_update),
            "num_merge": float(self.last_summary.num_merge),
            "merge_penalty_total": float(self.last_summary.merge_penalty_total),
            "merge_penalty_similarity": float(self.last_summary.merge_penalty_similarity),
            "merge_penalty_surprise": float(self.last_summary.merge_penalty_surprise),
            "merge_penalty_access": float(self.last_summary.merge_penalty_access),
            "merge_penalty_recency": float(self.last_summary.merge_penalty_recency),
            "usage_ratio": float(self.last_summary.usage_ratio),
        }

    def get_last_state(self, batch_idx: int = 0) -> Optional[Dict[str, torch.Tensor]]:
        if self.last_states is None or batch_idx >= len(self.last_states):
            return None
        return self.last_states[batch_idx]
