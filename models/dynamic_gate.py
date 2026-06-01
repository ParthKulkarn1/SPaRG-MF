import torch
import torch.nn as nn


class DynamicHeadGate(nn.Module):
    """
    Lightweight per-head gating controller.

    During **training** the gate outputs soft sigmoid scores so gradients flow.
    During **inference** the gate snaps to hard binary decisions derived from
    either its own learned parameters or from an externally supplied hardware
    mask (produced by the SSA Interception Engine).

    Parameters
    ----------
    dim : int
        Token embedding dimension (same as the transformer block width).
    num_heads : int
        Number of attention heads to gate.
    gate_hidden : int
        Hidden dim of the tiny gate MLP. Default = dim // 4.
    hard_at_eval : bool
        If True, round the gate to {0, 1} at eval time.
    batch_averaged : bool
        If True, average the head mask over the batch (static across batch).
        If False, allow each sample to gate its own heads dynamically (per-instance).
    time_dependent : bool
        If True, compute different gating masks for each simulation time step.
        If False, pool over time to produce a static temporal mask.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        gate_hidden: int = 0,
        hard_at_eval: bool = True,
        batch_averaged: bool = False,
        time_dependent: bool = False,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.hard_at_eval = hard_at_eval
        self.batch_averaged = batch_averaged
        self.time_dependent = time_dependent
        gate_hidden = gate_hidden or max(dim // 4, num_heads)

        # Tiny MLP: pooled token features → per-head gate score
        self.gate_mlp = nn.Sequential(
            nn.Linear(dim, gate_hidden),
            nn.GELU(),
            nn.Linear(gate_hidden, num_heads),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor [Time, Batch, N_tokens, Channels]

        Returns
        -------
        head_mask : Tensor [Time_dim, Batch_dim, num_heads, 1, 1]
            Soft (training) or hard (eval) mask broadcastable over attention
            dimensions [Time, Batch, Heads, Seq, Seq].
        """
        T, B, N, C = x.shape

        if self.time_dependent:
            # Pool over tokens: [T, B, C]
            pooled = x.mean(dim=2)
            # Flatten to [T*B, C] for MLP projection
            scores = self.gate_mlp(pooled.flatten(0, 1))  # [T*B, num_heads]
            scores = scores.view(T, B, self.num_heads)  # [T, B, num_heads]
            scores = torch.sigmoid(scores)

            if not self.training and self.hard_at_eval:
                scores = (scores > 0.5).float()

            if self.batch_averaged:
                scores = scores.mean(dim=1, keepdim=True)  # [T, 1, num_heads]
            
            # Reshape for attention broadcasting: [T, B, H, 1, 1]
            head_mask = scores.unsqueeze(-1).unsqueeze(-1)
        else:
            # Pool over Time and Tokens: [B, C]
            pooled = x.mean(dim=(0, 2))
            scores = self.gate_mlp(pooled)  # [B, num_heads]
            scores = torch.sigmoid(scores)

            if not self.training and self.hard_at_eval:
                scores = (scores > 0.5).float()

            if self.batch_averaged:
                scores = scores.mean(dim=0, keepdim=True)  # [1, num_heads]
                head_mask = scores.view(1, 1, self.num_heads, 1, 1)
            else:
                # [B, num_heads] -> [1, B, num_heads, 1, 1]
                head_mask = scores.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)

        return head_mask


class DynamicTokenGate(nn.Module):
    """
    Dynamic token gating controller.

    Supports:
      1. MLP: Learned tiny MLP projection.
      2. Magnitude: Parameter-free norm-based token saliency.
      3. Attention: Attention-weight driven token saliency.

    Parameters
    ----------
    dim : int
        Token embedding dimension.
    hard_at_eval : bool
        Round gate to {0,1} during eval.
    keep_ratio : float
        Minimum fraction of tokens to always keep (safety floor).
    gate_type : str
        Type of gating policy: 'mlp', 'magnitude', or 'attention'.
    batch_averaged : bool
        If True, average the token mask over the batch (static across batch).
        If False, allow each sample to gate its own tokens dynamically (per-instance).
    time_dependent : bool
        If True, compute different gating masks for each simulation time step.
        If False, pool over time to produce a static temporal mask.
    """

    def __init__(
        self,
        dim: int,
        hard_at_eval: bool = True,
        keep_ratio: float = 0.5,
        gate_type: str = 'mlp',
        batch_averaged: bool = False,
        time_dependent: bool = False,
    ):
        super().__init__()
        self.hard_at_eval = hard_at_eval
        self.keep_ratio = keep_ratio
        self.gate_type = gate_type.lower()
        self.batch_averaged = batch_averaged
        self.time_dependent = time_dependent

        assert self.gate_type in ['mlp', 'magnitude', 'attention'], \
            f"Unknown gate type: {gate_type}"

        if self.gate_type == 'mlp':
            self.gate_proj = nn.Sequential(
                nn.Linear(dim, dim // 4),
                nn.GELU(),
                nn.Linear(dim // 4, 1),
            )

    def forward(self, x: torch.Tensor, attn_weights: torch.Tensor = None) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor [Time, Batch, N_tokens, Channels]
        attn_weights : Tensor, optional
            Attention weights tensor from the block to guide attention gating.

        Returns
        -------
        token_mask : Tensor [Time_dim, Batch_dim, N_tokens, 1]
            Broadcastable mask for element-wise multiply on the token tensor.
        """
        T, B, N, C = x.shape

        # ── Step 1: Compute Saliency Scores ──
        if self.gate_type == 'mlp':
            if self.time_dependent:
                # [T*B, N, C]
                x_flat = x.flatten(0, 1)
                scores = self.gate_proj(x_flat).squeeze(-1)  # [T*B, N]
                scores = scores.view(T, B, N)
            else:
                pooled = x.mean(dim=0)  # [B, N, C]
                scores = self.gate_proj(pooled).squeeze(-1)  # [B, N]
            scores = torch.sigmoid(scores)

        elif self.gate_type == 'magnitude':
            if self.time_dependent:
                # L2 norm over channel dim per time step
                scores = x.norm(p=2, dim=-1)  # [T, B, N]
            else:
                pooled = x.mean(dim=0)  # [B, N, C]
                scores = pooled.norm(p=2, dim=-1)  # [B, N]
            
            # Normalize to [0, 1] softly for training gradients
            min_vals = scores.min(dim=-1, keepdim=True).values
            max_vals = scores.max(dim=-1, keepdim=True).values
            scores = (scores - min_vals) / (max_vals - min_vals + 1e-8)

        elif self.gate_type == 'attention':
            # Use attention weights if provided; fallback to magnitude if not
            if attn_weights is not None:
                # attn_weights shape is typically [T, B, H, N, N]
                # Sum attention received by each token (average over heads, sum over query dim)
                if len(attn_weights.shape) == 5:
                    scores = attn_weights.mean(dim=2).sum(dim=-2)  # [T, B, N]
                else:
                    scores = attn_weights.sum(dim=-1) # fallback
                
                if not self.time_dependent:
                    scores = scores.mean(dim=0)  # [B, N]
                
                # Normalize to [0, 1]
                min_vals = scores.min(dim=-1, keepdim=True).values
                max_vals = scores.max(dim=-1, keepdim=True).values
                scores = (scores - min_vals) / (max_vals - min_vals + 1e-8)
            else:
                # Fallback to magnitude
                if self.time_dependent:
                    scores = x.norm(p=2, dim=-1)
                else:
                    scores = x.mean(dim=0).norm(p=2, dim=-1)
                min_vals = scores.min(dim=-1, keepdim=True).values
                max_vals = scores.max(dim=-1, keepdim=True).values
                scores = (scores - min_vals) / (max_vals - min_vals + 1e-8)

        # ── Step 2: Apply Hard Gating threshold during Inference ──
        if not self.training and self.hard_at_eval:
            # Top-k selection along N dimension
            k = max(int(N * self.keep_ratio), 1)
            
            # scores shape could be [T, B, N] or [B, N]
            topk_vals, _ = scores.topk(k, dim=-1)
            threshold = topk_vals[..., -1:].detach()  # Threshold value (keep values >= this)
            scores = (scores >= threshold).float()

        # ── Step 3: Pool / Reshape to Target token_mask ──
        if self.time_dependent:
            if self.batch_averaged:
                # [T, B, N] -> average to [T, 1, N]
                scores = scores.mean(dim=1, keepdim=True)
                token_mask = scores.unsqueeze(-1)  # [T, 1, N, 1]
            else:
                token_mask = scores.unsqueeze(-1)  # [T, B, N, 1]
        else:
            if self.batch_averaged:
                # [B, N] -> average to [1, N] -> [1, 1, N, 1]
                scores = scores.mean(dim=0, keepdim=True)
                token_mask = scores.unsqueeze(0).unsqueeze(-1)
            else:
                # [B, N] -> [1, B, N, 1]
                token_mask = scores.unsqueeze(0).unsqueeze(-1)

        return token_mask
