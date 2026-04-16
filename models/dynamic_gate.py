import torch
import torch.nn as nn


class DynamicHeadGate(nn.Module):
    """
    Lightweight per-head gating controller.

    During **training** the gate outputs soft sigmoid scores so gradients flow.
    During **inference** the gate snaps to hard binary decisions derived from
    either its own learned parameters or from an externally supplied hardware
    mask (produced by the SSA Interception Engine).

    The gate takes a compact *summary* of the current input and emits one
    scalar per attention head indicating whether that head should be active.

    Parameters
    ----------
    dim : int
        Token embedding dimension (same as the transformer block width).
    num_heads : int
        Number of attention heads to gate.
    gate_hidden : int
        Hidden dim of the tiny gate MLP.  Default = dim // 4.
    hard_at_eval : bool
        If True, round the gate to {0, 1} at eval time.
    """

    def __init__(self, dim: int, num_heads: int, gate_hidden: int = 0, hard_at_eval: bool = True):
        super().__init__()
        self.num_heads = num_heads
        self.hard_at_eval = hard_at_eval
        gate_hidden = gate_hidden or max(dim // 4, num_heads)

        # Tiny MLP: global-average-pooled token features → per-head gate score
        self.gate_mlp = nn.Sequential(
            nn.Linear(dim, gate_hidden),
            nn.GELU(),
            nn.Linear(gate_hidden, num_heads),
        )

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor [Time, Batch, N_tokens, Channels]

        Returns
        -------
        head_mask : Tensor [1, 1, num_heads, 1, 1]
            Soft (training) or hard (eval) mask broadcastable over attention
            dimensions  [Time, Batch, Heads, Seq, Seq].
        """
        # Pool over Time and Tokens to create a compact representation
        # x shape: [T, B, N, C]  →  pooled: [B, C]
        pooled = x.mean(dim=(0, 2))  # [B, C]

        # Gate scores
        scores = self.gate_mlp(pooled)  # [B, num_heads]
        scores = torch.sigmoid(scores)  # soft in [0, 1]

        if not self.training and self.hard_at_eval:
            # Snap to binary for hardware-enforceable gating
            scores = (scores > 0.5).float()

        # Average over batch so we get a single policy
        scores = scores.mean(dim=0)  # [num_heads]

        # Reshape for broadcasting into attention weight tensor
        # Target: [1, 1, num_heads, 1, 1]
        head_mask = scores.view(1, 1, self.num_heads, 1, 1)
        return head_mask


class DynamicTokenGate(nn.Module):
    """
    Per-token gating controller.

    Emits a keep/drop score for every spatial token. Tokens below a
    confidence margin are masked out (set to zero) to save downstream
    compute.  During training the mask is soft; at eval it is hard.

    Parameters
    ----------
    dim : int
        Token embedding dimension.
    hard_at_eval : bool
        Round gate to {0,1} during eval.
    keep_ratio : float
        Minimum fraction of tokens to always keep (safety floor).
    """

    def __init__(self, dim: int, hard_at_eval: bool = True, keep_ratio: float = 0.5):
        super().__init__()
        self.hard_at_eval = hard_at_eval
        self.keep_ratio = keep_ratio

        self.gate_proj = nn.Sequential(
            nn.Linear(dim, dim // 4),
            nn.GELU(),
            nn.Linear(dim // 4, 1),
        )

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor [Time, Batch, N_tokens, Channels]

        Returns
        -------
        token_mask : Tensor [1, 1, N_tokens, 1]
            Broadcastable mask for element-wise multiply on the token tensor.
        """
        # Pool over Time → [B, N, C]
        pooled = x.mean(dim=0)

        scores = self.gate_proj(pooled).squeeze(-1)  # [B, N]
        scores = torch.sigmoid(scores)

        if not self.training and self.hard_at_eval:
            # Keep at least keep_ratio fraction of tokens
            N = scores.shape[-1]
            k = max(int(N * self.keep_ratio), 1)
            topk_vals, _ = scores.topk(k, dim=-1)
            threshold = topk_vals[:, -1:].detach()  # per-batch threshold
            scores = (scores >= threshold).float()

        # Average over batch, reshape for broadcast [1, 1, N, 1]
        token_mask = scores.mean(dim=0).unsqueeze(0).unsqueeze(0).unsqueeze(-1)
        return token_mask
