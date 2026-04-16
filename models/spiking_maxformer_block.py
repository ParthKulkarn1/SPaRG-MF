import torch
import torch.nn as nn
from spikingjelly.activation_based import layer

from .soft_sparse_ssa import SpikingSoftSparseSelfAttention
from .homeostasis import HomeostaticLIFNode
from .dynamic_gate import DynamicHeadGate, DynamicTokenGate
from .mixed_precision import MixedPrecisionController


class SpikingMLP(nn.Module):
    """Spiking FeedForward with Homeostatic LIF neurons."""

    def __init__(self, dim: int, mlp_ratio: float = 4.0, drop: float = 0.0, tau: float = 2.0):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.fc1 = layer.Linear(dim, hidden)
        self.norm = nn.LayerNorm(hidden)
        self.lif1 = HomeostaticLIFNode(tau=tau, target_rate=0.3)
        self.fc2 = layer.Linear(hidden, dim)
        self.drop = layer.Dropout(drop)
        self.lif2 = HomeostaticLIFNode(tau=tau, target_rate=0.3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.norm(x)
        x = self.lif1(x)
        x = self.fc2(x)
        x = self.drop(x)
        x = self.lif2(x)
        return x


class SpikingMaxFormerBlock(nn.Module):
    """
    Full Spiking Max-Former block.

    Integrates:
      - Soft-Sparse SSA
      - Dynamic Head Gating
      - Dynamic Token Gating
      - Mixed Precision routing
      - Homeostatic LIF neurons (inside MLP)
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        tau: float = 2.0,
        enable_head_gate: bool = True,
        enable_token_gate: bool = True,
        enable_mixed_prec: bool = True,
        token_keep_ratio: float = 0.5,
    ):
        super().__init__()

        # ----- norms -----
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        # ----- core attention -----
        self.attn = SpikingSoftSparseSelfAttention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias,
            attn_drop=attn_drop, proj_drop=drop, tau=tau,
        )

        # ----- Spiking MLP with homeostasis -----
        self.mlp = SpikingMLP(dim, mlp_ratio=mlp_ratio, drop=drop, tau=tau)

        # ----- optional gating / precision modules -----
        self.head_gate = (
            DynamicHeadGate(dim, num_heads) if enable_head_gate else None
        )
        self.token_gate = (
            DynamicTokenGate(dim, keep_ratio=token_keep_ratio) if enable_token_gate else None
        )
        self.mixed_prec = (
            MixedPrecisionController() if enable_mixed_prec else None
        )

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor, external_head_mask=None) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor [Time, Batch, N_tokens, Channels]
        external_head_mask : optional mask from SSA Interception Engine
            Shape [1, 1, num_heads, 1, 1].  If provided it overrides the
            learned head gate (used at inference with hardware-deduced policy).

        Returns
        -------
        x : Tensor  same shape
        """
        # --- Head gating decision ---
        if external_head_mask is not None:
            head_mask = external_head_mask
        elif self.head_gate is not None:
            head_mask = self.head_gate(x)
        else:
            head_mask = None

        # --- Soft-Sparse SSA with residual ---
        x = x + self.attn(self.norm1(x), head_mask=head_mask)

        # --- Token gating (applied between attention and MLP) ---
        if self.token_gate is not None:
            token_mask = self.token_gate(x)  # [1, 1, N, 1]

            # Mixed-precision routing: use token_mask as saliency
            if self.mixed_prec is not None:
                x = self.mixed_prec(x, token_mask)

            # Apply token mask (soft during train, hard during eval)
            x = x * token_mask

        # --- Spiking MLP with residual ---
        x = x + self.mlp(self.norm2(x))

        return x
