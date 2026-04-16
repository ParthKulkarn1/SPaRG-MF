"""
maxformer_snn.py — Hierarchical Spiking Max-Former

Faithful reproduction of the NeurIPS 2025 Max-Former architecture with
SPaRG-MF enhancements (soft-sparse SSA, dynamic head/token gating,
mixed precision, homeostatic spike control).

Architecture:
  Stage 1 : Embed_Orig_ImageNet (÷4) → 1× Block_DWC7   (dim/4, 56×56)
  Stage 2 : Embed_Max (÷2)           → 2× Block_DWC5   (dim/2, 28×28)
  Stage 3 : Embed_Max (÷2)           → 7× Block_SSA    (dim,   14×14)
  Head    : LIF → Linear → temporal mean

Default config (maxformer_10_512):
  embed_dims=512, depths=[1,2,7], T=4, num_heads=8
"""

import torch
import torch.nn as nn
from spikingjelly.activation_based import neuron, surrogate, functional

from .patch_embed import Embed_Orig_ImageNet, Embed_Max_Stage
from .mixer_hub import Block_DWC, Block_SSA
from .dynamic_gate import DynamicHeadGate, DynamicTokenGate
from .mixed_precision import MixedPrecisionController
from .homeostasis import HomeostaticLIFNode


class SpikingMaxFormer(nn.Module):
    """
    Frequency-Preserved, Soft-Sparse, Dynamically Gated,
    Mixed-Precision, Homeostatically Stabilized Spiking Max-Former.

    Parameters
    ----------
    img_size : int          Input image size (assumed square).
    in_channels : int       Number of input channels (3 for RGB).
    num_classes : int       Number of classification targets.
    embed_dims : int        Final-stage embedding dimension.
    depths : list[int]      Number of blocks per stage [stage1, stage2, stage3].
    num_heads : int         Attention heads in stage-3 SSA blocks.
    mlp_ratio : float       MLP expansion ratio.
    time_steps : int        SNN simulation time steps.
    enable_head_gate : bool Enable dynamic head gating on SSA blocks.
    enable_token_gate : bool Enable dynamic token gating on SSA blocks.
    enable_mixed_prec : bool Enable mixed-precision routing.
    token_keep_ratio : float Minimum fraction of tokens to keep.
    """

    def __init__(
        self,
        img_size: int = 224,
        in_channels: int = 3,
        num_classes: int = 1000,
        embed_dims: int = 512,
        depths: list = None,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        time_steps: int = 4,
        enable_head_gate: bool = True,
        enable_token_gate: bool = True,
        enable_mixed_prec: bool = True,
        token_keep_ratio: float = 0.5,
    ):
        super().__init__()
        if depths is None:
            depths = [1, 2, 7]  # Max-Former default
        self.depths = depths
        self.time_steps = time_steps
        self.embed_dims = embed_dims
        self.num_heads = num_heads

        dim1 = embed_dims // 4
        dim2 = embed_dims // 2
        dim3 = embed_dims

        # ══════════════════════════════════════════════════════════════
        #  Stage 1: Embed_Orig_ImageNet → DWC-7 blocks  (dim/4)
        # ══════════════════════════════════════════════════════════════
        self.patch_embed1 = Embed_Orig_ImageNet(
            in_channels=in_channels, embed_dims=dim1)

        self.stage1 = nn.ModuleList([
            Block_DWC(dim=dim1, kernel_size=7, mlp_ratio=mlp_ratio)
            for _ in range(depths[0])
        ])

        # ══════════════════════════════════════════════════════════════
        #  Stage 2: Embed_Max (÷2) → DWC-5 blocks  (dim/2)
        # ══════════════════════════════════════════════════════════════
        self.patch_embed2 = Embed_Max_Stage(
            in_channels=dim1, embed_dims=dim2)

        self.stage2 = nn.ModuleList([
            Block_DWC(dim=dim2, kernel_size=5, mlp_ratio=mlp_ratio)
            for _ in range(depths[1])
        ])

        # ══════════════════════════════════════════════════════════════
        #  Stage 3: Embed_Max (÷2) → SSA blocks  (dim)
        # ══════════════════════════════════════════════════════════════
        self.patch_embed3 = Embed_Max_Stage(
            in_channels=dim2, embed_dims=dim3)

        self.stage3 = nn.ModuleList([
            Block_SSA(dim=dim3, num_heads=num_heads, mlp_ratio=mlp_ratio)
            for _ in range(depths[2])
        ])

        # ══════════════════════════════════════════════════════════════
        #  Classification head
        # ══════════════════════════════════════════════════════════════
        self.head_lif = neuron.LIFNode(
            tau=2.0, detach_reset=True,
            surrogate_function=surrogate.ATan(), step_mode='m')
        self.head = (nn.Linear(embed_dims, num_classes)
                     if num_classes > 0 else nn.Identity())

        # ══════════════════════════════════════════════════════════════
        #  SPaRG-MF Enhancements (applied to SSA stage only)
        # ══════════════════════════════════════════════════════════════
        n_ssa = depths[2]

        # Dynamic head gates (one per SSA block)
        self.head_gates = nn.ModuleList([
            DynamicHeadGate(dim3, num_heads)
            if enable_head_gate else None
            for _ in range(n_ssa)
        ])

        # Dynamic token gates (one per SSA block)
        self.token_gates = nn.ModuleList([
            DynamicTokenGate(dim3, keep_ratio=token_keep_ratio)
            if enable_token_gate else None
            for _ in range(n_ssa)
        ])

        # Mixed precision controller (shared)
        self.mixed_prec = (
            MixedPrecisionController() if enable_mixed_prec else None
        )

        # ── Weight init ──
        self.apply(self._init_weights)

    # ------------------------------------------------------------------
    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Conv2d):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor,
                external_head_masks: list = None) -> torch.Tensor:
        """
        Parameters
        ----------
        x : [B, C, H, W]  standard image batch.
        external_head_masks : list[Tensor] | None
            Per SSA-block gating masks from the SSA Interception Engine.
            Length must equal depths[2].

        Returns
        -------
        logits : [B, num_classes]
        """
        # ── Repeat over time ──
        if len(x.shape) < 5:
            x = x.unsqueeze(0).repeat(self.time_steps, 1, 1, 1, 1)
        else:
            x = x.transpose(0, 1).contiguous()

        # ── Stage 1 ──
        x = self.patch_embed1(x)               # [T, B, D/4, H/4, W/4]
        for blk in self.stage1:
            x = blk(x)

        # ── Stage 2 ──
        x = self.patch_embed2(x)               # [T, B, D/2, H/8, W/8]
        for blk in self.stage2:
            x = blk(x)

        # ── Stage 3 (SSA + SPaRG-MF gating) ──
        x = self.patch_embed3(x)               # [T, B, D, H/16, W/16]
        for i, blk in enumerate(self.stage3):
            # Determine head mask
            if external_head_masks is not None and external_head_masks[i] is not None:
                head_mask = external_head_masks[i]
            elif self.head_gates[i] is not None:
                # DynamicHeadGate expects [T, B, N, C] but we have [T, B, C, H, W]
                # Pool spatially first
                T, B, C, H, W = x.shape
                x_flat = x.flatten(3).transpose(2, 3)  # [T, B, N, C]
                head_mask = self.head_gates[i](x_flat)
            else:
                head_mask = None

            x = blk(x, head_mask=head_mask)

            # Token gating + mixed precision (on flattened token view)
            if self.token_gates[i] is not None:
                T, B, C, H, W = x.shape
                x_flat = x.flatten(3).transpose(2, 3)  # [T, B, N, C]
                token_mask = self.token_gates[i](x_flat)

                if self.mixed_prec is not None:
                    x_flat = self.mixed_prec(x_flat, token_mask)

                x_flat = x_flat * token_mask
                x = x_flat.transpose(2, 3).reshape(T, B, C, H, W)

        # ── Classification head ──
        x = x.flatten(3).mean(3)               # [T, B, D]  global avg pool
        x = self.head_lif(x)
        x = self.head(x)
        x = x.mean(0)                           # [B, num_classes]  temporal mean
        return x

    # ------------------------------------------------------------------
    def reset_snn_state(self):
        """Reset all spiking neuron membrane potentials."""
        functional.reset_net(self)

    # ------------------------------------------------------------------
    def count_spikes(self) -> dict:
        """Return spike-rate stats from all HomeostaticLIFNodes."""
        stats = {}
        for name, mod in self.named_modules():
            if isinstance(mod, HomeostaticLIFNode):
                stats[name] = {
                    'avg_spike_rate': mod.avg_spike_rate.item(),
                    'v_threshold': mod.lif.v_threshold,
                }
        return stats

    # ------------------------------------------------------------------
    @property
    def n_ssa_blocks(self) -> int:
        """Number of SSA blocks (= number of gating masks needed)."""
        return self.depths[2]


# ═══════════════════════════════════════════════════════════════════════
#  Factory functions (matching original Max-Former configs)
# ═══════════════════════════════════════════════════════════════════════

from timm.models.registry import register_model

@register_model
def sparg_mf_10_384(**kwargs):
    return SpikingMaxFormer(embed_dims=384, depths=[1, 2, 7], **kwargs)

@register_model
def sparg_mf_10_512(**kwargs):
    return SpikingMaxFormer(embed_dims=512, depths=[1, 2, 7], **kwargs)

@register_model
def sparg_mf_10_768(**kwargs):
    return SpikingMaxFormer(embed_dims=768, depths=[1, 2, 7], **kwargs)

