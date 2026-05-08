"""
mixer_hub.py — Token Mixing blocks from the Max-Former paper.

Three mixer types form the hierarchical backbone:
  Stage 1 → Block_DWC7 : 7×7 DepthWise-Conv local mixer (high-res)
  Stage 2 → Block_DWC5 : 5×5 DepthWise-Conv local mixer (mid-res)
  Stage 3 → Block_SSA  : full Spiking Self-Attention with our soft-sparse
                          temperature control and head-gating hooks

All blocks follow the pattern:  Mixer + S_MLP (Spiking MLP).
All tensors: [T, B, C, H, W].
"""

import torch
import torch.nn as nn
from spikingjelly.activation_based import neuron, surrogate
from .homeostasis import HomeostaticLIFNode


# ═══════════════════════════════════════════════════════════════════════
#  Spiking MLP  (shared feed-forward for every block type)
# ═══════════════════════════════════════════════════════════════════════
class S_MLP(nn.Module):
    """
    Two-layer spiking MLP using 1×1 convolutions (equivalent to Linear on
    spatial tokens). Membrane shortcut between the two layers if dims match.
    """

    def __init__(self, in_features, hidden_features=None, out_features=None):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.res = (in_features == hidden_features)

        self.fc1_conv = nn.Conv2d(in_features, hidden_features, 1)
        self.fc1_bn = nn.BatchNorm2d(hidden_features)
        self.fc1_lif = HomeostaticLIFNode(
            tau=2.0, target_rate=0.3, step_mode='m')

        self.fc2_conv = nn.Conv2d(hidden_features, out_features, 1)
        self.fc2_bn = nn.BatchNorm2d(out_features)
        self.fc2_lif = HomeostaticLIFNode(
            tau=2.0, target_rate=0.3, step_mode='m')

        self.c_hidden = hidden_features
        self.c_output = out_features

    def forward(self, x):
        T, B, C, H, W = x.shape
        identity = x

        x = self.fc1_lif(x)
        x = self.fc1_conv(x.flatten(0, 1))
        x = self.fc1_bn(x).reshape(T, B, self.c_hidden, H, W).contiguous()
        if self.res:
            x = identity + x
            identity = x

        x = self.fc2_lif(x)
        x = self.fc2_conv(x.flatten(0, 1))
        x = self.fc2_bn(x).reshape(T, B, C, H, W).contiguous()
        x = x + identity
        return x


# ═══════════════════════════════════════════════════════════════════════
#  DWC Mixers (local, high-frequency preserving)
# ═══════════════════════════════════════════════════════════════════════
class Mixer_DWC(nn.Module):
    """Depth-wise convolution token mixer with configurable kernel size."""

    def __init__(self, dim, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim, kernel_size,
                              padding=kernel_size // 2, groups=dim)
        self.conv_bn = nn.BatchNorm2d(dim)
        self.conv_neuron = neuron.LIFNode(
            tau=2.0, detach_reset=True,
            surrogate_function=surrogate.ATan(), step_mode='m')

    def forward(self, x):
        T, B, C, H, W = x.shape
        identity = x
        x = self.conv_neuron(x).reshape(T * B, -1, H, W).contiguous()
        x = self.conv(x)
        x = self.conv_bn(x).reshape(T, B, -1, H, W).contiguous()
        x = x + identity
        return x


class Block_DWC(nn.Module):
    """DWC Mixer + S_MLP block.  kernel_size controls the receptive field."""

    def __init__(self, dim, kernel_size=7, mlp_ratio=4.0):
        super().__init__()
        self.mixer = Mixer_DWC(dim, kernel_size=kernel_size)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = S_MLP(in_features=dim, hidden_features=mlp_hidden_dim)

    def forward(self, x):
        x = self.mixer(x)
        x = self.mlp(x)
        return x


# ═══════════════════════════════════════════════════════════════════════
#  Soft-Sparse Spiking Self-Attention (SSA) — our enhanced version
# ═══════════════════════════════════════════════════════════════════════
class SoftSparseSSA(nn.Module):
    """
    Spiking Self-Attention with:
      1. Separate Q, K, V Conv1d projections + BN + LIF  (following original SSA)
      2. Learnable temperature scaling for soft sparsity
      3. Optional external head_mask for hardware gating

    Attention formula (original Spikformer-style):
        attn = Q^T K   (spike-driven linear attention, NOT softmax)
        out  = (Q · attn) * scale

    We add temperature-controlled sharpening and head masking on top.
    """

    def __init__(self, dim, num_heads=8):
        super().__init__()
        assert dim % num_heads == 0
        self.dim = dim
        self.num_heads = num_heads
        self.scale = 0.125

        self.x_lif = neuron.LIFNode(
            tau=2.0, detach_reset=True,
            surrogate_function=surrogate.ATan(), step_mode='m')

        self.q_conv = nn.Conv1d(dim, dim, 1, bias=False)
        self.q_bn = nn.BatchNorm1d(dim)
        self.q_lif = neuron.LIFNode(
            tau=2.0, detach_reset=True,
            surrogate_function=surrogate.ATan(), step_mode='m')

        self.k_conv = nn.Conv1d(dim, dim, 1, bias=False)
        self.k_bn = nn.BatchNorm1d(dim)
        self.k_lif = neuron.LIFNode(
            tau=2.0, detach_reset=True,
            surrogate_function=surrogate.ATan(), step_mode='m')

        self.v_conv = nn.Conv1d(dim, dim, 1, bias=False)
        self.v_bn = nn.BatchNorm1d(dim)
        self.v_lif = neuron.LIFNode(
            tau=2.0, detach_reset=True,
            surrogate_function=surrogate.ATan(), step_mode='m')

        self.attn_lif = neuron.LIFNode(
            tau=2.0, v_threshold=0.5, detach_reset=True,
            surrogate_function=surrogate.ATan(), step_mode='m')

        self.proj_conv = nn.Conv1d(dim, dim, 1)
        self.proj_bn = nn.BatchNorm1d(dim)

        # ── SPaRG-MF addition: soft-sparse temperature ──
        self.temperature = nn.Parameter(torch.ones(1))

    def forward(self, x, head_mask=None):
        """
        x : [T, B, C, H, W]
        head_mask : optional [1, 1, num_heads, 1, 1]  binary mask
        Returns : [T, B, C, H, W]
        """
        T, B, C, H, W = x.shape
        identity = x

        x = self.x_lif(x)
        x = x.flatten(3).contiguous()          # [T, B, C, N]  where N = H*W
        N = x.shape[3]
        x_for_qkv = x.flatten(0, 1).contiguous()  # [T*B, C, N]

        # ── Q ──
        q = self.q_bn(self.q_conv(x_for_qkv)).reshape(T, B, C, N).contiguous()
        q = self.q_lif(q)
        q = q.transpose(-1, -2).reshape(
            T, B, N, self.num_heads, C // self.num_heads
        ).permute(0, 1, 3, 2, 4).contiguous()  # [T, B, H, N, D]

        # ── K ──
        k = self.k_bn(self.k_conv(x_for_qkv)).reshape(T, B, C, N).contiguous()
        k = self.k_lif(k)
        k = k.transpose(-1, -2).reshape(
            T, B, N, self.num_heads, C // self.num_heads
        ).permute(0, 1, 3, 2, 4).contiguous()                                  

        # ── V ──
        v = self.v_bn(self.v_conv(x_for_qkv)).reshape(T, B, C, N).contiguous()
        v = self.v_lif(v)
        v = v.transpose(-1, -2).reshape(
            T, B, N, self.num_heads, C // self.num_heads
        ).permute(0, 1, 3, 2, 4).contiguous()

        # ── Spike-driven linear attention ──
        attn = k.transpose(-2, -1) @ v                  # [T, B, H, D, D]
        x = (q @ attn) * self.scale                      # [T, B, H, N, D]

        # ── SPaRG-MF: temperature sharpening ──
        x = x / torch.clamp(self.temperature, min=1e-3)

        # ── SPaRG-MF: head gating ──
        if head_mask is not None:
            # head_mask: [1, 1, num_heads, 1, 1] → broadcast to [T, B, H, N, D]
            x = x * head_mask

        x = x.transpose(3, 4).reshape(T, B, C, N).contiguous()
        x = self.attn_lif(x)
        x = x.flatten(0, 1)
        x = self.proj_bn(self.proj_conv(x)).reshape(T, B, C, H, W)

        x = x + identity                                 # membrane shortcut
        return x


class Block_SSA(nn.Module):
    """SSA Mixer + S_MLP block."""

    def __init__(self, dim, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.attn = SoftSparseSSA(dim, num_heads=num_heads)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = S_MLP(in_features=dim, hidden_features=mlp_hidden_dim)

    def forward(self, x, head_mask=None):
        x = self.attn(x, head_mask=head_mask)
        x = self.mlp(x)
        return x
