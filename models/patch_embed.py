"""
patch_embed.py — Frequency-Preserving Patch Embeddings for Spiking Max-Former.

Implements the original Max-Former embedding strategy from the NeurIPS 2025
paper: multi-stage convolutional downsampling with MaxPool to preserve
high-frequency detail that spiking (LIF) neurons would otherwise suppress.

Three embedding modules are provided:
  - Embed_Orig_ImageNet : initial 224→56 stem (÷4)
  - Embed_Max           : inter-stage MaxPool downsampler (÷2)
  - Embed (helper)      : single Conv-BN stage with optional LIF

All tensors use shape convention [T, B, C, H, W] following SpikingJelly.
"""

import torch
import torch.nn as nn
from spikingjelly.activation_based import neuron, surrogate


# ═══════════════════════════════════════════════════════════════════════
#  Atomic building-block used by every embedding module
# ═══════════════════════════════════════════════════════════════════════
class Embed(nn.Module):
    """
    Single Conv-BN stage with optional pre-activation LIF.

    Parameters
    ----------
    in_channels, out_channels : int
    kernel_size, stride, padding : int
    shortcut : bool
        If True the input is already spiking, so skip the LIF pre-activation.
    """

    def __init__(self, in_channels, out_channels,
                 kernel_size=3, stride=1, padding=1, shortcut=False):
        super().__init__()
        self.shortcut = shortcut
        self.embed_lif = neuron.LIFNode(
            tau=2.0, detach_reset=True,
            surrogate_function=surrogate.ATan(), step_mode='m')
        self.embed_conv = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.embed_bn = nn.BatchNorm2d(out_channels)

    def forward(self, x, dual=False):
        """
        x : [T, B, C, H, W]
        Returns [T*B, C_out, H', W']   (flattened time-batch for next conv).
        If dual=True also returns the pre-conv spiking tensor for shortcuts.
        """
        if not self.shortcut:
            x = self.embed_lif(x)          # spike pre-activation

        x_feat = x                          # save for membrane shortcut
        T, B = x.shape[:2]
        x = self.embed_conv(x.flatten(0, 1))  # [T*B, C_out, H', W']
        x = self.embed_bn(x)

        if dual:
            return x, x_feat
        return x


# ═══════════════════════════════════════════════════════════════════════
#  Max-Pool Embed (high-freq preserving inter-stage downsampler)
# ═══════════════════════════════════════════════════════════════════════
class Max_Embed(nn.Module):
    """Conv-BN followed by 3×3 MaxPool (stride=2) for spatial downsample."""

    def __init__(self, in_channels, out_channels,
                 kernel_size=3, stride=1, padding=1, shortcut=False):
        super().__init__()
        self.shortcut = shortcut
        self.embed_lif = neuron.LIFNode(
            tau=2.0, detach_reset=True,
            surrogate_function=surrogate.ATan(), step_mode='m')
        self.embed_conv = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.embed_bn = nn.BatchNorm2d(out_channels)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x, dual=False):
        if not self.shortcut:
            x = self.embed_lif(x)
        x_feat = x
        T, B = x.shape[:2]
        x = self.embed_conv(x.flatten(0, 1))
        x = self.embed_bn(x)
        x = self.maxpool(x)                  # ÷2 spatial
        if dual:
            return x, x_feat
        return x


# ═══════════════════════════════════════════════════════════════════════
#  Stage-1 stem: Embed_Orig_ImageNet  (224→56, ÷4)
# ═══════════════════════════════════════════════════════════════════════
class Embed_Orig_ImageNet(nn.Module):
    """
    First patch embedding for ImageNet-sized images.
    4 overlapping Conv-BN stages with stride-2 to go from H×W → H/4 × W/4,
    plus a membrane shortcut.
    """

    def __init__(self, in_channels=3, embed_dims=256):
        super().__init__()
        self.embed1 = Embed(in_channels, embed_dims // 2,
                            kernel_size=3, stride=2, padding=1, shortcut=True)
        self.embed2 = Embed(embed_dims // 2, embed_dims,
                            kernel_size=3, stride=2, padding=1)
        self.embed3 = Embed(embed_dims, embed_dims,
                            kernel_size=3, stride=1, padding=1)
        # shortcut path  (1×1 conv, stride 2)
        self.embed4 = Embed(embed_dims // 2, embed_dims,
                            kernel_size=1, stride=2, padding=0, shortcut=True)

    def forward(self, x):
        """x: [T, B, C, H, W]  →  [T, B, embed_dims, H/4, W/4]"""
        T, B, C, H, W = x.shape

        x = self.embed1(x)                                                # [T*B, D/2, H/2, W/2]
        x = x.reshape(T, B, -1, H // 2, W // 2).contiguous()

        x, x_feat = self.embed2(x, dual=True)                            # [T*B, D, H/4, W/4]
        x = x.reshape(T, B, -1, H // 4, W // 4).contiguous()

        x = self.embed3(x)                                                # [T*B, D, H/4, W/4]

        # membrane shortcut
        x_feat = self.embed4(x_feat)                                      # [T*B, D, H/4, W/4]

        x = (x + x_feat).reshape(T, B, -1, H // 4, W // 4).contiguous()
        return x


# ═══════════════════════════════════════════════════════════════════════
#  Inter-stage downsampler: Embed_Max  (÷2 via MaxPool)
# ═══════════════════════════════════════════════════════════════════════
class Embed_Max_Stage(nn.Module):
    """
    MaxPool-based embedding that halves spatial dims and doubles channels.
    Includes a membrane shortcut (3×3 path + 1×1 shortcut path, summed).
    """

    def __init__(self, in_channels, embed_dims):
        super().__init__()
        self.max_embed1 = Max_Embed(in_channels, embed_dims,
                                    kernel_size=3, stride=1, padding=1)
        self.embed1 = Embed(embed_dims, embed_dims,
                            kernel_size=3, stride=1, padding=1)
        self.max_embed2 = Max_Embed(in_channels, embed_dims,
                                    kernel_size=1, stride=1, padding=0,
                                    shortcut=True)

    def forward(self, x):
        """x: [T, B, C_in, H, W]  →  [T, B, embed_dims, H/2, W/2]"""
        T, B, C, H, W = x.shape

        x, x_feat = self.max_embed1(x, dual=True)              # [T*B, D, H/2, W/2]
        x = x.reshape(T, B, -1, H // 2, W // 2).contiguous()

        x = self.embed1(x)                                      # [T*B, D, H/2, W/2]

        # shortcut path (input must already be spiking when shortcut=True)
        x_feat = self.max_embed2(x_feat)                        # [T*B, D, H/2, W/2]

        x = (x + x_feat).reshape(T, B, -1, H // 2, W // 2).contiguous()
        return x
