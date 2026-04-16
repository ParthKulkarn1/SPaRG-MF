import torch
import torch.nn as nn


class StraightThroughQuantize(torch.autograd.Function):
    """
    Straight-Through Estimator (STE) quantization.

    Forward: quantize activations to `num_bits` precision.
    Backward: pass gradients through unchanged (identity).
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, num_bits: int = 8):
        # Symmetric uniform quantization
        qmin = -(2 ** (num_bits - 1))
        qmax = 2 ** (num_bits - 1) - 1
        scale = x.abs().max().clamp(min=1e-8) / qmax
        x_q = (x / scale).round().clamp(qmin, qmax) * scale
        return x_q

    @staticmethod
    def backward(ctx, grad_output):
        # STE: pass gradient unchanged
        return grad_output, None


class MixedPrecisionController(nn.Module):
    """
    Token-confidence-driven mixed-precision wrapper.

    Given a saliency score per token (from the dynamic gate or SSA
    interceptor), this module routes:
      - high-saliency tokens through higher precision (FP16 / 8-bit)
      - low-saliency tokens through lower precision (4-bit)

    During training the quantization uses a Straight-Through Estimator
    so gradients still flow.

    Parameters
    ----------
    high_bits : int
        Bit-width for "important" tokens.
    low_bits : int
        Bit-width for "background" tokens.
    saliency_threshold : float
        Tokens with saliency >= threshold are routed to high precision.
    """

    def __init__(self, high_bits: int = 8, low_bits: int = 4, saliency_threshold: float = 0.5):
        super().__init__()
        self.high_bits = high_bits
        self.low_bits = low_bits
        self.saliency_threshold = saliency_threshold

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor, token_saliency: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor [Time, Batch, N_tokens, Channels]
            Activation tensor to quantize.
        token_saliency : Tensor broadcastable to [1, 1, N_tokens, 1]
            Per-token importance score in [0, 1].

        Returns
        -------
        x_mixed : Tensor  same shape as x
        """
        # Build boolean mask for high / low precision regions
        high_mask = (token_saliency >= self.saliency_threshold)  # bool

        x_high = StraightThroughQuantize.apply(x, self.high_bits)
        x_low = StraightThroughQuantize.apply(x, self.low_bits)

        # Select: where saliency is high use higher precision, else lower
        x_mixed = torch.where(high_mask, x_high, x_low)
        return x_mixed
