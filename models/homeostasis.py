import torch
import torch.nn as nn
from spikingjelly.activation_based import neuron, surrogate, base


class HomeostaticLIFNode(base.MemoryModule):
    """
    Adaptive-threshold LIF neuron with homeostatic spike-rate control.

    The firing threshold is dynamically adjusted based on a moving-average
    spike rate so that:
      - Neurons that fire too often have their threshold raised  (dampen)
      - Neurons that fire too rarely have their threshold lowered (excite)

    This prevents dead pathways and dominant pathways after
    sparsification / head gating.

    Parameters
    ----------
    tau : float
        Membrane time constant for the underlying LIF dynamics.
    v_threshold_init : float
        Initial firing threshold.
    target_rate : float
        Desired mean firing rate (fraction of time steps that emit a spike).
    homeo_rate : float
        EMA coefficient for the moving-average spike-rate tracker (0..1).
        Smaller = slower adaptation.
    adapt_scale : float
        How aggressively the threshold adjusts per adaptation step.
    step_mode : str
        SpikingJelly step mode; 'm' for multi-step (required here).
    """

    def __init__(
        self,
        tau: float = 2.0,
        v_threshold_init: float = 1.0,
        target_rate: float = 0.3,
        homeo_rate: float = 0.01,
        adapt_scale: float = 0.05,
        step_mode: str = 'm',
    ):
        super().__init__()

        # Core spiking neuron
        self.lif = neuron.LIFNode(
            tau=tau,
            v_threshold=v_threshold_init,
            surrogate_function=surrogate.ATan(),
            step_mode=step_mode,
        )

        # Homeostasis bookkeeping (not learned; updated in-place)
        self.target_rate = target_rate
        self.homeo_rate = homeo_rate
        self.adapt_scale = adapt_scale

        # Running average spike rate (scalar, shared across all neurons in this
        # module instance).  Registered as a buffer so it persists across
        # forward calls but is not optimized.
        self.register_buffer('avg_spike_rate', torch.tensor(target_rate))

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor  [Time, Batch, *spatial]
            Membrane-potential input (pre-spike).

        Returns
        -------
        spikes : Tensor  same shape as x
        """
        spikes = self.lif(x)

        # --- Homeostatic threshold adaptation (no grad) ----------------
        with torch.no_grad():
            # Current instantaneous rate (fraction of 1s in the spike tensor)
            current_rate = spikes.mean()

            # Exponential moving average
            self.avg_spike_rate = (
                (1.0 - self.homeo_rate) * self.avg_spike_rate
                + self.homeo_rate * current_rate
            )

            # Adjust threshold
            rate_error = self.avg_spike_rate - self.target_rate
            self.lif.v_threshold = max(
                0.1, self.lif.v_threshold + self.adapt_scale * rate_error.item()
            )

        return spikes

    # ------------------------------------------------------------------
    def reset(self):
        """Reset internal LIF state (call between samples / epochs)."""
        self.lif.reset()

    def extra_repr(self) -> str:
        return (
            f"target_rate={self.target_rate}, "
            f"homeo_rate={self.homeo_rate}, "
            f"adapt_scale={self.adapt_scale}, "
            f"v_threshold={self.lif.v_threshold:.4f}"
        )
