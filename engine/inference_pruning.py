import torch
import torch.nn as nn
from spikingjelly.activation_based import functional
from .ssa_interceptor import SSAInterceptor


class SNNInferencePruningEngine:
    """
    SNN Inference Pruning and Calibration Engine.

    Facilitates post-training inference-only pruning without retraining by:
      1. Running a calibration dataset forward pass to collect SNN firing statistics.
      2. Analyzing metrics: Head Quality (Q-value), Redundancy, Saliency, and Entropy.
      3. Constructing customized hardware gating masks based on chosen metrics and thresholds.
      4. Running accuracy/sparsity sweep sweeps to evaluate Pareto efficiency.
    """

    def __init__(self, model: nn.Module, num_heads: int):
        self.model = model
        self.num_heads = num_heads
        self.interceptor = SSAInterceptor(model)

    def calibrate(self, dataloader, device, num_batches: int = 5):
        """
        Run forward calibration passes using the given dataloader to collect statistics.
        """
        self.model.eval()
        self.interceptor.attach()

        print(f"Running {num_batches} calibration batches on {device}...")
        batches_processed = 0
        with torch.no_grad():
            for inputs, _ in dataloader:
                if batches_processed >= num_batches:
                    break
                inputs = inputs.to(device)
                
                # Reset network states for clean temporal accumulation
                self.model.reset_snn_state()
                
                # Forward pass triggers hooks
                _ = self.model(inputs)
                batches_processed += 1

        self.interceptor.detach()
        print("Calibration completed successfully.")

    def generate_static_masks(self, metric: str = 'importance', threshold: float = 1e-5, prune_ratio: float = None):
        """
        Generate static binary head-gating masks for Stage-3 based on a metric threshold or prune ratio.

        Supported Metrics:
          - 'importance': Prune heads with low Variance * Mean activations.
          - 'q_value': Prune heads with Query-Key alignment (cosine similarity) < threshold.
          - 'redundancy': Prune heads with average cross-head correlation > threshold.
          - 'entropy': Prune heads with attention Shannon entropy > threshold.
          - 'spike_entropy': Prune heads with Bernoulli spike output entropy < threshold.

        Parameters
        ----------
        metric : str
            Diagnostic metric to use.
        threshold : float
            Static absolute threshold. Ignored if prune_ratio is provided.
        prune_ratio : float, optional
            Fraction of heads (e.g. 0.25) to prune globally across all blocks.
            If provided, thresholds are dynamically set to the given percentile.

        Returns
        -------
        masks : list[Tensor]
            One mask tensor of shape [1, 1, num_heads, 1, 1] per SSA block.
        """
        if not self.interceptor.block_data:
            raise ValueError("No calibration data found. Please run calibrate() first.")

        # Compute metric score per block
        metric = metric.lower()
        if metric == 'importance':
            scores_dict = self.interceptor.compute_head_importance(self.num_heads)
            lower_is_worse = True
        elif metric == 'q_value':
            scores_dict = self.interceptor.compute_head_q_value()
            lower_is_worse = True
        elif metric == 'redundancy':
            scores_dict = self.interceptor.compute_head_redundancy(self.num_heads)
            lower_is_worse = False  # Higher redundancy is worse
        elif metric == 'entropy':
            scores_dict = self.interceptor.compute_attention_entropy()
            lower_is_worse = False  # Higher entropy (random focus) is worse
        elif metric == 'spike_entropy':
            scores_dict = self.interceptor.compute_spike_entropy(self.num_heads)
            lower_is_worse = True   # Saturated/dead heads have low entropy, which is worse
        else:
            raise ValueError(f"Unknown pruning metric: {metric}")

        if scores_dict is None:
            raise RuntimeError(f"Failed to compute score dict for metric: {metric}")

        # If prune_ratio is provided, compute a dynamic threshold based on percentiles
        if prune_ratio is not None:
            assert 0.0 <= prune_ratio <= 1.0, "prune_ratio must be between 0.0 and 1.0"
            # Pool all head scores across all blocks
            all_scores = []
            for block_name in sorted(scores_dict.keys()):
                all_scores.append(scores_dict[block_name])
            all_scores = torch.cat(all_scores, dim=0) # [Num_Blocks * Num_Heads]
            
            # Find the percentile score
            k = max(int(all_scores.numel() * prune_ratio), 1)
            if lower_is_worse:
                # We want to prune the bottom X% lowest scores
                topk_vals, _ = all_scores.topk(all_scores.numel() - k + 1, largest=True)
                threshold = topk_vals[-1].item()
                keep_fn = lambda score: score >= threshold
            else:
                # We want to prune the top X% highest scores (redundancy/entropy)
                topk_vals, _ = all_scores.topk(all_scores.numel() - k + 1, largest=False)
                threshold = topk_vals[-1].item()
                keep_fn = lambda score: score <= threshold
        else:
            # Revert to static absolute thresholding
            if lower_is_worse:
                keep_fn = lambda score: score >= threshold
            else:
                keep_fn = lambda score: score <= threshold

        masks = []
        # Sort blocks to ensure order match
        for block_name in sorted(scores_dict.keys()):
            scores = scores_dict[block_name]
            
            # Construct binary mask based on keep condition
            mask = keep_fn(scores).float()
            
            # Target shape for broadcast in self-attention: [1, 1, num_heads, 1, 1]
            mask_tensor = mask.view(1, 1, self.num_heads, 1, 1)
            masks.append(mask_tensor)

        return masks

    def evaluate_mask_efficiency(self, dataloader, device, masks):
        """
        Evaluate the validation accuracy of the model under a given set of static gating masks.
        Also calculates the head sparsity percentage.
        """
        self.model.eval()
        correct = 0
        total = 0
        
        # Calculate sparsity
        total_heads = 0
        active_heads = 0
        for m in masks:
            total_heads += m.numel()
            active_heads += int(m.sum().item())
        sparsity = 100.0 * (1.0 - (active_heads / total_heads))

        with torch.no_grad():
            for inputs, targets in dataloader:
                inputs, targets = inputs.to(device), targets.to(device)
                
                # Apply static masks
                self.model.reset_snn_state()
                masks_device = [m.to(device) for m in masks]
                outputs = self.model(inputs, external_head_masks=masks_device)
                
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        accuracy = 100.0 * correct / total
        return accuracy, sparsity
