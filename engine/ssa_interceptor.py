import torch
from collections import defaultdict


class SSAInterceptor:
    """
    Hooks into the inference pass to empirically deduce robust neuromorphic
    gating pathways from Spiking Self-Attention dynamics.

    Attaches to every SoftSparseSSA module's attn_lif (the spiking node
    immediately after the attention computation).  Captures:
      - Post-attention spike maps     → head importance
      - Pre-softmax Q/K interactions  → token saliency

    Works with the hierarchical Max-Former where SSA only exists in Stage 3.
    """

    def __init__(self, model):
        self.model = model
        self.handles = []
        # Store per-block attention data
        self.block_data = defaultdict(list)
        self._current_block = None

    def _make_hook(self, block_name):
        """Create a closure that tags captured data with its block name."""
        def hook_fn(module, input, output):
            # input[0] is the pre-lif attention tensor: [T, B, C, N]
            self.block_data[block_name].append(
                input[0].detach().cpu()
            )
        return hook_fn

    def attach(self):
        """Attach hooks to all SSA attn_lif nodes in the model."""
        self.block_data = defaultdict(list)
        self.handles = []

        for name, module in self.model.named_modules():
            # Target: stage3.X.attn.attn_lif  (the post-attention spike node)
            if 'stage3' in name and name.endswith('attn.attn_lif'):
                # Extract block index from name (e.g. "stage3.2.attn.attn_lif" → "stage3.2")
                block_name = '.'.join(name.split('.')[:2])
                handle = module.register_forward_hook(self._make_hook(block_name))
                self.handles.append(handle)
                print(f"  Attached SSA Hook → {name}  (block: {block_name})")

        if not self.handles:
            print("  WARNING: No SSA hooks attached! Check model structure.")
        else:
            print(f"  Total hooks: {len(self.handles)}")

    def detach(self):
        """Remove all hooks."""
        for handle in self.handles:
            handle.remove()
        self.handles = []

    def compute_head_importance(self, num_heads=None):
        """
        Compute per-head importance for each SSA block.

        Importance = Var(activation) * Mean(activation)
        across all captured time steps and spatial positions.

        Returns
        -------
        dict : {block_name: Tensor[num_heads]}
        """
        if not self.block_data:
            print("No attention data collected. Run forward passes first.")
            return None

        results = {}
        for block_name, tensors in self.block_data.items():
            # Each tensor: [T, B, C, N]  (pre-lif attention output)
            all_data = torch.cat(tensors, dim=0)  # [total_T, B, C, N]

            if num_heads is not None:
                # Reshape to expose heads: [total_T, B, H, D, N]
                T_tot, B, C, N = all_data.shape
                D = C // num_heads
                all_data = all_data.reshape(T_tot, B, num_heads, D, N)
                # Per-head statistics
                mean_act = all_data.mean(dim=(0, 1, 3, 4))  # [H]
                var_act = all_data.var(dim=(0, 1, 3, 4))     # [H]
            else:
                # Treat entire channel dim as one
                mean_act = all_data.mean()
                var_act = all_data.var()

            results[block_name] = mean_act * var_act

        return results

    def generate_hardware_gating_masks(self, num_heads, threshold=1e-5):
        """
        Generate per-block binary head masks for hardware enforcement.

        Returns
        -------
        list[Tensor]  : one mask per SSA block, shape [1, 1, num_heads, 1, 1]
        """
        importance = self.compute_head_importance(num_heads=num_heads)
        if importance is None:
            return None

        masks = []
        # Sort by block name to ensure correct ordering
        for block_name in sorted(importance.keys()):
            scores = importance[block_name]
            mask = (scores > threshold).float()
            mask = mask.view(1, 1, num_heads, 1, 1)
            masks.append(mask)

        return masks

    def compute_token_saliency(self):
        """
        Compute per-token saliency persistence across sustained inference.
        Useful for static token pruning policies.

        Returns
        -------
        dict : {block_name: Tensor[N]}  mean activation per spatial token
        """
        if not self.block_data:
            return None

        results = {}
        for block_name, tensors in self.block_data.items():
            all_data = torch.cat(tensors, dim=0)  # [total_T, B, C, N]
            # Saliency = mean activation per token position
            results[block_name] = all_data.mean(dim=(0, 1, 2))  # [N]

        return results

    def summary(self, num_heads=None):
        """Print a human-readable summary of intercepted statistics."""
        print(f"\n{'='*50}")
        print(f"  SSA Interception Summary")
        print(f"{'='*50}")
        for block_name in sorted(self.block_data.keys()):
            tensors = self.block_data[block_name]
            n_captures = len(tensors)
            shape = tensors[0].shape if tensors else "N/A"
            print(f"  {block_name}: {n_captures} captures, tensor shape: {shape}")

        importance = self.compute_head_importance(num_heads=num_heads)
        if importance:
            print(f"\n  Head Importance Scores:")
            for name, scores in sorted(importance.items()):
                print(f"    {name}: {scores}")
