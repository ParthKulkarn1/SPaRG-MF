import torch
import math
from collections import defaultdict


class SSAInterceptor:
    """
    Hooks into the inference pass of SPaRG-MF to empirically deduce neuromorphic
    gating pathways and analyze multi-head attention dynamics.

    Captures:
      - Post-attention pre-LIF activations
      - Q, K, V state representations
      - Softmax or Linear attention maps

    Provides mathematical diagnostics for:
      1. Q-Value (Query-Key alignment quality)
      2. Head Redundancy (pairwise head cosine similarity)
      3. Attention Entropy (dispersion/focus of attention weights)
      4. Spike Entropy (Bernoulli information capacity of channel outputs)
    """

    def __init__(self, model):
        self.model = model
        self.handles = []
        # Store captured block states: {block_name: [dict, dict, ...]}
        self.block_data = defaultdict(list)

    def _make_hook(self, block_name, module):
        """Create a closure hook to capture activations, attention maps, and QKV states."""
        def hook_fn(mod, input, output):
            # input[0] is the pre-lif attention tensor: [T, B, C, N]
            act = input[0].detach().cpu()
            
            # Fetch Q, K, V, and Attention maps from the module
            q = getattr(module, 'last_q', None)
            k = getattr(module, 'last_k', None)
            v = getattr(module, 'last_v', None)
            attn = getattr(module, 'last_attn_weights', None)

            self.block_data[block_name].append({
                'activation': act,
                'q': q.cpu() if q is not None else None,
                'k': k.cpu() if k is not None else None,
                'v': v.cpu() if v is not None else None,
                'attn': attn.cpu() if attn is not None else None
            })
        return hook_fn

    def attach(self):
        """Attach hooks to all stage3 attention lif nodes in the model."""
        self.block_data = defaultdict(list)
        self.handles = []

        # Access model's named modules
        named_mods = dict(self.model.named_modules())

        for name, module in self.model.named_modules():
            # Target: stage3.X.attn.attn_lif (post-attention spiking node)
            if 'stage3' in name and name.endswith('attn.attn_lif'):
                # Extract block name (e.g. "stage3.2")
                block_name = '.'.join(name.split('.')[:2])
                
                # Fetch parent attention module (e.g., "stage3.2.attn")
                parent_name = '.'.join(name.split('.')[:-1])
                parent_module = named_mods[parent_name]

                handle = module.register_forward_hook(self._make_hook(block_name, parent_module))
                self.handles.append(handle)
                print(f"  Attached SSA Hook -> {name}  (block: {block_name})")

        if not self.handles:
            print("  WARNING: No SSA hooks attached! Check model structure.")
        else:
            print(f"  Total hooks: {len(self.handles)}")

    def detach(self):
        """Remove all active hooks."""
        for handle in self.handles:
            handle.remove()
        self.handles = []

    def compute_head_importance(self, num_heads):
        """
        Compute standard importance score: Variance(act) * Mean(act)
        across all captured batches, time steps, and spatial tokens.
        """
        if not self.block_data:
            return None

        results = {}
        for block_name, captures in self.block_data.items():
            acts = torch.cat([c['activation'] for c in captures], dim=0)  # [total_T, B, C, N]
            T_tot, B, C, N = acts.shape
            D = C // num_heads
            acts = acts.reshape(T_tot, B, num_heads, D, N)

            mean_act = acts.mean(dim=(0, 1, 3, 4))  # [num_heads]
            var_act = acts.var(dim=(0, 1, 3, 4))    # [num_heads]
            results[block_name] = mean_act * var_act

        return results

    def compute_head_q_value(self):
        """
        Compute Query-Key alignment quality (Q-value) per head:
        mean of cosine similarity between Q and K states.
        High alignment carries high quality, whereas random carries low.
        """
        if not self.block_data:
            return None

        results = {}
        for block_name, captures in self.block_data.items():
            if captures[0]['q'] is None or captures[0]['k'] is None:
                continue

            q_all = torch.cat([c['q'] for c in captures], dim=0)  # [T, B, H, N, D]
            k_all = torch.cat([c['k'] for c in captures], dim=0)  # [T, B, H, N, D]

            # Cosine similarity across the head dimension D
            # Shape: [T, B, H, N]
            cos_sim = torch.sum(q_all * k_all, dim=-1) / (
                q_all.norm(dim=-1) * k_all.norm(dim=-1) + 1e-8
            )

            # Average over Time, Batch, and Tokens -> [H]
            mean_q_value = cos_sim.mean(dim=(0, 1, 3))
            results[block_name] = mean_q_value

        return results

    def compute_head_redundancy(self, num_heads):
        """
        Compute the average cosine similarity of each head's representations
        with all other heads in the same layer.
        Values close to 1 represent high redundancy.
        """
        if not self.block_data:
            return None

        results = {}
        for block_name, captures in self.block_data.items():
            acts = torch.cat([c['activation'] for c in captures], dim=0)  # [total_T, B, C, N]
            T_tot, B, C, N = acts.shape
            D = C // num_heads
            
            # Reshape to expose heads and flatten all other dimensions
            # Shape: [H, total_T * B * D * N]
            head_features = acts.reshape(T_tot, B, num_heads, D, N)
            head_features = head_features.permute(2, 0, 1, 3, 4).reshape(num_heads, -1)

            # L2 Normalize
            norms = head_features.norm(dim=-1, keepdim=True) + 1e-8
            normalized = head_features / norms

            # Compute pairwise cosine similarities: [H, H]
            sim_matrix = normalized @ normalized.T

            # Average correlation for each head excluding self-correlation
            mean_redundancy = (sim_matrix.sum(dim=-1) - 1.0) / (num_heads - 1)
            results[block_name] = mean_redundancy

        return results

    def compute_attention_entropy(self):
        """
        Compute Shannon entropy of the attention maps.
        Low entropy = sharp focus (high information).
        High entropy = uniform distribution (low information / redundancy).
        """
        if not self.block_data:
            return None

        results = {}
        for block_name, captures in self.block_data.items():
            if captures[0]['attn'] is None:
                continue

            attn_all = torch.cat([c['attn'] for c in captures], dim=0)  # [T, B, H, D, D] or [T, B, H, N, N]
            
            # Normalize attention over the last dimension to construct probability maps
            probs = attn_all / (attn_all.sum(dim=-1, keepdim=True) + 1e-8)
            
            # Shannon entropy: -sum(p * log(p))
            entropy = - (probs * torch.log(probs + 1e-8)).sum(dim=-1)
            
            # Average over remaining dimensions -> [H]
            mean_entropy = entropy.mean(dim=(0, 1, 3))
            results[block_name] = mean_entropy

        return results

    def compute_spike_entropy(self, num_heads):
        """
        Compute Bernoulli information entropy of the spiking channel output per head.
        Saturated neurons (always 1) or silent neurons (always 0) carry 0 information.
        Dynamic, varying pathways carry higher Bernoulli entropy.
        """
        if not self.block_data:
            return None

        results = {}
        for block_name, captures in self.block_data.items():
            acts = torch.cat([c['activation'] for c in captures], dim=0)  # [total_T, B, C, N]
            T_tot, B, C, N = acts.shape
            D = C // num_heads
            
            # Reshape to [total_T, B, H, D, N]
            acts = acts.reshape(T_tot, B, num_heads, D, N)
            
            # Firing rate p per neuron (average over Time, Batch, Tokens): [H, D]
            p = acts.mean(dim=(0, 1, 4))
            p = p.clamp(min=1e-6, max=1.0 - 1e-6)

            # Bernoulli entropy: -p * log(p) - (1-p) * log(1-p)
            bern_entropy = - p * torch.log(p) - (1.0 - p) * torch.log(1.0 - p)
            
            # Average Bernoulli entropy across the D channels of each head
            results[block_name] = bern_entropy.mean(dim=-1)  # [H]

        return results

    def generate_hardware_gating_masks(self, num_heads, threshold=1e-5):
        """Generate binary head masks based on variance-mean importance."""
        importance = self.compute_head_importance(num_heads=num_heads)
        if importance is None:
            return None

        masks = []
        for block_name in sorted(importance.keys()):
            scores = importance[block_name]
            mask = (scores > threshold).float()
            mask = mask.view(1, 1, num_heads, 1, 1)
            masks.append(mask)

        return masks

    def summary(self, num_heads):
        """Print a comprehensive summary of all computed SNN attention diagnostics."""
        print(f"\n{'='*65}")
        print(f"  SPaRG-MF SSA Interception Metrics Summary")
        print(f"{'='*65}")
        
        importance = self.compute_head_importance(num_heads)
        q_values = self.compute_head_q_value()
        redundancy = self.compute_head_redundancy(num_heads)
        attn_entropy = self.compute_attention_entropy()
        spike_entropy = self.compute_spike_entropy(num_heads)

        for block_name in sorted(self.block_data.keys()):
            captures = self.block_data[block_name]
            print(f"\n  [ Block {block_name} ] - {len(captures)} forward captures")
            
            if importance:
                scores = importance[block_name]
                print(f"    + Importance (Var*Mean): " + ", ".join([f"H{j}:{scores[j]:.2e}" for j in range(num_heads)]))
            if q_values and block_name in q_values:
                q_v = q_values[block_name]
                print(f"    + Q-Value (QK-Align)  : " + ", ".join([f"H{j}:{q_v[j]:.3f}" for j in range(num_heads)]))
            if redundancy:
                red = redundancy[block_name]
                print(f"    + Head Redundancy     : " + ", ".join([f"H{j}:{red[j]:.3f}" for j in range(num_heads)]))
            if attn_entropy and block_name in attn_entropy:
                ent = attn_entropy[block_name]
                print(f"    + Attention Entropy   : " + ", ".join([f"H{j}:{ent[j]:.3f}" for j in range(num_heads)]))
            if spike_entropy:
                spk_ent = spike_entropy[block_name]
                print(f"    + Spike Bern Entropy  : " + ", ".join([f"H{j}:{spk_ent[j]:.3f}" for j in range(num_heads)]))
