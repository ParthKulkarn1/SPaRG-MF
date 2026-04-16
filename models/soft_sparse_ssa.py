import torch
import torch.nn as nn
from spikingjelly.activation_based import neuron, layer, surrogate

class SpikingSoftSparseSelfAttention(nn.Module):
    """
    Soft-Sparse Spiking Self Attention.
    Uses a learnable temperature parameter to sharpen/sparsify the attention distribution 
    without breaking surrogate gradients. Includes hooks for dynamic head gating.
    """
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0., tau=2.0):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        
        # Linear layers extended over Time (T) dimension provided by SpikingJelly Layer wrapper
        self.qkv = layer.Linear(dim, dim * 3, bias=qkv_bias)
        
        # Soft Sparse Temperature
        self.temperature = nn.Parameter(torch.ones(1) * 1.0)
        
        self.attn_drop = layer.Dropout(attn_drop)
        self.proj = layer.Linear(dim, dim)
        self.proj_drop = layer.Dropout(proj_drop)
        
        # Post-attention Spike Emitter
        self.lif = neuron.LIFNode(tau=tau, surrogate_function=surrogate.ATan(), step_mode='m')

    def forward(self, x, head_mask=None):
        """
        x: [Time, Batch, N_tokens, Channels]
        head_mask: Optional binary mask of shape [1, 1, num_heads, 1, 1] to physically drop heads
        """
        T, B, N, C = x.shape
        
        # Compute QKV across all time steps
        qkv = self.qkv(x).reshape(T, B, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(3, 0, 1, 4, 2, 5) # [3, Time, Batch, Head, N_tokens, Head_dim]
        
        q, k, v = qkv[0], qkv[1], qkv[2]   
        
        # Attention score
        attn = (q @ k.transpose(-2, -1)) * self.scale
        
        # Soft-Sparsity Step (sharpening with temperature)
        attn = attn / torch.clamp(self.temperature, min=1e-3)
        
        # Inference Hardware Constraint: Zero out heads pruned by Interception Engine
        if head_mask is not None:
            attn = attn.masked_fill(head_mask == 0, float('-inf'))
            
        # Attention probabilities
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        # Context aggregation
        x = (attn @ v).transpose(2, 3).reshape(T, B, N, C)
        
        # Output projection
        x = self.proj(x)
        x = self.proj_drop(x)
        
        # Spiking Activation
        x = self.lif(x)
        
        return x
