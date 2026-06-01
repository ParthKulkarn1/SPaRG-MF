import torch
import torch.nn as nn
from spikingjelly.activation_based import neuron, layer, surrogate


class SpikingSoftSparseSelfAttention(nn.Module):
    """
    Soft-Sparse Spiking Self Attention.
    Uses a learnable temperature parameter to sharpen/sparsify the attention distribution 
    without breaking surrogate gradients. Includes hooks for dynamic head gating and attention sparsification.

    Parameters
    ----------
    dim : int
        Token embedding dimension.
    num_heads : int
        Number of attention heads.
    qkv_bias : bool
        If True, add bias to Query, Key, and Value linear layers.
    attn_drop : float
        Dropout probability on attention maps.
    proj_drop : float
        Dropout probability on output projection.
    tau : float
        LIF membrane time constant.
    sparse_threshold : float, optional
        Minimum attention score threshold. Entries below this are masked out.
    sparse_topk : float, optional
        Fraction of attention entries (e.g. 0.3) to keep per query token.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        tau: float = 2.0,
        sparse_threshold: float = None,
        sparse_topk: float = None,
    ):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.sparse_threshold = sparse_threshold
        self.sparse_topk = sparse_topk
        
        # Linear layers extended over Time (T) dimension provided by SpikingJelly Layer wrapper
        self.qkv = layer.Linear(dim, dim * 3, bias=qkv_bias)
        
        # Soft Sparse Temperature
        self.temperature = nn.Parameter(torch.ones(1) * 1.0)
        
        self.attn_drop = layer.Dropout(attn_drop)
        self.proj = layer.Linear(dim, dim)
        self.proj_drop = layer.Dropout(proj_drop)
        
        # Post-attention Spike Emitter
        self.lif = neuron.LIFNode(tau=tau, surrogate_function=surrogate.ATan(), step_mode='m')

    def forward(self, x: torch.Tensor, head_mask: torch.Tensor = None) -> torch.Tensor:
        """
        x: [Time, Batch, N_tokens, Channels]
        head_mask: Optional mask of shape [T_dim, B_dim, num_heads, 1, 1] to physically drop heads
        """
        T, B, N, C = x.shape
        
        # Compute QKV across all time steps
        qkv = self.qkv(x).reshape(T, B, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(3, 0, 1, 4, 2, 5) # [3, Time, Batch, Head, N_tokens, Head_dim]
        
        q, k, v = qkv[0], qkv[1], qkv[2]   
        
        # Save Q, K, V states
        self.last_q = q.detach()
        self.last_k = k.detach()
        self.last_v = v.detach()
        
        # Attention score
        attn = (q @ k.transpose(-2, -1)) * self.scale  # [T, B, Head, N, N]
        
        # Soft-Sparsity Step (sharpening with temperature)
        attn = attn / torch.clamp(self.temperature, min=1e-3)
        
        # Physical dynamic / static head gating
        if head_mask is not None:
            # head_mask has shape [T_dim, B_dim, num_heads, 1, 1], mask out deactivated heads
            attn = attn.masked_fill(head_mask == 0, float('-inf'))
            
        # Attention Sparsification (Top-k or threshold)
        if self.sparse_threshold is not None:
            attn = attn.masked_fill(attn < self.sparse_threshold, float('-inf'))
        elif self.sparse_topk is not None:
            # Keep only top-k entries per query token along the key dimension (last dimension)
            k_val = max(int(N * self.sparse_topk), 1)
            topk_vals, _ = attn.topk(k_val, dim=-1)
            threshold = topk_vals[..., -1:].detach()
            attn = attn.masked_fill(attn < threshold, float('-inf'))

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
