import torch
import torch.nn as nn


class DiversityLoss(nn.Module):
    """
    Self-supervised regularizer to reduce attention head redundancy and prevent hallucinations.

    Computes the pairwise cosine similarity of Query (or Key) representations across
    different heads in the same attention layer. Penalizes the off-diagonal elements
    (cross-head correlation) to force the heads to capture diverse, orthogonal features.
    
    Parameters
    ----------
    weight : float
        Regularization multiplier in the total loss function.
    """

    def __init__(self, weight: float = 0.01):
        super().__init__()
        self.weight = weight

    def forward(self, model: nn.Module) -> torch.Tensor:
        """
        Scans the model for active SoftSparseSSA or SpikingSoftSparseSelfAttention modules,
        extracts their last Q (query) state, and computes head orthogonalization penalties.
        """
        total_loss = 0.0
        count = 0
        
        # Check if the model has parameters to identify the correct device
        first_param = next(model.parameters(), None)
        if first_param is None:
            return torch.tensor(0.0)
        device = first_param.device

        for name, module in model.named_modules():
            # Targets: SoftSparseSSA or SpikingSoftSparseSelfAttention (typically ends with 'attn')
            if name.endswith('attn'):
                q = getattr(module, 'last_q', None)
                if q is not None:
                    # q shape: [T, B, H, N, D]
                    T, B, H, N, D = q.shape
                    
                    if H <= 1:
                        continue

                    # Reshape to expose heads and flatten other dimensions: [H, T * B * N * D]
                    features = q.permute(2, 0, 1, 3, 4).reshape(H, -1)
                    
                    # Normalize features to unit vectors to compute cosine similarity
                    norms = features.norm(dim=-1, keepdim=True) + 1e-8
                    normalized = features / norms
                    
                    # Cosine similarity matrix: [H, H]
                    sim_matrix = normalized @ normalized.T
                    
                    # Target: identity matrix (0 correlation between different heads)
                    # We penalize all off-diagonal correlations
                    eye = torch.eye(H, device=device)
                    off_diag = sim_matrix * (1.0 - eye)
                    
                    # Mean squared correlation of off-diagonal elements
                    loss = (off_diag ** 2).sum() / (H * (H - 1))
                    
                    total_loss = total_loss + loss
                    count = count + 1

        if count > 0:
            return (total_loss / count) * self.weight
        return torch.tensor(0.0, device=device)
