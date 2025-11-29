import torch
import torch.nn as nn

class SupervisedNTXentLoss(nn.Module):
    """
    Multi-positive supervised contrastive loss.

    - Uses two views z_i, z_j: each (B, D), already L2-normalized.
    - labels: (B,) integer labels (e.g., faculty IDs).
    - Positives for each anchor = all embeddings in the batch (both views)
      that share the same label (excluding itself).
    """
    def __init__(self, temperature: float = 0.05):
        super().__init__()
        self.temperature = temperature

    def forward(self, z_i, z_j, labels):
        device = z_i.device
        batch_size = z_i.size(0)

        # Stack the two views: (2B, D)
        z = torch.cat([z_i, z_j], dim=0)
        z = nn.functional.normalize(z, p=2, dim=-1)

        # Duplicate labels for both views: (2B,)
        labels = labels.to(device)
        labels_all = torch.cat([labels, labels], dim=0)

        # Similarity matrix: (2B, 2B)
        sim = torch.matmul(z, z.T) / self.temperature

        # Mask to remove self-similarity
        self_mask = torch.eye(2 * batch_size, dtype=torch.bool, device=device)

        # Label equality mask: (2B, 2B)
        label_eq = labels_all.unsqueeze(0) == labels_all.unsqueeze(1)
        # Positives: same label, not self
        positive_mask = label_eq & (~self_mask)

        # For numerical stability
        sim_max, _ = sim.max(dim=1, keepdim=True)
        sim = sim - sim_max.detach()

        exp_sim = torch.exp(sim)

        # Denominator: sum over all j != i
        exp_sim = exp_sim.masked_fill(self_mask, 0.0)
        denom = exp_sim.sum(dim=1) + 1e-12

        # Numerator: sum over positives
        pos_exp = exp_sim * positive_mask.float()
        numer = pos_exp.sum(dim=1) + 1e-12

        # Only anchors that have at least one positive
        valid = positive_mask.sum(dim=1) > 0
        loss = -torch.log(numer / denom)
        loss = loss[valid].mean()

        return loss
    
    
def isotropy_regularizer(z: torch.Tensor) -> torch.Tensor:
    """
    z: (batch, dim), assumed already L2-normalized along dim=-1.

    Encourages:
      - feature-wise mean ≈ 0
      - feature-wise variance ≈ 1
    """
    # feature means across the batch
    mean = z.mean(dim=0)                    # (dim,)
    # feature variances across the batch
    var = z.var(dim=0, unbiased=False)      # (dim,)

    mean_loss = (mean ** 2).mean()          # want mean -> 0
    var_loss  = ((var - 1.0) ** 2).mean()   # want var -> 1

    return mean_loss + var_loss    