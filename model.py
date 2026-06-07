import torch
import torch.nn as nn
import torch.nn.functional as F

class EZPC(torch.nn.Module):
    def __init__(self, concept_matrix, weight):
        super(EZPC, self).__init__()
        # Set class parameters
        self.concept_matrix = concept_matrix
        self.weight = weight

        # Init the A matrix
        self.A = nn.Parameter(concept_matrix.T.clone().detach(), requires_grad=True)

    @torch.no_grad()
    def normalize_weights(self):
        # Projects the columns of A back to the unit sphere
        self.A.copy_(self.A / (self.A.norm(dim=0, keepdim=True) + 1e-8))

    def forward(self, vx, ty):
        # Matching loss
        matching_loss = F.mse_loss(self.A, self.concept_matrix.T)

        # Compute CLIP's probability distribution
        clip_probs = F.softmax(vx @ ty.T, dim=1)

        # Compute CBM's probability distribution
        ezpc_logits = (vx @ self.A @ self.A.T @ ty.T)
        ezpc_log_probs = F.log_softmax(ezpc_logits, dim=1)

        # Calculate KL divergence
        reconstruction_loss = F.kl_div(
            ezpc_log_probs,
            clip_probs,
            reduction='batchmean'
        )

        # Total loss
        total_loss = matching_loss + self.weight * reconstruction_loss

        return matching_loss, reconstruction_loss, total_loss