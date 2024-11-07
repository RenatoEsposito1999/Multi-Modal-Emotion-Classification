import torch
import torch.nn as nn
import torch.nn.functional as F

class SupervisedConstrastiveLoss(nn.Module):
    def __init__(self,temperature=0.5):
        super(SupervisedConstrastiveLoss,self).__init__()
        self.temperature = temperature

    def forward(self, embeddings, labels):
        """
        embeddings: Tensor of shape (batch_size * 2, d_model)
                    - The first half of the embeddings tensor is assumed to be from one modality (e.g., audio),
                      and the second half from another modality (e.g., video).
        labels: Tensor of shape (batch_size)
                - Each label corresponds to one pair of embeddings from different modalities.
        """
        # Normalize the embeddings
        embeddings = F.normalize(embeddings, dim=1)
        # Split the embeddings tensor into the two modalities
        modality1, modality2 = embeddings[:3], embeddings[3:]
        print(modality1, modality2)
        # Compute cosine similarity between each pair from modality1 and modality2
        cosine_sim = torch.matmul(modality1, modality2.T) / self.temperature
        print("cosine_sim: ", cosine_sim)
        # Create a mask to indicate positive pairs (same label across the two modalities)
        labels_eq = labels.unsqueeze(1) == labels.unsqueeze(0)
        mask = labels_eq.float()  # Mask has shape (batch_size, batch_size)
        
        # Calculate positive similarity: only cross-modality pairs with the same label
        pos_sim = cosine_sim * mask
        sum_pos_sim = pos_sim.sum(1)
        # Calculate the denominator for normalization (sum of all similarities in each row)
        exp_sim = torch.exp(cosine_sim)
        denominator = exp_sim.sum(1)
        # Calculate supervised contrastive loss
        loss = -torch.log(sum_pos_sim / denominator)
        return loss.mean()