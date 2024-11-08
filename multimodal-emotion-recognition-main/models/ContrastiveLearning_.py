import torch
import torch.nn as nn
import torch.nn.functional as F

# Starting testing phase
'''
        embeds = torch.cat((embeddings,embeddings),dim=0)
        labels = torch.cat((labels,labels),dim=0)
        print("embedss",embeds.shape)
        print("labels",labels.shape)
        print("cosine sim",cosine_sim.shape)
        print("mask:",mask.shape)
'''
class SupervisedContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.5):
        super(SupervisedContrastiveLoss, self).__init__()
        self.temperature = temperature

    def forward(self, audio_embeddings, video_embeddings, labels):
        """
        audio_embeds: Tensor of shape (batch_size, embed_dim) for audio embeddings
        video_embeds: Tensor of shape (batch_size, embed_dim) for video embeddings
        labels: Tensor of shape (batch_size) with integer labels for each pair
        """

        # Normalize embeddings for cosine similarity
        audio_embeds = F.normalize(audio_embeddings, dim=1)
        video_embeds = F.normalize(video_embeddings, dim=1)
        
        # Concatenate audio and video embeddings along the batch dimension
        embeddings = torch.cat([audio_embeds, video_embeds], dim=0)
        labels = torch.cat([labels, labels], dim=0)  # Duplicate labels for both modalities
        
        # Compute cosine similarity matrix
        cosine_sim = torch.matmul(embeddings, embeddings.T) / self.temperature
        
        # Create mask where labels match
        labels_eq = labels.unsqueeze(1) == labels.unsqueeze(0)
        mask = labels_eq.float() - torch.eye(labels_eq.size(0), device=labels.device)
        
        # Positive pair similarity (only between same-label pairs)
        pos_sim = torch.exp(cosine_sim) * mask

        # Normalize each row by subtracting log-sum-exponential of all pairs
        loss = -torch.log(pos_sim.sum(1) / (torch.exp(cosine_sim).sum(1) - torch.exp(cosine_sim.diag())) + 1e-8)
        return loss.mean()