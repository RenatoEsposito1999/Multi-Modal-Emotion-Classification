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

    
    def forward(self,audio_embedding,video_embedding,eeg_embedding,sync_labels,async_labels):

        """
        embedding: Tensor of shape (batch_size,embed_dim)
                - Audio embed synced with Video embed
                - EEG embed asynced with the rest of two 
        labels: Tensor of shape (batch_size)
                - Sync_labels: Each label correspond to the class given to audio/video
                - Async_labels: Each label correspond to the EEG class
        """
        audio_embeds = F.normalize(audio_embedding,dim=1)
        video_embeds = F.normalize(video_embedding,dim=1)
        eeg_embeds = F.normalize(eeg_embedding,dim=1)

        video_audio_sim = torch.matmul(video_embeds,audio_embeds.T)/self.temperature
        video_eeg_sim = torch.matmul(video_embeds,eeg_embeds.T)/self.temperature
        audio_eeg_sim = torch.matmul(audio_embeds,eeg_embeds.T)/self.temperature

        # Start computation of similarity for sync source

        sync_mask = sync_labels.unsqueeze(1) == sync_labels.unsqueeze(0)
        pos_mask_sync = sync_mask.float()

        async_mask = async_labels.unsqueeze(1) == sync_labels.unsqueeze(0)
        pos_mask_async = async_mask.float()
        neg_mask_async = 1 - pos_mask_sync

        pos_sim_video_audio = torch.exp(video_audio_sim)*pos_mask_sync
        sum_pos_sim_video_audio = pos_sim_video_audio.sum(1)

        pos_sim_video_eeg = torch.exp(video_eeg_sim) * pos_mask_async
        neg_sim_video_eeg = torch.exp(video_eeg_sim) * neg_mask_async

        pos_sim_audio_eeg = torch.exp(audio_eeg_sim) * pos_mask_async
        neg_sim_audio_eeg = torch.exp(audio_eeg_sim) * neg_mask_async

        sum_pos_sim_video_eeg = pos_sim_video_eeg.sum(1)
        sum_pos_sim_audio_eeg = pos_sim_audio_eeg.sum(1)

        sum_neg_sim_video_eeg = neg_sim_video_eeg.sum(1)
        sum_neg_sim_audio_eeg = neg_sim_audio_eeg.sum(1)

        all_sim_video_audio = torch.exp(video_audio_sim).sum(1)
        all_sim_video_eeg = torch.exp(video_eeg_sim).sum(1)
        all_sim_audio_eeg = torch.exp(audio_eeg_sim).sum(1)

        # Contrastive loss calculation
        # Adding negative pairs for EEG-video and EEG-audio pairs in denominator to repel mismatched labels
        loss_video_audio = -torch.log((sum_pos_sim_video_audio + 1e-8) / (all_sim_video_audio + 1e-8))
        loss_video_eeg = -torch.log((sum_pos_sim_video_eeg + 1e-8) / (all_sim_video_eeg + sum_neg_sim_video_eeg + 1e-8))
        loss_audio_eeg = -torch.log((sum_pos_sim_audio_eeg + 1e-8) / (all_sim_audio_eeg + sum_neg_sim_audio_eeg + 1e-8))

        # Final loss is the mean across all pairs
        loss = (loss_video_audio.mean() + loss_video_eeg.mean() + loss_audio_eeg.mean()) / 3.0
        return loss



