import torch
import torch.nn as nn

from Multimodal_transformer.Preprocessing_CNN.Audio_preprocessing import AudioCNNPool
from Multimodal_transformer.Preprocessing_CNN.EEG_preprocessing import EEGCNNPreprocessor
from Multimodal_transformer.Preprocessing_CNN.Video_preprocessing import EfficientFaceTemporal

from Multimodal_transformer.Transformers.Transformer_funcs import EEGTransformerEncoder
from Multimodal_transformer.Transformers.Transformer_funcs import AttentionBlock

class MultimodalTransformer(nn.Module):
    def __init__(self,num_classes=4,seq_length=15,pretr_ef='None',num_heads=1):
        super(MultimodalTransformer,self).__init__()

        self.embeds_dim = 128
        self.input_dim_video = 128
        self.input_dim_audio = 128
        self.input_dim_eeg = 128

        self.audio_preprocessing = AudioCNNPool(num_classes=num_classes)
        self.video_preprocessing = EfficientFaceTemporal([4, 8, 4], [29, 116, 232, 464, 1024], num_classes, seq_length)
        self.EEG_preprocessing = EEGCNNPreprocessor(d_model=self.embeds_dim, num_channels=62, cnn_out_channels=self.embeds_dim)

        self.av = AttentionBlock(in_dim_k=self.input_dim_video, in_dim_q=self.input_dim_audio, out_dim=self.embeds_dim, num_heads=num_heads)
        self.va = AttentionBlock(in_dim_k=self.input_dim_audio, in_dim_q=self.input_dim_video, out_dim=self.embeds_dim, num_heads=num_heads)

        self.EEG_Transformer = EEGTransformerEncoder(d_model=self.embeds_dim,num_heads=num_heads,sequence_length=128)

        self.classifier_audio_video = nn.Sequential(
                    nn.Linear(self.embeds_dim*2, num_classes),
                )
        
        self.classifier_eeg= nn.Sequential(
                    nn.Linear(self.embeds_dim, num_classes),
                )
        
        self.softmax = nn.Softmax(dim=1)
        
        
        
    def forward(self,x_audio,x_visual,x_eeg):

        x_audio = self.audio_preprocessing.forward_stage1(x_audio)
        proj_x_a = self.audio_preprocessing.forward_stage2(x_audio)

        x_visual = self.video_preprocessing.forward_features(x_visual) 
        x_visual = self.video_preprocessing.forward_stage1(x_visual)
        proj_x_v = self.video_preprocessing.forward_stage2(x_visual)

        proj_x_a = proj_x_a.permute(0, 2, 1)
        proj_x_v = proj_x_v.permute(0, 2, 1)
        
        h_av = self.av(proj_x_v, proj_x_a)
        h_va = self.va(proj_x_a, proj_x_v)

        audio_pooled = h_av.mean([1]) #mean accross temporal dimension
        video_pooled = h_va.mean([1])
        proj_x_eeg = self.EEG_preprocessing.forward(x_eeg)
        eeg_pooled = self.EEG_Transformer.forward(proj_x_eeg)
        
        concat_audio_video = torch.cat((audio_pooled, video_pooled), dim=-1)
        
        classification_audio_video = self.classifier_audio_video(concat_audio_video)
        
        classification_eeg = self.classifier_eeg(eeg_pooled)
        
        probabilities_audio_video = self.softmax(classification_audio_video)
        probabilities_eeg = self.softmax(classification_eeg)
        
        audio_video_confidence, audio_video_labels = torch.max(probabilities_audio_video, dim=1)
        eeg_confidence, eeg_labels = torch.max(probabilities_eeg, dim=1)
        
        th = 0.6
        max_output = -1 * torch.ones(audio_video_labels.size(0))
        
        for i in range(audio_video_confidence.size(0)):
            if (audio_video_confidence[i] > th and eeg_confidence[i] > th):
                if(audio_video_confidence[i] > eeg_confidence[i]):
                    max_output[i] = audio_video_labels[i]
                else:
                    max_output[i] = eeg_labels[i]
            elif(audio_video_confidence[i] > th):
                max_output[i] = audio_video_labels[i]
            elif(eeg_confidence[i] > th):
                max_output[i] = eeg_labels[i]
            elif(audio_video_confidence[i] > eeg_confidence[i]):
                    max_output[i] = audio_video_labels[i]
            else:
                    max_output[i] = eeg_labels[i]
        
            
        return audio_pooled,video_pooled,eeg_pooled,classification_audio_video, classification_eeg, max_output