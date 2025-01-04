import sys
sys.path.append('../../EEG_model')
sys.path.append('../../audio_video_emotion_recognition_model')

from torcheeg.models import FBCCNN
from Multimodal_transformer.MultimodalTransformer import MultimodalTransformer
import torch
from torch import nn

def generate_models(opts):
    model_eeg = FBCCNN(num_classes=4, in_channels=4, grid_size=(9, 9)).to(opts.device)
    
    model_av = MultimodalTransformer(
        opts.n_classes, 
        seq_length=opts.sample_duration,
        pretr_ef=opts.pretrain_path,
        num_heads=opts.num_heads
    )
    
    # Move AV model to appropriate device
    if opts.device != 'cpu':
        model_av = model_av.to(opts.device)
        model_av = nn.DataParallel(model_av, device_ids=None)
        pytorch_total_params = sum(p.numel() for p in model_av.parameters() if p.requires_grad)
        print("Total number of trainable parameters: ", pytorch_total_params)
        
        
    model_eeg.load_state_dict(
        torch.load('../EEG_model/results/best_state.pth')['state_dict']
    )
    
    # Load AV model weights based on device
    av_weights_path = f'../audio_video_emotion_recognition_model/{opts.result_path}/{opts.store_name}_best{"_cpu_" if opts.device == "cpu" else ""}.pth'
    av_state = torch.load(av_weights_path, map_location=opts.device)
    model_av.load_state_dict(av_state['state_dict'])
    
    model_av.eval()
    model_eeg.eval()
    
    return model_av, model_eeg