import sys
sys.path.append('../EEG_model')
sys.path.append('../audio_video_emotion_recognition_model')

from datasets.generate_dataset_RAVDESS import get_training_set_RAVDESS
from opts_meta_model import parse_opts 
from torcheeg.models import FBCCNN
from datasets.seediv_dataset import generate_dataset_SEEDIV
from Multimodal_transformer.MultimodalTransformer import MultimodalTransformer
from EmotionStackingClassigier import EmotionStackingClassifier
from utils import transforms
from Data_preprocessing import input_preprocessing_predict
import random

import torch
import numpy as np
from torch import nn
#from sklearn.linear_model import LogisticRegression
from torch.utils.data import DataLoader, Dataset
#from sklearn.metrics import accuracy_score
from tqdm import tqdm

dict_label = {
    0: "Neutral", 
    1: "Happy", 
    2: "Angry",
    3: "Sad"
}
def predict_testing(opts):
    # Initialize models
    model_av = MultimodalTransformer(
        opts.n_classes, 
        seq_length=opts.sample_duration,
        pretr_ef=opts.pretrain_path,
        num_heads=opts.num_heads
    )

    if opts.device != 'cpu':
        model_av = model_av.to(opts.device)
        model_av = nn.DataParallel(model_av, device_ids=None)
        pytorch_total_params = sum(p.numel() for p in model_av.parameters() if p.requires_grad)
        print("Total number of trainable parameters: ", pytorch_total_params)

    # Load models and datasets
    model_eeg = FBCCNN(num_classes=4, in_channels=4, grid_size=(9, 9)).to(opts.device)
    eeg_dataset = generate_dataset_SEEDIV(opts.path_eeg, opts.path_cached)

    # Set models to evaluation mode
    model_av.eval()
    model_eeg.eval()

    # Load weights
    weights = torch.load('results/Complete_model.pth')
    model_av.load_state_dict(weights['model_av_state_dict'])
    model_eeg.load_state_dict(weights['model_eeg_state_dict'])
    model_eeg.to(opts.device)

    # Initialize meta-model
    MetaModel = EmotionStackingClassifier(
        model1=model_av,
        model2=model_eeg,
        opts=opts,
        max_iter=1000
    )
    MetaModel.meta_model = weights['meta_model_state_dict']
    MetaModel.eval()

    # Input preprocessing
    video_audio_path = "./raw_data_video/sad_correct.mp4"
    audio_var, video_var = input_preprocessing_predict.preprocessing_audio_video(
        video_audio_path,
        video_norm_value=opts.video_norm_value,
        batch_size=1
    )

    # Load EEG data
    loader_eeg = DataLoader(
        eeg_dataset, 
        batch_size=1,
        shuffle=False,
        pin_memory=True
    )

    # Organize EEG data
    print("Reached data synchronization...")
    labels = MetaModel.organize_by_labels(loader_eeg)

    # Select a random EEG sample deterministically
    sample_data = (((labels[3])[0]).unsqueeze(0))
    sample_data = sample_data.to(opts.device)
    print(next(model_eeg.parameters()).is_cuda)

    # Perform predictions
    print("Initializing prediction step")
    with torch.no_grad():
        eeg_predict = model_eeg(sample_data)
        av_predict = model_av(audio_var, video_var)
        final_prediction = MetaModel.predict(audio_var,video_var,sample_data)

    # Output predictions
    print(f"EEG prediction: {eeg_predict}")
    print(f"AV prediction: {av_predict}")
    print(f"Final prediction: {final_prediction}: {dict_label[final_prediction[0]]}")