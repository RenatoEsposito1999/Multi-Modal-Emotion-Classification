import sys
sys.path.append('../EEG_model')
sys.path.append('../audio_video_emotion_recognition_model')

from datasets.seediv_dataset import generate_dataset_SEEDIV
from Data_preprocessing import input_preprocessing_predict
import torch
from torch.utils.data import DataLoader


dict_label = {
    0: "Neutral", 
    1: "Happy", 
    2: "Angry",
    3: "Sad"
}
def predict_testing(opts, stacking_classifier):
    
    eeg_dataset = generate_dataset_SEEDIV(opts.path_eeg, opts.path_cached)

    # Load weights
    weights = torch.load('results/Complete_model.pth')
    stacking_classifier.model1.load_state_dict(weights['model_av_state_dict'])
    stacking_classifier.model2.load_state_dict(weights['model_eeg_state_dict'])
    stacking_classifier.meta_model = weights['meta_model_state_dict']
    stacking_classifier.eval()

    # Input preprocessing
    video_audio_path = "./raw_data_video/happy_wrong_1.mp4"
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
    labels = stacking_classifier.organize_by_labels(loader_eeg)

    # Select a random EEG sample deterministically
    sample_data = (((labels[1])[0]).unsqueeze(0))
    sample_data = sample_data.to(opts.device)
    print(next(stacking_classifier.model2.parameters()).is_cuda)

    # Perform predictions
    print("Initializing prediction step")
    with torch.no_grad():
        eeg_predict = stacking_classifier.model2(sample_data)
        av_predict = stacking_classifier.model1(audio_var, video_var)
        final_prediction = stacking_classifier.forward(audio_var,video_var,sample_data)

    # Output predictions
    print(f"EEG prediction: {eeg_predict}")
    print(f"AV prediction: {av_predict}")
    print(f"Final prediction: {final_prediction}: {dict_label[final_prediction[0]]}")
    