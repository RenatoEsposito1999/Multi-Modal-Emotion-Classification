from datasets.ravdess import RAVDESS
from opts import parse_opts
import transforms

def get_test_set(opt, spatial_transform=None, audio_transform=None):
    assert opt.dataset in ['RAVDESS'], print('Unsupported dataset: {}'.format(opt.dataset))
    assert opt.test_subset in ['val', 'test']

    if opt.test_subset == 'val':
        subset = 'validation'
    elif opt.test_subset == 'test':
        subset = 'testing'
    if opt.dataset == 'RAVDESS':
        test_data = RAVDESS(
            opt.annotation_path,
            subset,
            spatial_transform=spatial_transform, data_type='audiovisual',audio_transform=audio_transform)
    print("TEST DATA: ", test_data) 
    # TEST DATA:  RAVDESS Dataset - Subset: , Size: 480, 
    # Spatial Transform: <transforms.Compose object at 0x148a745e0>, 
    # Audio Transform: None, Data Type: audiovisual       
    return test_data

opt = parse_opts()
video_transform = transforms.Compose([
                transforms.ToTensor(opt.video_norm_value)])
test_data = get_test_set(opt, spatial_transform=video_transform)
for video in test_data.data:
    if("01-01-07-01-02-01-05" in video["video_path"]):
        print(video)
#print(test_data.data[0]["video_path"])
# TEST DATA:  RAVDESS Dataset - Subset: , Size: 480, 
# Spatial Transform: <transforms.Compose object at 0x11fdd9490>, 
# Audio Transform: None, Data Type: audiovisual
#{'video_path': '/Users/giuseppefiorillo/Documents/RAVDESS/ACTOR06/02-01-08-01-02-01-06_facecroppad.npy', 
# 'audio_path': '/Users/giuseppefiorillo/Documents/RAVDESS/ACTOR06/03-01-08-01-02-01-06_croppad.wav', 
# 'label': 7}

print(test_data.data_type)

import numpy as np

# Specifica il percorso del file .npy
file_npy_path = test_data.data[0]["video_path"]  # Sostituisci con il percorso reale del tuo file .npy

# Carica il file .npy
data = np.load(file_npy_path, allow_pickle=True)  # Assicurati di impostare allow_pickle su True se il tuo file .npy contiene oggetti Python

# Visualizza il contenuto e la forma dell'array
print("Forma dell'array:", data.shape)

from scipy.io import wavfile
import numpy as np

# Specifica il percorso del file audio WAV
file_wav_path = test_data.data[0]["audio_path"]  # Sostituisci con il percorso reale del tuo file WAV

# Carica il file WAV
sample_rate, audio_data = wavfile.read(file_wav_path)

# Visualizza informazioni sull'audio
print(f"Frequenza di campionamento: {sample_rate} Hz")
print(f"Forma dell'array audio: {audio_data.shape}")
print(f"Durata dell'audio: {len(audio_data) / sample_rate:.2f} secondi")
