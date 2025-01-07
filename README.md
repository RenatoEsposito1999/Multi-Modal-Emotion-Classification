# Cognitive Robotics Project: Multi-Modal Emotion Classification

## Project Overview
METTERE LA FOTO DLL'ARCHITETTURA SPIEGARE IL METAMODEL AGGIUNGERE DA QUALCHE PARTE L'INIDCAZIONE DEI PESI DEI MODELLI PER REPLCARE IL TUTTO. 
INDICARE CHE C'è ANCHE IL README DI AUDIO VIDEO EMOTION CLASS MODEL 
This project focuses on developing a multi-modal emotion classification system that enhances human-robot interaction by combining audio, video and EEG inputs. Two deep learning models are integrated to achieve this:

1. **Audio-Video Emotion Classification Model**
   Based on the paper "[Learning Audio-Visual Emotional Representations with Hybrid Fusion Strategies](https://arxiv.org/abs/2201.11095#)", this model classifies four emotions using audio and video inputs.

2. **FBCCNN (Feature-Based Convolutional Neural Network)**
   Based on the paper "[Emotion Recognition Based on EEG Using Generative Adversarial Nets and Convolutional Neural Network](https://onlinelibrary.wiley.com/doi/10.1155/2021/2520394)", this model uses EEG data to enhance emotion classification.

## Project Structure
```
project-root
│
├── src
│   ├── audio_video_model        # Implementation of the audio-video model
│   ├── fbccnn_model             # Implementation of the FBCCNN model
│   └── data_processing          # Scripts for data preprocessing
│
├── data                         # Dataset for audio, video, and EEG inputs
├── results                      # Reports and experiment outputs
├── envs                         # Conda environment configuration files
├── docs                         # Additional documentation
└── README.md                    # Project documentation (this file)
```
## Datasets
LINKARE I DATASET PER IL TRAINIING. 
## Dependencies
The main dependencies are:
- Python 3.9
- PyTorch 2.6
- Torcheeg 1.1.3

All dependencies are specified in the `.yml` files located in the `envs` directory. 

### Setting up the Environment
To replicate the development environment, you can use Conda. The `.yml` files required for creating the environment are located in the `envs` directory.

To create the environment in a windows system, run:

```bash
conda env create -f envs/environment_windows.yml
conda activate cognitive_robotics_env
```
To create the environment in a linux system, run:
```bash
conda env create -f envs/environment_linux.yml
conda activate cognitive_robotics_env
```
## How to Run the Project

1. **Data Preprocessing:**
   ```bash
   python src/data_processing/preprocess.py
   ```

2. **Train the Audio-Video Model:**
   ```bash
   python src/audio_video_model/train.py
   ```

3. **Train the FBCCNN Model:**
   ```bash
   python src/fbccnn_model/train.py
   ```

4. **Test the Integrated System:**
   ```bash
   python test_combined_model.py
   ```

## Results
The following metrics are plotted:
- **For training:** accuracy and loss.
- **For validation:** accuracy and loss.
- **For testing:** accuracy, loss, and confusion matrix.

Detailed plots for the audio-video model can be found in the `audio_video_emotion_recognition_model/Image` directory, while plots for the EEG model are available in the `EEG_model/Images` directory.


## Contributors
- Esposito Renato (me)
- [Mele Vincenzo](https://github.com/MeleVincenzo)
- [Verrilli Stefano](https://github.com/StefanoVerrilli)

## References
1. [Learning Audio-Visual Emotional Representations with Hybrid Fusion Strategies](https://arxiv.org/abs/2201.11095#)
2. [Emotion Recognition Based on EEG Using Generative Adversarial Nets and Convolutional Neural Network](https://onlinelibrary.wiley.com/doi/10.1155/2021/2520394)
