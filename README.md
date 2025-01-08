# Cognitive Robotics Project: Multi-Modal Emotion Classification

## Project Overview 
This project focuses on developing a multi-modal emotion classification system that enhances human-robot interaction by combining audio, video and EEG inputs. Two deep learning models and a meta model are integrated to achieve this:

![Alt text](/architecture_overview.jpg)

1. **Audio-Video Emotion Classification Model**
   Based on the paper "[Learning Audio-Visual Emotional Representations with Hybrid Fusion Strategies](https://arxiv.org/abs/2201.11095#)", this model classifies four emotions using audio and video inputs.

2. **FBCCNN (Feature-Based Convolutional Neural Network)**
   Based on the paper "[Emotion Recognition Based on EEG Using Generative Adversarial Nets and Convolutional Neural Network](https://onlinelibrary.wiley.com/doi/10.1155/2021/2520394)", this model uses EEG data to enhance emotion classification.

3. **Meta-model**
   This model receive in inputs the predictions of the two deep learning models, and through a Logistic Regression function obtain the final prediction, that are: Neutral, Happy, Angry, Sad

## Project Structure
```
project-root
│
├───audio_video_emotion_recognition_model
│   │   
│   ├───datasets       
│   ├───Data_preprocessing        
│   ├───Image      
│   ├───Multimodal_transformer 
│   │   ├───Preprocessing_CNN  
│   │   │   ├───Preprocessing_utils
│   │   │   
│   │   ├───Transformers
│   │           
│   ├───results      
│   ├───utils
│                 
├───EEG_model  
│   ├───datasets   
│   ├───Images     
│   ├───results       
│   ├───utils 
│     
├───envs    
├───Meta_model        
│   ├───results      
└───Shared
```
## Datasets
The dataset used for training of audio-video emotion recognition model is RAVDESS, that can be downloaded [here](https://zenodo.org/records/1188976#.YkgJVijP2bh)

The dataset used for training the eeg-model is SEED-IV, that can be requested [here](https://bcmi.sjtu.edu.cn/~seed/seed-iv.html#)
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
Models must be individually trained before the meta model can be trained.
1. **Audio-video remotion recognition model:**
   ```bash
   cd audio_video_emotion_recognition_model
   ```

   Before use the model it's mandatory to perform the preprocessing steps:

   Inside each of three scripts, specify the path (full path!) where you have downloaded the data.
   Then run:
   ```python
   cd ravdess_preprocessing
   python extract_faces.py
   python extract_audios.py
   python create_annotations.py
   ```
   As a result you will have annotations.txt file that you can use further for training.
   - Training - Validation - Testing:
   ```bash
   python main.py
   ```
   If you want to perform just one of those steps add the arguments `--no-train` or `--no-val` or `--test`. For more details see [opts file](/audio_video_emotion_recognition_model/opts_audio_video.py)
   - Prediction: (For those who want to try the single model)
   ```bash
   python main.py --no-train --no-val --test --predict
   ```

2. **EEG-model:**
   - Training - Validation - Testing:
   ```bash
   python main.py --path_eeg [Path of dataset SEED IV]
   ```
   If you have the folder of cached preprocessed dataset seed IV, you can specify it with argument `--path_cached`

   If you want to perform just one of those steps add the arguments `--no-train` or `--no-val` or `--test`. For more details see [opts file](/EEG_model/opts_eeg.py)

3. **Meta model:**
   - Training - Validation - Testing:
   ```bash
   python main.py --path_eeg [Path of dataset SEED IV]
   ```
   If you have the folder of cached preprocessed dataset seed IV, you can specify it with argument `--path_cached`

   If you want to perform just one of those steps add the arguments `--no-train` or `--no-val` or `--test`. For more details see [opts file](/Meta_model/opts_meta_model.py)

## Test by yourself
If you want to test by yourself you can find the pretrained weights of the models in the `results` directories of the respective models. 

## Results
The following metrics are plotted:
- **For training:** accuracy and loss.
- **For validation:** accuracy and loss.
- **For testing:** accuracy, loss, and confusion matrix.

Detailed plots for the audio-video model can be found in the `audio_video_emotion_recognition_model/Image` directory, while plots for the EEG model are available in the `EEG_model/Images` directory. 

For the meta-model you can visualize in the `Meta_model/Images` the confusion matrix computed using the test set of audio-video and eeg.


## Contributors
- Esposito Renato (me)
- [Mele Vincenzo](https://github.com/MeleVincenzo)
- [Verrilli Stefano](https://github.com/StefanoVerrilli)

## References
1. [Learning Audio-Visual Emotional Representations with Hybrid Fusion Strategies](https://arxiv.org/abs/2201.11095#)
2. [Emotion Recognition Based on EEG Using Generative Adversarial Nets and Convolutional Neural Network](https://onlinelibrary.wiley.com/doi/10.1155/2021/2520394)
