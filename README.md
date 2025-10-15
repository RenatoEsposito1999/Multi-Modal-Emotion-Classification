[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
# Multi-Modal Emotion Classification
## Index
1. [Prefase](#prefase)
2. [Project Overview](#project-overview)  
3. [Project Structure](#project-structure)  
4. [Datasets](#datasets)  
5. [Dependencies](#dependencies)  
6. [Development environment](#development-environment)  
7. [Setting up the Environment](#setting-up-the-environment)  
8. [How to Run the Project](#how-to-run-the-project)  
   - [Audio-video Emotion Recognition Model](#audio-video-emotion-recognition-model)  
   - [EEG-model](#eeg-model)  
   - [Meta model](#meta-model)  
9. [Test by Yourself](#test-by-yourself)  
10. [Results](#results)  
11. [Contributors](#contributors)  
12. [References](#references)

## Prefase
This is an experimental project for the exam of [Cognitive Robotics](https://elearning.uniparthenope.it/course/view.php?id=2127) of the [University of Naples Parthenope](https://www.uniparthenope.it), whose goal is to create a system in which a humanoid reacts to emotions perceived in relation to a human.
The perception of emotions occurs thanks to an [EEG helmet](https://www.emotiv.com/products/epoc?srsltid=AfmBOoqXqNjy1TJGp1Xeu0thk4qH4JkpPCJ5Gl9mbcTkrahp0_odNl2o) and audio and video sensors installed in the [Pepper](https://corporate-internal-prod.aldebaran.com/en/pepper) robot.
This repository refers to the first part of this project which consists in the creation of a model for emotion recognition.
The second part, which is not on github, consists in defining behaviors on the robot pepper based on emotions detected by the model.

## Project Overview 
This project focuses on developing a multi-modal emotion classification system combining audio, video and EEG inputs. Two deep learning models and a meta model are integrated to achieve this:

![Alt text](/architecture_overview.jpg)

1. **Audio-Video Emotion Classification Model**
   Based on the paper "[Learning Audio-Visual Emotional Representations with Hybrid Fusion Strategies](https://arxiv.org/abs/2201.11095#)", this model classifies four emotions using audio and video inputs.

2. **FBCCNN (Feature-Based Convolutional Neural Network)**
   Based on the paper "[Emotion Recognition Based on EEG Using Generative Adversarial Nets and Convolutional Neural Network](https://onlinelibrary.wiley.com/doi/10.1155/2021/2520394)", this model uses EEG data to enhance emotion classification.

3. **Meta-model**
   This model receive as input the predictions of the two deep learning models and through a logistic regression function obtain the final prediction that are: neutral, happy, angry, sad.

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

### Development environment
The development was performed in a Linux CentOS environment on the [machine](https://informatica.uniparthenope.it/index.php/en/portfolio-2/251-nuovo-supercalcolatore-a-disposizione-dei-laureandi-dei-cds-di-informatica-e-di-informatica-applicata-ml-e-bd) made available by the [University of Naples Parthenope](https://www.uniparthenope.it).
The machine is equipped with 8 computational nodes each equipped with 32 cores and 192 Giga Bytes of RAM for a total of 296 CPU cores. 4 of these 8 computational nodes are each equipped with 4 GPUs for a total of 16 NVIDIA V100 NVLINK devices. Each of these GPUs is equipped with 5120 CUDA cores and 32GB of RAM for a total of 81920 GPU cores. The computational nodes are connected to each other through a high-performance network.

### Setting up the Environment
At the moment the `.yml` file for the windows environment is not complete, because some libraries related to the audio-video and eeg models are missing, these libraries can be easily downloaded through conda.
The linux environment is complete instead.

To replicate the development environment, you can use [Conda](https://anaconda.org/anaconda/conda).
The `.yml` files required for creating the environment are located in the `envs` directory.



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
   - Training - Testing:
   ```bash
   python main.py --path_eeg [Path of dataset SEED IV]
   ```
   If you have the folder of cached preprocessed dataset seed IV, you can specify it with argument `--path_cached`

   If you want to do only the prediction add the argument `--predict`

   If you want to perform just one of those steps add the arguments `--no-train` or `--test`. For more details see [opts file](/Meta_model/opts_meta_model.py)

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
3. [Github of Learning Audio-Visual Emotional Representations with Hybrid Fusion Strategies](https://github.com/katerynaCh/multimodal-emotion-recognition)
4. [Esposito, R., Mele, V., Verrilli, S., Minopoli, S., D'Errico, L., De Santis, L., & Staffa, M. (2025). Cascade Multi-Modal Emotion Recognition Leveraging Audio-Video and EEG Signals. In: Proceedings of EMPATH-IA 2025: EMpowering PAtients THrough AI, Multimedia, and Explainable HCI 2025. CEUR Workshop Proceedings, vol. 4040, CEUR-WS.org. ISSN 1613-0073.](https://ceur-ws.org/Vol-4040/paper1.pdf)
