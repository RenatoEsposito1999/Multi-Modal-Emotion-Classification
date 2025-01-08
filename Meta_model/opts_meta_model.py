import argparse

def parse_opts():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda', type=str, help='Specify the device to run. Defaults to cuda, fallsback to cpu')
    parser.add_argument('--path_eeg', default="", type=str, help='Path of seed iv')
    parser.add_argument('--path_cached', default="", type=str, help='Path of cached dataset')
    parser.add_argument('--batch_size', default=8, type=int, help='Batch Size')
    parser.add_argument('--annotation_path', default='../audio_video_emotion_recognition_model/Data_preprocessing/annotations.txt', type=str, help='Annotation file path')
    parser.add_argument('--video_norm_value', default=255, type=int, help='If 1, range of inputs is [0-255]. If 255, range of inputs is [0-1].')
    parser.add_argument('--dataset', default='RAVDESS', type=str, help='Used dataset. Currently supporting Ravdess')
    parser.add_argument('--n_classes', default=4, type=int, help='Number of classes')
    parser.add_argument('--sample_duration', default=15, type=int, help='Temporal duration of inputs, ravdess = 15')
    parser.add_argument('--pretrain_path', default='EfficientFace_Trained_on_AffectNet7.pth.tar', type=str, help='Pretrained model (.pth), efficientface')
    parser.add_argument('--num_heads', default=1, type=int, help='number of heads, in the paper 1 or 4')
    parser.add_argument('--n_threads', default=8, type=int, help='Number of threads for multi-thread loading')
    parser.add_argument('--result_path', default='results', type=str, help='Result directory path')
    parser.add_argument('--store_name', default='RAVDESS_multimodalcnn_15', type=str, help='Name to store checkpoints')
    
    parser.add_argument('--no_train', action='store_true', help='If true, training is not performed.')
    parser.set_defaults(no_train=False)
    parser.add_argument('--no_val', action='store_true', help='If true, validation is not performed.')
    parser.set_defaults(no_val=False)
    parser.add_argument('--test', action='store_true', help='If true, test is performed.')
    parser.set_defaults(test=False)
    parser.add_argument('--predict', action='store_true', help='If true, predict is performed.')
    parser.set_defaults(predict=False)
    parser.add_argument('--test_subset', default='test', type=str, help='Used subset in test (val | test)')
    args = parser.parse_args()

    return args