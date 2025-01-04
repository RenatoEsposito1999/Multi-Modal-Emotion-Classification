import random
import numpy as np
import torch 
from train_meta_model import train_meta_classifier
from predict import predict_testing
from opts_meta_model import parse_opts
from EmotionStackingClassifier import EmotionStackingClassifier
from generate_models import generate_models


def set_random_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

if __name__ == "__main__":
    opts = parse_opts()
    set_random_seed(42)
    
    model_av, model_eeg = generate_models(opts)
    
    stacking_classifier = EmotionStackingClassifier(
        model1=model_av,
        model2=model_eeg,
        batch_size= opts.batch_size,
        max_iter=1000
    )
    
    if not opts.no_train:
        train_meta_classifier(opts, stacking_classifier)
    if opts.predict:
        predict_testing(opts, stacking_classifier)