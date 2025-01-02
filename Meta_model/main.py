import random
import numpy as np
import torch 
from train_meta_model import train_meta_classifier
from predict import predict_testing
from opts_meta_model import parse_opts


def set_random_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

if __name__ == "__main__":
    set_random_seed(42)
    #classifier = train_meta_classifier()
    predict_testing(parse_opts())