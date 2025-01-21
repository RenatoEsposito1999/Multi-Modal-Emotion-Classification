from opts_eeg import parse_opts
import torch
import os
from utils.set_seed import set_random_seed
from datasets.generate_dataloader import get_dataloaders
from torcheeg.models import FBCCNN
from torch import nn
from datasets.seediv_dataset import generate_dataset_SEEDIV
from training_validation import training_validation
from testing import test


if __name__ == '__main__': 
    opt = parse_opts()
    
    if opt.device != 'cpu':
        opt.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
    if not os.path.exists(opt.result_path):
        os.makedirs(opt.result_path)
        
        
    set_random_seed(opt.manual_seed)
    
    dataset = generate_dataset_SEEDIV(opt.path_eeg, opt.path_cached)

    
    model = FBCCNN(num_classes=4, in_channels=4, grid_size=(9, 9)).to(opt.device)
    
    train_loader, val_loader, test_loader = get_dataloaders(dataset, 42, opt.batch_size)
    
  
    loss_fn = nn.CrossEntropyLoss()
    
    
    if not opt.no_train or not opt.no_val:
        training_validation(opt,  model,  train_loader, val_loader, loss_fn)
        
        
    if not opt.test:
        test(opt, model, test_loader, loss_fn)
        
    
        
        
        
        
    
    
    
    
    
        
    