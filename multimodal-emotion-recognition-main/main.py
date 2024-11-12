# -*- coding: utf-8 -*-
"""
Created on Mon Oct 25 14:07:29 2021

@author: chumache
"""
import os
import json
import numpy as np
import torch
from torch import nn, optim
from torch.optim import lr_scheduler

from opts import parse_opts
from model import generate_model
import transforms 
from dataset import get_training_set, get_validation_set, get_test_set
from utils import Logger, adjust_learning_rate, save_checkpoint
from train import train_epoch_multimodal
from validation import val_epoch_multimodal
import time

from SimulatedDataset import SimulatedEEGDataset


if __name__ == '__main__':
    opt = parse_opts()
    n_folds = 1
    test_accuracies = []
    
    if opt.device != 'cpu':
        opt.device = 'cuda' if torch.cuda.is_available() else 'cpu'  

    pretrained = opt.pretrain_path != 'None'    
    
    if not os.path.exists(opt.result_path):
        os.makedirs(opt.result_path)
        
    opt.arch = '{}'.format(opt.model)  
    opt.store_name = '_'.join([opt.dataset, opt.model, str(opt.sample_duration)])
    
    EEGDataset_train = SimulatedEEGDataset(num_samples=1920)
    EEGDataset_validation = SimulatedEEGDataset(num_samples=480)
    EEGDataset_testing = SimulatedEEGDataset(num_samples=480)
   
            
    for fold in range(n_folds):
        print(opt)

        torch.manual_seed(opt.manual_seed)
        model, parameters = generate_model(opt)
        
        #Define loss for training
        criterion = nn.CrossEntropyLoss()
        criterion = criterion.to(opt.device)
        
        #Training Phase
        if not opt.no_train:
            
            video_transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotate(),
                transforms.ToTensor(opt.video_norm_value)])
        
            training_data = get_training_set(opt, spatial_transform=video_transform) 
        
            train_loader = torch.utils.data.DataLoader(
                training_data,
                batch_size=opt.batch_size,
                shuffle=True,
                num_workers=opt.n_threads,
                pin_memory=True)
            
            EEGDataLoader_train = torch.utils.data.DataLoader(
                EEGDataset_train,
                batch_size=opt.batch_size,
                shuffle=True,
                num_workers=opt.n_threads,
                pin_memory=True
             )
        
            train_logger = Logger(
                os.path.join(opt.result_path, 'train'+str(fold)+'.log'),
                ['epoch', 'loss', 'prec1', 'lr'])
            train_batch_logger = Logger(
                os.path.join(opt.result_path, 'train_batch'+str(fold)+'.log'),
                ['epoch', 'batch', 'iter', 'loss', 'prec1', 'lr'])
            
            optimizer = optim.SGD(
                parameters,
                lr=opt.learning_rate,
                momentum=opt.momentum,
                dampening=opt.dampening,
                weight_decay=opt.weight_decay,
                nesterov=False)
            
            scheduler = lr_scheduler.ReduceLROnPlateau(
                optimizer, 'min', patience=opt.lr_patience)
        
        #Validation Phase
        if not opt.no_val:
            video_transform = transforms.Compose([
                transforms.ToTensor(opt.video_norm_value)])     
        
            validation_data = get_validation_set(opt, spatial_transform=video_transform)
            
            val_loader = torch.utils.data.DataLoader(
                validation_data,
                batch_size=opt.batch_size,
                shuffle=False,
                num_workers=opt.n_threads,
                pin_memory=True)
            
            EEGDataLoader_val = torch.utils.data.DataLoader(
                EEGDataset_validation,
                batch_size=opt.batch_size,
                shuffle=False,
                num_workers=opt.n_threads,
                pin_memory=True)
        
            val_logger = Logger(
                    os.path.join(opt.result_path, 'val'+str(fold)+'.log'), ['epoch', 'loss', 'prec1'])
            test_logger = Logger(
                    os.path.join(opt.result_path, 'test'+str(fold)+'.log'), ['epoch', 'loss', 'prec1'])

            
        best_prec1 = 0
        best_loss = 1e10
        if opt.resume_path:
            print('loading checkpoint {}'.format(opt.resume_path))
            checkpoint = torch.load(opt.resume_path)
            assert opt.arch == checkpoint['arch']
            best_prec1 = checkpoint['best_prec1']
            opt.begin_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])

        #Start training and validation
        for i in range(opt.begin_epoch, opt.n_epochs + 1):
            if not opt.no_train:
                adjust_learning_rate(optimizer, i, opt)
                train_epoch_multimodal(i, train_loader, model, criterion, optimizer, opt,
                            train_logger, train_batch_logger, EEGDataLoader_train)
                state = {
                    'epoch': i,
                    'arch': opt.arch,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'best_prec1': best_prec1
                    }
                save_checkpoint(state, False, opt, fold, train=True)
            
            if not opt.no_val:
                
                validation_loss, prec1 = val_epoch_multimodal(EEGDataLoader_val, i, val_loader, model, criterion, opt,
                                            val_logger)
                is_best = prec1 > best_prec1
                best_prec1 = max(prec1, best_prec1)
                state = {
                'epoch': i,
                'arch': opt.arch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_prec1': best_prec1
                }
               
                save_checkpoint(state, is_best, opt, fold, train=False)

        # Testing Phase       
        if opt.test:

            test_logger = Logger(
                    os.path.join(opt.result_path, 'test'+str(fold)+'.log'), ['epoch', 'loss', 'prec1'])

            video_transform = transforms.Compose([
                transforms.ToTensor(opt.video_norm_value)])
                
            test_data = get_test_set(opt, spatial_transform=video_transform) 
        
            #load best model
            best_state = torch.load('%s/%s_best' % (opt.result_path, opt.store_name)+str(fold)+'.pth')
            #best_state = torch.load('c:/Users/Vince/Desktop/COGNITIVE_ROBOTICS/cognitive-robotics-project/multimodal-emotion-recognition-main/lt_1head_moddrop_2.pth', map_location=torch.device('cpu'))
            model.load_state_dict(best_state['state_dict'])
        
            test_loader = torch.utils.data.DataLoader(
                test_data,
                batch_size=opt.batch_size,
                shuffle=False,
                num_workers=opt.n_threads,
                pin_memory=True)
            
            EEGDataLoader_test = torch.utils.data.DataLoader(
                EEGDataset_testing,
                batch_size=opt.batch_size,
                shuffle=True,
                num_workers=opt.n_threads,
                pin_memory=True
            )
            
            test_loss, test_prec1 = val_epoch_multimodal(EEGDataLoader_test, 100, test_loader, model, criterion, opt, test_logger)
            
            with open(os.path.join(opt.result_path, 'test_set_bestval'+str(fold)+'.txt'), 'a') as f:
                    f.write('Prec1: ' + str(test_prec1) + '; Loss: ' + str(test_loss))
            test_accuracies.append(test_prec1) 
                
            
    with open(os.path.join(opt.result_path, 'test_set_bestval.txt'), 'a') as f:
        f.write('Prec1: ' + str(np.mean(np.array(test_accuracies))) +'+'+str(np.std(np.array(test_accuracies))) + '\n')
