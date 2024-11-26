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
from training_preprocessing.eeg_preprocessing import EEGDataset,  create_dataset_from_file_npz, save_dataset_to_npz



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
    
    '''EEGDataset_train = SimulatedEEGDataset(num_samples=1920)
    EEGDataset_validation = SimulatedEEGDataset(num_samples=480)
    EEGDataset_testing = SimulatedEEGDataset(num_samples=480)'''
    print("Creo il training set: ")
    
    
    #Modificare il size del training, validation e testing del EEG:
    #Traning = 756
    #Validation = 216
    #Testing = 108
    #Valutare se lasciare il contrastive
    #Fatto questo modificare il validation val_epoch_multimodal del main.
 
    
    # Specifica i nomi dei file CSV da cercare
    required_files = ["EEGTrain.npz", "EEGVal.npz", "EEGTest.npz"]
    
    # Directory in cui cercare i file (usa la directory corrente se non specificato)
    directory = "./"
    
    # Verifica se i file esistono
    found_files = [file for file in required_files if os.path.exists(os.path.join(directory, file))]
    
    if len(found_files) == len(required_files):
        print("OK")
        EEGDataset_train = create_dataset_from_file_npz(required_files[0])
        EEGDataset_val = create_dataset_from_file_npz(required_files[1])
        EEGDataset_test = create_dataset_from_file_npz(required_files[2])
            
    else:
        EEGDataset_complete = EEGDataset(path="C:/Users/Vince/Desktop/COGNITIVE_ROBOTICS/datasets/SEED_IV/SEED_IV/eeg_raw_data")
        EEGDataset_train, EEGDataset_val, EEGDataset_test = torch.utils.data.random_split(EEGDataset_complete, [756, 216, 108])
        print(len(EEGDataset_train))
        print(len(EEGDataset_val))
        print(len(EEGDataset_test))
        save_dataset_to_npz(EEGDataset_train, "./EEGTrain.npz")
        save_dataset_to_npz(EEGDataset_val, "./EEGVal.npz")
        save_dataset_to_npz(EEGDataset_test, "./EEGTest.npz")
    
                      
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
            print(len(training_data))
        
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
                ['epoch', 'loss', 'prec1_audio_video','prec1_eeg', 'lr'])
            train_batch_logger = Logger(
                os.path.join(opt.result_path, 'train_batch'+str(fold)+'.log'),
                ['epoch', 'batch', 'iter', 'loss', 'prec1_audio_video','prec1_eeg', 'lr'])
            
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
                EEGDataset_val,
                batch_size=opt.batch_size,
                shuffle=False,
                num_workers=opt.n_threads,
                pin_memory=True)
        
            val_logger = Logger(
                    os.path.join(opt.result_path, 'val'+str(fold)+'.log'), ['epoch', 'loss', 'prec1_audio_video','prec1_eeg'])
            test_logger = Logger(
                    os.path.join(opt.result_path, 'test'+str(fold)+'.log'), ['epoch', 'loss', 'prec1_audio_video','prec1_eeg'])

            
        best_prec1_audio_video = 0
        best_prec1_eeg = 0
        best_loss = 1e10
        if opt.resume_path:
            print('loading checkpoint {}'.format(opt.resume_path))
            checkpoint = torch.load(opt.resume_path)
            assert opt.arch == checkpoint['arch']
            best_prec1_audio_video = checkpoint['best_prec1_audio_video']
            best_prec1_eeg = checkpoint['best_prec1_eeg']
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
                    'best_prec1_audio_video': best_prec1_audio_video,
                    'best_prec1_eeg': best_prec1_eeg
                    }
                save_checkpoint(state, False, opt, fold, train=True)
            
            if not opt.no_val:
                
                validation_loss, prec1_audio_video, prec1_eeg = val_epoch_multimodal(EEGDataLoader_val, i, val_loader, model, criterion, opt,
                                            val_logger)
                is_best = prec1_audio_video > best_prec1_audio_video and prec1_eeg > best_prec1_eeg
                best_prec1_audio_video = max(prec1_audio_video, best_prec1_audio_video)
                best_prec1_eeg = max(prec1_eeg, best_prec1_eeg)
                state = {
                'epoch': i,
                'arch': opt.arch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_prec1_audio_video': best_prec1_audio_video,
                'best_prec1_eeg': best_prec1_eeg
                }
               
                save_checkpoint(state, is_best, opt, fold, train=False)

        # Testing Phase       
        if opt.test:

            test_logger = Logger(
                    os.path.join(opt.result_path, 'test'+str(fold)+'.log'), ['epoch', 'loss', 'prec1_audio_video','prec1_eeg'])

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
                EEGDataset_test,
                batch_size=opt.batch_size,
                shuffle=True,
                num_workers=opt.n_threads,
                pin_memory=True
            )
            
            test_loss, test_prec1_audio_video, test_prec1_eeg = val_epoch_multimodal(EEGDataLoader_test, 100, test_loader, model, criterion, opt, test_logger)
            
            with open(os.path.join(opt.result_path, 'test_set_bestval'+str(fold)+'.txt'), 'a') as f:
                    f.write('Prec1_audio_video: ' + str(test_prec1_audio_video)+ '; Prec1_eeg: ' + str(test_prec1_eeg) + '; Loss: ' + str(test_loss))
