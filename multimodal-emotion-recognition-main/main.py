# -*- coding: utf-8 -*-
"""
Created on Mon Oct 25 14:07:29 2021

@author: chumache
"""
import os
import numpy as np
import torch
from torch import nn, optim
from torch.optim import lr_scheduler

from opts import parse_opts
from model import generate_model
import transforms 
from dataset import get_training_set, get_validation_set, get_test_set
from utils import Logger, save_checkpoint
from train import train_epoch_multimodal
from validation import val_epoch_multimodal

from training_preprocessing.eeg_preprocessing import EEGDataset,  create_dataset_from_file_npz, save_dataset_to_npz
from training_preprocessing.synchronized_data import Synchronized_data



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
       
    # Specifica i nomi dei file CSV da cercare
    required_files = ["EEGTrain.npz", "EEGVal.npz", "EEGTest.npz"]
    
    # Directory in cui cercare i file (usa la directory corrente se non specificato)
    directory = "./"
    
    # Verifica se i file esistono
    found_files = [file for file in required_files if os.path.exists(os.path.join(directory, file))]
    
    if len(found_files) == len(required_files):
        EEGDataset_train = create_dataset_from_file_npz(required_files[0])
        EEGDataset_val = create_dataset_from_file_npz(required_files[1])
        EEGDataset_test = create_dataset_from_file_npz(required_files[2])
            
    else:
        # Da modificare completamente
        EEGDataset_complete = EEGDataset(path=opt.eeg_dataset_path)
        EEGDataset_train, EEGDataset_val, EEGDataset_test = torch.utils.data.random_split(EEGDataset_complete, [756, 216, 108])
        save_dataset_to_npz(EEGDataset_train, "./EEGTrain.npz")
        save_dataset_to_npz(EEGDataset_val, "./EEGVal.npz")
        save_dataset_to_npz(EEGDataset_test, "./EEGTest.npz")
    
                      
    for fold in range(n_folds):

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
            
            EEGData_train = Synchronized_data(EEGDataset_train)
        
            train_logger = Logger(
                os.path.join(opt.result_path, 'train'+str(fold)+'.log'),
                ['epoch', 'loss', 'prec1', 'lr'])
            train_batch_logger = Logger(
                os.path.join(opt.result_path, 'train_batch'+str(fold)+'.log'),
                ['epoch', 'batch', 'iter', 'loss', 'prec1', 'lr'])
            
            optimizer = optim.Adam(
                            parameters,
                            lr=opt.learning_rate,
                            betas=(0.9, 0.98),
                            eps=1e-9,
                            weight_decay=opt.weight_decay,
                            amsgrad=True
                        )
 
                
            scheduler = lr_scheduler.StepLR(optimizer, 20, 0.1)
        
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
            
            EEGData_val = Synchronized_data(EEGDataset_val)
        
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
            optimizer.load_state_dict(checkpoint["optimizer"])
            
        

        #Start training and validation
        for i in range(opt.begin_epoch, opt.n_epochs + 1):
            if not opt.no_train:
                #adjust_learning_rate(optimizer, i, opt)
                train_epoch_multimodal(i, train_loader, model, criterion, optimizer, scheduler, opt,
                            train_logger, train_batch_logger, EEGData_train)
                scheduler.step()
                state = {
                    'epoch': i,
                    'arch': opt.arch,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'best_prec1': best_prec1,
                    }
                save_checkpoint(state, model,  False, opt, fold, train=True)
            
            if not opt.no_val:
                
                validation_loss, prec1 = val_epoch_multimodal(EEGData_val, i, val_loader, model, criterion, opt,
                                            val_logger)
                is_best = prec1 > best_prec1
                best_prec1 = max(prec1, best_prec1)
                state = {
                'epoch': i,
                'arch': opt.arch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_prec1': best_prec1,
                }
               
                save_checkpoint(state, model, is_best, opt, fold, train=False)

        # Testing Phase       
        if opt.test:

            test_logger = Logger(
                    os.path.join(opt.result_path, 'test'+str(fold)+'.log'), ['epoch', 'loss', 'prec1'])

            video_transform = transforms.Compose([
                transforms.ToTensor(opt.video_norm_value)])
                
            test_data = get_test_set(opt, spatial_transform=video_transform) 
        
            #load best model
            best_state = torch.load('%s/%s_best' % (opt.result_path, opt.store_name)+str(fold)+'.pth')
                       
            model.load_state_dict(best_state['state_dict'])
            
            #QUESTA PARTE DI CODICE SEREVE PER SALVARE I PESI IN CPU
            # Se il modello è avvolto in DataParallel
            '''if isinstance(model, torch.nn.DataParallel):
                model = model.module  # Ottieni il modello originale'''
                
            
            #da metterlo in save checkpoint
            #torch.save(model.module.state_dict(), 'model.pth')  # Salva solo lo state_dict del modello originale
            
            
            
            #state_cpu = torch.load("./model.pth")
            #model.load_state_dict(state_cpu)
            
            
            #OK:
            '''prefix = 'module.'
            n_clip = len(prefix)
            state_dict = {k[len(prefix):] if k.startswith(prefix) else k: v for k, v in best_state["state_dict"].items()}
            model.load_state_dict(state_dict)'''
            #model = torch.load("entire_model.pth", map_location=torch.device('cpu'))
            
             
            test_loader = torch.utils.data.DataLoader(
                test_data,
                batch_size=opt.batch_size,
                shuffle=False,
                num_workers=opt.n_threads,
                pin_memory=True)
            
            EEGData_test = Synchronized_data(EEGDataset_test)
            
            test_loss, prec1 = val_epoch_multimodal(EEGData_test, 100, test_loader, model, criterion, opt, test_logger)
            
            with open(os.path.join(opt.result_path, 'test_set_bestval'+str(fold)+'.txt'), 'a') as f:
                    f.write('Prec1: ' + str(prec1)+ '; Loss: ' + str(test_loss))
