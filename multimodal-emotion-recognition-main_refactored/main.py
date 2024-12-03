# -*- coding: utf-8 -*-
"""
Created on Mon Oct 25 14:07:29 2021

@author: chumache
"""
import os
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pickle


from opts import parse_opts
from trainining_validation_processing import training_validation_processing
from testing_processing import testing_processing
from Multimodal_transformer.MultimodalTransformer import MultimodalTransformer
from Data_preprocessing import eeg_preprocessing
from datasets.eeg_dataset import EEGDataset
from predict import predict


def pad_and_mask(sequence, max_length):
    """
    Pad a single sequence to the given max_length and create a mask.
    """
    length = sequence.shape[0]
    padding = max_length - length
    padded_sequence = np.pad(sequence, ((0, padding), (0, 0)), mode='constant', constant_values=0)
    mask = [1] * length + [0] * padding  # 1 for real data, 0 for padding
    return padded_sequence, mask

if __name__ == '__main__':
    
    opt = parse_opts()
 
   
    if opt.device != 'cpu':
        opt.device = 'cuda' if torch.cuda.is_available() else 'cpu'  

    if not os.path.exists(opt.result_path):
        os.makedirs(opt.result_path)
        
    opt.arch = '{}'.format(opt.model)  
    opt.store_name = '_'.join([opt.dataset, opt.model, str(opt.sample_duration)])
                      
  
    torch.manual_seed(opt.manual_seed)
    model = MultimodalTransformer(opt.n_classes, seq_length = opt.sample_duration, pretr_ef=opt.pretrain_path, num_heads=opt.num_heads)

    if opt.device != 'cpu':
        model = model.to(opt.device)
        model = nn.DataParallel(model, device_ids=None)
        pytorch_total_params = sum(p.numel() for p in model.parameters() if
                               p.requires_grad)
        print("Total number of trainable parameters: ", pytorch_total_params)
        
    
    #Define loss for training-validation-testing
    criterion_loss = nn.CrossEntropyLoss()
    criterion_loss = criterion_loss.to(opt.device)
    
    #In this function apply the preprocess for eeg data, in particular create the three files .npz into the folder EEG_data
    #eeg_preprocessing.preprocess(opt.eeg_dataset_path, opt)
    EEGDataset_complete = EEGDataset(opt.eeg_dataset_path, 14)
    
    train_split_eeg, validation_split_eeg, test_split_eeg = torch.utils.data.random_split(EEGDataset_complete, [756, 216, 108])
    
    found_files = [file for file in ["EEGTra.pkl", "EEGVal.pkl", "EEGTest.pkl"] if os.path.exists(os.path.join("./", file))]
    
    if(len(found_files)!=3):
    
        # Define custom collate function for padding and masking
        def collate_fn(batch):
            sequences, labels = zip(*batch)
            max_sequence_length=0
            for seq in sequences:
                max_sequence_length = max(seq.shape[0], max_sequence_length)
            padded_sequences, masks = zip(*[pad_and_mask(seq, max_sequence_length) for seq in sequences])
            
            # Convert to tensors
            padded_sequences = torch.tensor(padded_sequences, dtype=torch.float32)
            masks = torch.tensor(masks, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.long)
            
            return padded_sequences, labels, masks
        
        
        
        dataloader_training_eeg = DataLoader(train_split_eeg, batch_size=opt.batch_size, shuffle=True, collate_fn=collate_fn)
        dataloader_val_eeg = DataLoader(validation_split_eeg, batch_size=opt.batch_size, shuffle=True, collate_fn=collate_fn)
        dataloader_test_eeg = DataLoader(test_split_eeg, batch_size=opt.batch_size, shuffle=True, collate_fn=collate_fn)
        
       
        
        with open('EEGTrain.pkl', 'wb') as f:
            pickle.dump(dataloader_training_eeg, f)
        with open('EEGVal.pkl', 'wb') as f:
            pickle.dump(dataloader_val_eeg, f)
        with open('EEGTest.pkl', 'wb') as f:
            pickle.dump(dataloader_test_eeg, f)
    else:
        print("Ci sono!")
        with open('EEGTrain.pkl', 'rb') as f:
            dataloader_training_eeg = pickle.load(f)
        with open('EEGVal.pkl', 'rb') as f:
            dataloader_val_eeg = pickle.load(f)
        with open('EEGTest.pkl', 'rb') as f:
            dataloader_test_eeg = pickle.load(f)
    
    #Training-Validation Phase
    if not opt.no_train or not opt.no_val:
        training_validation_processing(opt, model ,criterion_loss, dataloader_training_eeg, dataloader_val_eeg)

    # Testing Phase       
    if opt.test:
        testing_processing(opt, model, criterion_loss)
        
    if opt.predict:
        predict(opt, model)
        

            
