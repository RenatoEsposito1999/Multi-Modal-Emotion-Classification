import sys

sys.path.append('../Shared')

from plot_data import *

import torch
from train import train
from validation import valid
from utils.logger import logger
import pickle

def training_validation(opt, model, train_loader, val_loader, loss):
    optimizer = torch.optim.SGD(
            model.parameters(),
            lr=0.04,
            momentum=0.9,
            dampening=0.9,
            weight_decay=1e-3,
            nesterov=False)
    
    train_logger = logger("results/train_logger.csv",["N. epoch", "Train Loss", "Train Accuracy"])
    val_logger = logger("results/val_logger.csv",["N. epoch", "Validation Loss", "Validation Accuracy"])
    
    
    if opt.resume_path:
        #This is used to resume the trained stopped, so import all the necessary
        print('loading checkpoint {}'.format(opt.resume_path))
        checkpoint = torch.load(opt.resume_path)
        best_val_loss = checkpoint['best_val_loss']
        best_val_acc = checkpoint["best_val_acc"]
        opt.begin_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint["optimizer"])
    
    best_val_loss = float("inf")
    best_val_acc = 0.0  # Initialize with a default value
    final_test_acc = 0.0
    best_epoch = 0
    train_loss_list = []
    train_acc_list = []
    val_loss_list = []
    val_acc_list = []
    train_times = []
    
    
    for i in range (opt.begin_epoch, opt.n_epochs + 1):
        if not opt.no_train:
            train_loss, train_acc = train(opt,train_loader, model, loss, optimizer)
            train_loss_list.append(train_loss)
            train_acc_list.append(train_acc)
            train_logger.add_row([i,train_loss, train_acc])
        
        if not opt.no_val:
            val_acc, val_loss = valid(opt, val_loader, model, loss)
            val_acc_list.append(val_acc)
            val_loss_list.append(val_loss)
            val_logger.add_row([i, val_loss, val_acc])
            best_current_epoch = False
            if best_val_loss > val_loss:
                best_val_loss = val_loss
                best_val_acc = val_acc
                best_epoch = i
                best_current_epoch = True
                
        with open("results/train_loss_acc_val_loss_acc.pkl", "wb") as file: 
            pickle.dump((train_loss_list, train_acc_list, val_loss_list, val_acc_list), file)
            
        state = {
            "epoch": i,
            "optimizer": optimizer.state_dict(),
            "best_val_loss": best_val_loss,
            "best_val_acc": best_val_acc,
            "state_dict": model.state_dict(),
            "seed": opt.manual_seed  
        }
        
        torch.save(state, f"results/checkpoint_state.pth")
        
        if best_current_epoch == True:
            torch.save(state, f"results/best_state.pth")
            best_current_epoch = False
            
        
        if (i + 1) % 10 == 0:
            log_format = (
                "Epoch {}: loss={:.4f}, train_acc={:.4f}, val_acc={:.4f}"
            )
            print(log_format.format(i + 1, train_loss, train_acc, val_acc))
            
            
    print("Best Epoch {}".format(best_epoch))
    print("Done!")
    
    
    plot_data(train_loss_list,"Images/training_loss.png","Training Loss", "Loss", "Epoch")
    plot_data(train_acc_list,"Images/training_acc.png","Training Accuracy", "Acc", "Epoch")
    plot_data(val_loss_list,"Images/validation_loss.png","Validation Loss", "Loss", "Epoch")
    plot_data(val_acc_list,"Images/validation_acc.png","Validation Accuracy", "Acc", "Epoch")
            