import sys

sys.path.append("../Shared")

from datasets.generate_dataset_RAVDESS import get_test_set_RAVDESS
from utils.logger import Logger
from test import testing
from utils import transforms
import os
import torch
from plot_data import plot_data, compute_confusion_matrix


'''
    This is a function to prepare all the data to perform the testing. 
    At the end writes a file in which can see some information
    
    Args:
        -opt: all the necessary arguments.
        -Model: the model in which load the weigths
        -criterion_loss: the loss choosen
        
    Return:
        none
'''

def testing_processing(opt, model, criterion_loss):  
    #Prepare the logger in which store the information
    test_logger = Logger(
        os.path.join(opt.result_path, 'test.log'), ['epoch', 'loss', 'prec1'])

    video_transform = transforms.Compose([
        transforms.ToTensor(opt.video_norm_value)])
    
    
    #Generate test set for audio video                
    test_data_audio_video = get_test_set_RAVDESS(opt, spatial_transform=video_transform) 
    
    
    #Prepare the test loader
    test_loader_audio_video = torch.utils.data.DataLoader(
        test_data_audio_video,
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=opt.n_threads,
        pin_memory=True)
    
    #load best state, there are two file pth for separating if the machine hase cuda or not
    if(opt.device=="cuda"):
        best_state = torch.load('%s/%s_best' % (opt.result_path, opt.store_name)+'.pth')
    else:
        best_state = torch.load('%s/%s_best' % (opt.result_path, opt.store_name)+'_cpu_.pth', map_location="cpu")
        
    #Load the weigths on the model
    model.load_state_dict(best_state['state_dict'])
        
    
    #Compute the testing
    test_loss, prec1, prec1_list, prec1_avarage_list,losses_avarage_list,  predicted_labels, all_true_labels = testing(best_state["epoch"], test_loader_audio_video, model, criterion_loss, opt, test_logger)
    #Save information into a file text
    with open(os.path.join(opt.result_path, 'test_set_best.txt'), 'a') as f:
            f.write('Prec1: ' + str(prec1)+ '; Loss: ' + str(test_loss))
            
    plot_data(prec1_avarage_list, "Images/test_accuracy.pdf", "Test Accuracy - Audio-Video Model", "accuracy", "batch", "accuracy")
    plot_data(losses_avarage_list, "Images/test_loss.pdf", "Test Loss - Audio-Video Model", "loss", "batch", "loss")
    
    compute_confusion_matrix(all_true_labels, predicted_labels, "Images/confusion_matrix.pdf", "Confusion Matrix - Audio-Video Model")
    
    