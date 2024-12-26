from datasets.generate_dataset_RAVDESS import get_test_set_RAVDESS
from utils.logger import Logger
from test import testing
from utils import transforms
import os
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns


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
    
    if not os.path.exists("Image"):
        os.makedirs("Image")
        
        
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
    test_loss, prec1, prec1_list, prec1_avarage_list, predicted_labels, all_true_labels = testing(best_state["epoch"], test_loader_audio_video, model, criterion_loss, opt, test_logger)
    #Save information into a file text
    with open(os.path.join(opt.result_path, 'test_set_best.txt'), 'a') as f:
            f.write('Prec1: ' + str(prec1)+ '; Loss: ' + str(test_loss))
            
    plt.figure(figsize=(8, 6))
    plt.plot(prec1_list, label='Test Precision', marker='o', linestyle='-')
    plt.xlabel('Batches')
    plt.ylabel('Precision')
    plt.title('Test Precision')
    plt.legend()
    plt.grid()
    plt.savefig('Image/test_precision.jpeg', format='jpeg') 
    plt.close()
    
    
    plt.figure(figsize=(8, 6))
    plt.plot(prec1_avarage_list, label='Test Precision Avarage', marker='o', linestyle='-')
    plt.xlabel('Batches')
    plt.ylabel('Precision')
    plt.title('Test Precision Avarages')
    plt.legend()
    plt.grid()
    plt.savefig('Image/test_precision_avarage.jpeg', format='jpeg') 
    plt.close()
    
    cm = confusion_matrix(all_true_labels, predicted_labels)
    # Usa Seaborn per plottare la confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Neutral', 'Happy', 'Angry', 'Sad'], yticklabels=['Neutral', 'Happy', 'Angry', 'Sad'])
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.savefig('Image/Confusion_matrix.jpeg', format='jpeg')
    plt.close()
    
    