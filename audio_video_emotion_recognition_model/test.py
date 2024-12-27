'''
This code is based on https://github.com/okankop/Efficient-3DCNNs
'''
import torch
from torch.autograd import Variable
import time
from utils.average_meter import AverageMeter
from utils.precision import calculate_precision
import numpy as np

def testing(epoch, data_loader, model, criterion, opt, logger, dist=None):
    print('Testing at epoch {}'.format(epoch))   
    model.eval()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses_avarage = AverageMeter()
    prec1_avarage = AverageMeter()

    end_time = time.time()
    prec1_list = []
    prec1_avarage_list = []
    losses_avarage_list = []
    predicted_labels = []
    all_true_labels = []
    
    for i, item1 in enumerate(data_loader):
      
        data_time.update(time.time() - end_time)
        
        inputs_audio, inputs_visual, targets = item1
    
        inputs_visual = inputs_visual.permute(0,2,1,3,4)
        inputs_visual = inputs_visual.reshape(inputs_visual.shape[0]*inputs_visual.shape[1], inputs_visual.shape[2], inputs_visual.shape[3], inputs_visual.shape[4])
        
        targets = targets.to(opt.device)
        inputs_visual = inputs_visual.to(opt.device)
        inputs_audio = inputs_audio.to(opt.device)
    
        with torch.no_grad():
            logits_output  = model(inputs_audio, inputs_visual)
            
        total_loss = criterion(logits_output, targets)
        
        
        predicted_labels.extend((torch.max(logits_output, 1).indices).cpu().numpy()) 
        all_true_labels.extend(targets.cpu().numpy()) 
        prec1 = calculate_precision(logits_output.data, targets.data)
        prec1_list.append(prec1.item())
        losses_avarage.update(total_loss.data, opt.batch_size)
        losses_avarage_list.append(losses_avarage.avg.item())
        
        prec1_avarage.update(prec1, opt.batch_size)
        prec1_avarage_list.append(prec1_avarage.avg.item())
        batch_time.update(time.time() - end_time)
        end_time = time.time()
        
        print('Epoch: [{0}][{1}/{2}]\t'
              'Time {batch_time.val:.5f} ({batch_time.avg:.5f})\t'
              'Data {data_time.val:.5f} ({data_time.avg:.5f})\t'
              'Loss {loss}\t'
              'Prec@1 {prec1_avarage.val:.5f} ({prec1_avarage.avg:.5f})\t'.format(
                  epoch,
                  i + 1,
                  len(data_loader),
                  batch_time=batch_time,
                  data_time=data_time,
                  loss=total_loss,
                  prec1_avarage=prec1_avarage))

    logger.log({'epoch': epoch,
                'loss': losses_avarage.avg.item(),
                'prec1': prec1_avarage.avg.item()})
    
    

    return losses_avarage.avg.item(), prec1_avarage.avg.item(), prec1_list, prec1_avarage_list, losses_avarage_list, predicted_labels, all_true_labels

    
