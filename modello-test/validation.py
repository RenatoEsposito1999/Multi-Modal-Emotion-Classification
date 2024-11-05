'''
This code is based on https://github.com/okankop/Efficient-3DCNNs
'''
import torch
from torch.autograd import Variable
import time
from utils import AverageMeter, calculate_accuracy
'''
def val_epoch_multimodal(epoch, data_loader, model, criterion, opt, logger,modality='both',dist=None ):
    #for evaluation with single modality, specify which modality to keep and which distortion to apply for the other modaltiy:
    #'noise', 'addnoise' or 'zeros'. for paper procedure, with 'softhard' mask use 'zeros' for evaluation, with 'noise' use 'noise'
    print('validation at epoch {}'.format(epoch))
    assert modality in ['both', 'audio', 'video']    
    model.eval()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    end_time = time.time()
   
    for i, (inputs_audio, inputs_visual, targets) in enumerate(data_loader):
        #print("audio:", inputs_audio.shape) #audio: torch.Size([8, 10, 156])                                       
        #print("video: ", inputs_visual.shape) #video:  torch.Size([8, 3, 15, 224, 224])
        data_time.update(time.time() - end_time)

        if modality == 'audio':
            print('Skipping video modality')
            if dist == 'noise':
                print('Evaluating with full noise')
                inputs_visual = torch.randn(inputs_visual.size())
            elif dist == 'addnoise': #opt.mask == -4:
                print('Evaluating with noise')
                inputs_visual = inputs_visual + (torch.mean(inputs_visual) + torch.std(inputs_visual)*torch.randn(inputs_visual.size()))
            elif dist == 'zeros':
                print("entro qua:")
                inputs_visual = torch.zeros(inputs_visual.size())
            else:
                print('UNKNOWN DIST!')
        elif modality == 'video':
            print('Skipping audio modality')
            if dist == 'noise':
                print('Evaluating with noise')
                inputs_audio = torch.randn(inputs_audio.size())
            elif dist == 'addnoise': #opt.mask == -4:
                print('Evaluating with added noise')
                inputs_audio = inputs_audio + (torch.mean(inputs_audio) + torch.std(inputs_audio)*torch.randn(inputs_audio.size()))

            elif dist == 'zeros':
                inputs_audio = torch.zeros(inputs_audio.size())
        inputs_visual = inputs_visual.permute(0,2,1,3,4)
        inputs_visual = inputs_visual.reshape(inputs_visual.shape[0]*inputs_visual.shape[1], inputs_visual.shape[2], inputs_visual.shape[3], inputs_visual.shape[4])
        


        
        targets = targets.to(opt.device)
        with torch.no_grad():
            inputs_visual = Variable(inputs_visual)
            inputs_audio = Variable(inputs_audio)
            targets = Variable(targets)
        print(targets)
      
        outputs = model(inputs_audio, inputs_visual)
        loss = criterion(outputs, targets)
        prec1, prec5 = calculate_accuracy(outputs.data, targets.data, topk=(1,5))
        top1.update(prec1, inputs_audio.size(0))
        top5.update(prec5, inputs_audio.size(0))

        losses.update(loss.data, inputs_audio.size(0))

        batch_time.update(time.time() - end_time)
        end_time = time.time()

        print('Epoch: [{0}][{1}/{2}]\t'
              'Time {batch_time.val:.5f} ({batch_time.avg:.5f})\t'
              'Data {data_time.val:.5f} ({data_time.avg:.5f})\t'
              'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
              'Prec@1 {top1.val:.5f} ({top1.avg:.5f})\t'
              'Prec@5 {top5.val:.5f} ({top5.avg:.5f})'.format(
                  epoch,
                  i + 1,
                  len(data_loader),
                  batch_time=batch_time,
                  data_time=data_time,
                  loss=losses,
                  top1=top1,
                  top5=top5))

    logger.log({'epoch': epoch,
                'loss': losses.avg.item(),
                'prec1': top1.avg.item(),
                'prec5': top5.avg.item()})

    return losses.avg.item(), top1.avg.item()

def val_epoch(epoch, data_loader, model, criterion, opt, logger, modality='both', dist=None):
    print('validation at epoch {}'.format(epoch))
    if opt.model == 'multimodalcnn':
        return val_epoch_multimodal(epoch, data_loader, model, criterion, opt, logger, modality, dist=dist)
    
'''
import torch
from sklearn.metrics import confusion_matrix
import numpy as np
 
def val_epoch_multimodal(epoch, data_loader, model, criterion, opt, logger, modality='both', dist=None):
    print('validation at epoch {}'.format(epoch))
    assert modality in ['both', 'audio', 'video']    
    model.eval()
 
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
 
    end_time = time.time()
    # Collect all predictions and true labels for the confusion matrix
    all_preds = []
    all_labels = []
 
    for i, (inputs_audio, inputs_visual, targets) in enumerate(data_loader):
        data_time.update(time.time() - end_time)
 
        # Handle modality-specific adjustments
        if modality == 'audio':
            inputs_visual = torch.randn(inputs_visual.size()) if dist == 'noise' else torch.zeros(inputs_visual.size())
        elif modality == 'video':
            inputs_audio = torch.randn(inputs_audio.size()) if dist == 'noise' else torch.zeros(inputs_audio.size())
 
        inputs_visual = inputs_visual.permute(0, 2, 1, 3, 4)
        inputs_visual = inputs_visual.reshape(inputs_visual.shape[0] * inputs_visual.shape[1],
                                              inputs_visual.shape[2], inputs_visual.shape[3], inputs_visual.shape[4])
 
        targets = targets.to(opt.device)
        with torch.no_grad():
            inputs_visual = Variable(inputs_visual).to(opt.device)
            inputs_audio = Variable(inputs_audio).to(opt.device)
            targets = Variable(targets).to(opt.device)
 
        outputs = model(inputs_audio, inputs_visual)
        loss = criterion(outputs, targets)
        prec1, prec5 = calculate_accuracy(outputs.data, targets.data, topk=(1,5))
        top1.update(prec1, inputs_audio.size(0))
        top5.update(prec5, inputs_audio.size(0))
        losses.update(loss.data, inputs_audio.size(0))
 
        # Store predictions and true labels for confusion matrix
        _, preds = outputs.max(1)  # Get the predicted class
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(targets.cpu().numpy())
 
        batch_time.update(time.time() - end_time)
        end_time = time.time()
 
        print('Epoch: [{0}][{1}/{2}]\t'
              'Time {batch_time.val:.5f} ({batch_time.avg:.5f})\t'
              'Data {data_time.val:.5f} ({data_time.avg:.5f})\t'
              'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
              'Prec@1 {top1.val:.5f} ({top1.avg:.5f})\t'
              'Prec@5 {top5.val:.5f} ({top5.avg:.5f})'.format(
                  epoch, i + 1, len(data_loader),
                  batch_time=batch_time,
                  data_time=data_time,
                  loss=losses,
                  top1=top1,
                  top5=top5))
 
    # Log results
    logger.log({'epoch': epoch,
                'loss': losses.avg.item(),
                'prec1': top1.avg.item(),
                'prec5': top5.avg.item()})
 
    # Return loss, accuracy, predictions, and labels for confusion matrix
    return losses.avg.item(), top1.avg.item(), np.array(all_preds), np.array(all_labels)
 
def val_epoch(epoch, data_loader, model, criterion, opt, logger, modality='both', dist=None):
    print('validation at epoch {}'.format(epoch))
    if opt.model == 'multimodalcnn':
        return val_epoch_multimodal(epoch, data_loader, model, criterion, opt, logger, modality, dist=dist)