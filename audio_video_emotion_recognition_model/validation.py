'''
This code is based on https://github.com/okankop/Efficient-3DCNNs
'''
import torch
from torch.autograd import Variable
import time
from utils.average_meter import AverageMeter
from utils.precision import calculate_precision


def val_epoch_multimodal(epoch, data_loader, model, criterion, opt, logger, dist=None):
    #for evaluation with single modality, specify which modality to keep and which distortion to apply for the other modaltiy:
    #'noise', 'addnoise' or 'zeros'. for paper procedure, with 'softhard' mask use 'zeros' for evaluation, with 'noise' use 'noise'
    print('validation at epoch {}'.format(epoch))   
    model.eval()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses_avarage = AverageMeter()
    prec1_avarage = AverageMeter()

    end_time = time.time()
    
    
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

        prec1 = calculate_precision(logits_output.data, targets.data)
   
        losses_avarage.update(total_loss.data, opt.batch_size)
        
        prec1_avarage.update(prec1, opt.batch_size)
        
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
    
    

    return losses_avarage.avg.item(), prec1_avarage.avg.item()

    
