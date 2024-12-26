'''
This code is based on https://github.com/okankop/Efficient-3DCNNs
'''
import torch
from torch.autograd import Variable
import time
from utils.average_meter import AverageMeter
from utils.precision import calculate_precision

'''
    This function perform the training for the i-th epoch.
    
    Args:
        -epoch: the i-th epoch
        -data_loader_audio_video: this is the data_loader for audio video
        -model: the model that want to train
        -criterion_loss: the loss
        -optimizer: the algortithm choosen to udate the weigths
        -opt: all the arguments
        -epoch_logger: save in file log the information about i-th epoch
        -batch_logger: save in file log the information about the batch    
    Returns:
        None
'''



def train_epoch_multimodal(epoch, data_loader_audio_video, model, criterion_loss, optimizer, opt,
                epoch_logger, batch_logger):
    print('train at epoch {}'.format(epoch))

    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses_avarage = AverageMeter()
    prec1_avarage = AverageMeter()
 
        
    end_time = time.time()
 
    for i, item1 in enumerate(data_loader_audio_video):
        data_time.update(time.time() - end_time)
    
        audio_inputs, visual_inputs, targets = item1
        
        if opt.mask is not None:
            with torch.no_grad():   
                if opt.mask == 'softhard':
                    audio_inputs, visual_inputs, targets = apply_dropout(audio_inputs, visual_inputs, targets)

        visual_inputs = visual_inputs.permute(0,2,1,3,4)
        visual_inputs = visual_inputs.reshape(visual_inputs.shape[0]*visual_inputs.shape[1], visual_inputs.shape[2], visual_inputs.shape[3], visual_inputs.shape[4])
        
        targets = targets.to(opt.device)
        visual_inputs = visual_inputs.to(opt.device)
        audio_inputs = audio_inputs.to(opt.device)
        
        logits_output = model(audio_inputs, visual_inputs)
       
        total_loss = criterion_loss(logits_output, targets)
               
        prec1 = calculate_precision(logits_output.data, targets.data)
    
        losses_avarage.update(total_loss.data, opt.batch_size)
        prec1_avarage.update(prec1, opt.batch_size)
       
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        batch_time.update(time.time() - end_time)
        end_time = time.time()

        batch_logger.log({
            'epoch': epoch,
            'batch': i + 1,
            'iter': (epoch - 1) * len(data_loader_audio_video) + (i + 1),
            'loss': losses_avarage.val.item(),
            'prec1': prec1_avarage.val.item(),
            'lr': optimizer.param_groups[0]['lr']
        })
        if i % 10 ==0:
            print('Epoch: [{0}][{1}/{2}]\t lr: {lr:.5f}\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                    'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                    'Prec@1 {prec1_avarage.val:.5f} ({prec1_avarage.avg:.5f})\t'.format(
                        epoch,
                        i,
                        len(data_loader_audio_video),
                        batch_time=batch_time,
                        data_time=data_time,
                        loss=losses_avarage,
                        prec1_avarage=prec1_avarage,
                        lr=optimizer.param_groups[0]['lr']))

    epoch_logger.log({
        'epoch': epoch,
        'loss': losses_avarage.avg.item(),
        'prec1': prec1_avarage.avg.item(),
        'lr': optimizer.param_groups[0]['lr']
    })
    
    
def apply_dropout(audio_inputs, visual_inputs, targets):
    coefficients = torch.randint(low=0, high=100,size=(audio_inputs.size(0),1,1))/100
    vision_coefficients = 1 - coefficients
    coefficients = coefficients.repeat(1,audio_inputs.size(1),audio_inputs.size(2))
    vision_coefficients = vision_coefficients.unsqueeze(-1).unsqueeze(-1).repeat(1,visual_inputs.size(1), visual_inputs.size(2), visual_inputs.size(3), visual_inputs.size(4))

    audio_inputs = torch.cat((audio_inputs, audio_inputs*coefficients, torch.zeros(audio_inputs.size()), audio_inputs), dim=0) 
    visual_inputs = torch.cat((visual_inputs, visual_inputs*vision_coefficients, visual_inputs, torch.zeros(visual_inputs.size())), dim=0)   
                    
    targets = torch.cat((targets, targets, targets, targets), dim=0)
    shuffle = torch.randperm(audio_inputs.size()[0])
    audio_inputs = audio_inputs[shuffle]
    visual_inputs = visual_inputs[shuffle]
    targets = targets[shuffle]
    
    return audio_inputs, visual_inputs, targets

 

    
    
