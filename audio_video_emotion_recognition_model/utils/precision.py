import torch

def calculate_precision(outputs, targets):
    precision_1 = torch.sum(torch.max(outputs, 1).indices==targets)
    precision_1 = precision_1 / targets.size(0) #batchsize
    return precision_1