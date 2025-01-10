import sys

sys.path.append('../Shared')


import torch
from plot_data import *
labels_dict = {0:0,1:3,2:2,3:1}

def map_labels(labels):
    labels = labels.tolist()
    final_labels = []
    for label in labels:
        new_label = labels_dict[label]
        final_labels.append(new_label)
    return torch.tensor(final_labels,dtype=torch.long)

def test(opt, model, dataloader, loss_fn):
    best_state = torch.load('results/best_state.pth')
    model.load_state_dict(best_state['state_dict'])
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    precision_list = []
    test_loss_list = []
    all_ground_truth = []
    all_predicted_labels = []
    #val_loss, precision_1_sum = 0, 0
    with torch.no_grad():
        for batch in dataloader:
            X = batch[0].to(opt.device)
            y = map_labels(batch[1]).to(opt.device)
            pred = model(X)
            test_loss_list.append(loss_fn(pred, y).item())
            precision_1 = torch.sum(torch.max(pred, 1).indices==y).item()
            precision_1 = precision_1 / opt.batch_size
            precision_list.append(precision_1)
            all_predicted_labels.extend((torch.max(pred, 1).indices).cpu().numpy()) 
            all_ground_truth.extend(y.cpu().numpy())
    plot_data(precision_list,"Images/test_accuracy.png","Test Accuracy", "Accuracy", "Batch")
    plot_data(test_loss_list,"Images/test_loss.png", "Test Loss", "Loss", "Batch")
    compute_confusion_matrix(all_ground_truth,all_predicted_labels,'Images/Confusion_matrix.jpeg')

