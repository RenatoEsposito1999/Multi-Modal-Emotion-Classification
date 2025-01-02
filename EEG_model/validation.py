import torch

labels_dict = {0:0,1:3,2:2,3:1}

def map_labels(labels):
    labels = labels.tolist()
    final_labels = []
    for label in labels:
        new_label = labels_dict[label]
        final_labels.append(new_label)
    return torch.tensor(final_labels,dtype=torch.long)

def valid(opt, dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    val_loss, precision_1_sum = 0, 0
    with torch.no_grad():
        for batch in dataloader:
            X = batch[0].to(opt.device)
            y = map_labels(batch[1]).to(opt.device)

            pred = model(X)
            val_loss += loss_fn(pred, y).item()
            #correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            precision_1 = torch.sum(torch.max(pred, 1).indices==y).item()
            precision_1 = precision_1 / opt.batch_size
            precision_1_sum += precision_1
    val_loss /= num_batches
    
    #correct /= size
    return precision_1_sum / num_batches,  val_loss