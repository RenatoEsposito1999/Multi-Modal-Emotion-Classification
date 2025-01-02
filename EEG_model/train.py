import torch


labels_dict = {0:0,1:3,2:2,3:1}

def map_labels(labels):
    labels = labels.tolist()
    final_labels = []
    for label in labels:
        new_label = labels_dict[label]
        final_labels.append(new_label)
    return torch.tensor(final_labels,dtype=torch.long)


def train(opt, dataloader, model, loss_fn, optimizer):
    num_batches = len(dataloader)
    total_loss = 0.0
    precision_1_sum = 0.0
    model.train()
    for batch_idx, batch in enumerate(dataloader):
        X = batch[0].to(opt.device)
        y = map_labels(batch[1]).to(opt.device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)
       
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        precision_1 = torch.sum(torch.max(pred, 1).indices==y).item()
        precision_1 = precision_1 / opt.batch_size
        precision_1_sum +=precision_1
        total_loss += loss.item()
        
    return total_loss / num_batches, precision_1_sum / num_batches