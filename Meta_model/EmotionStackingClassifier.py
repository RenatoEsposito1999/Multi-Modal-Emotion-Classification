import sys
sys.path.append('../EEG_model')
sys.path.append('../audio_video_emotion_recognition_model')

from tqdm import tqdm
import torch
import numpy as np
from torch import nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from tqdm import tqdm


labels_dict = {0:0,1:3,2:2,3:1}
def map_labels(labels):
    labels = labels.tolist()
    final_labels = []
    for label in labels:
        new_label = labels_dict[label]
        final_labels.append(new_label)
    return torch.tensor(final_labels,dtype=torch.long)

class EmotionStackingClassifier(nn.Module):
    def __init__(self, model1, model2, batch_size, max_iter=1000):
        super(EmotionStackingClassifier,self).__init__()

        self.model1 = model1
        self.model2 = model2
        self.meta_model = LogisticRegression(
            max_iter=max_iter,
            solver='lbfgs',
            class_weight='balanced'
        )
        self.batch_size = batch_size

    def organize_by_labels(self, dataloader):
        all_data = []
        all_labels = []
        
        # Collect all data and labels first
        for data, labels in dataloader:
            all_data.extend(data)
            all_labels.extend(labels)
            
        # Organize by labels
        print("Organizing labels...")
        label_data = {label: [] for label in range(4)}
        for data, label in tqdm(zip(all_data, all_labels),desc="Sorting dataset"):
            label = labels_dict[label.item()]
            label_data[label].append(data)
            
        return label_data

    def synchronize_datasets(self, dataloader1, dataloader2):
        label_data2 = self.organize_by_labels(dataloader2)
        batch_size = self.batch_size
        synced_batches = []
        current_batch = {
            'audio': [], 
            'video': [], 
            'eeg': [], 
            'labels': []
        }

        print("Starting synchronization")
        for audio, clip, labels in tqdm(dataloader1, desc="Synchronizing Data"):
            for i, label in enumerate(labels):
                label = label.item()
                if label_data2[label]:  # Check if matching data exists
                    eeg_data = label_data2[label].pop(0)
                    
                    # Add to current batch
                    current_batch['audio'].append(audio[i])
                    current_batch['video'].append(clip[i])
                    current_batch['eeg'].append(eeg_data)
                    current_batch['labels'].append(label)
                    
                    # When batch is full, add to synced_batches
                    if len(current_batch['labels']) == batch_size:
                        # Convert lists to tensors
                        batch_tensors = {
                            'audio': torch.stack(current_batch['audio']),
                            'video': torch.stack(current_batch['video']),
                            'eeg': torch.stack(current_batch['eeg']),
                            'labels': torch.tensor(current_batch['labels'])
                        }
                        synced_batches.append(batch_tensors)
                        
                        # Reset current batch
                        current_batch = {
                            'audio': [], 
                            'video': [], 
                            'eeg': [], 
                            'labels': []
                        }
                else:
                    print("empty list")
        
        # Handle any remaining samples in the last batch
        if current_batch['labels']:
            batch_tensors = {
                'audio': torch.stack(current_batch['audio']),
                'video': torch.stack(current_batch['video']),
                'eeg': torch.stack(current_batch['eeg']),
                'labels': torch.tensor(current_batch['labels'])
            }
            synced_batches.append(batch_tensors)

        return synced_batches

    def prepare_meta_features(self, synced_batches):
        all_meta_features = []
        all_labels = []
        device = next(self.model1.parameters()).device

        print("Preparing meta-features...")
        for batch in tqdm(synced_batches, desc="Processing batches"):
            # Move batch data to device
            audio = batch['audio'].to(device)
            video = batch['video'].to(device)
            eeg = batch['eeg'].to(device)
            
            # Reshape video: [B, C, T, H, W] -> [B*T, C, H, W]
            B, C, T, H, W = video.shape
            video = video.transpose(1, 2).reshape(-1, C, H, W)
            
            # Get predictions for both models
            with torch.no_grad():
                preds1 = self.model1(audio, video).cpu().numpy()
                preds2 = self.model2(eeg).cpu().numpy()
            
            # Combine predictions as meta-features
            meta_features = np.hstack((preds1, preds2))
            all_meta_features.append(meta_features)
            all_labels.extend(batch['labels'].numpy())

        # Combine all batches
        meta_features = np.vstack(all_meta_features)
        labels = np.array(all_labels)

        return meta_features, labels

    def fit(self, dataloader1, dataloader2, epochs=100, patience=3):
        print("Synchronizing datasets...")
        synced_batches = self.synchronize_datasets(dataloader1, dataloader2)
        
        print("Preparing meta-features...")
        meta_features, targets = self.prepare_meta_features(synced_batches)

        best_val_acc = 0

        for epoch in range(epochs):
            print(f"Epoch {epoch + 1}/{epochs} - Training meta-model...")

            # Train the logistic regression model
            self.meta_model.fit(meta_features, targets)

            # Evaluate training accuracy
            train_preds = self.meta_model.predict(meta_features)
            train_acc = accuracy_score(targets, train_preds)
            print(f"Epoch {epoch + 1}: Train Accuracy = {train_acc:.4f}")

            # Early stopping
            if train_acc > best_val_acc:
                best_val_acc = train_acc


        print(f"Best Training Accuracy: {best_val_acc:.4f}")
        
    def test(self, dataloader1, dataloader2):
        print("Synchronizing datasets...")
        synced_batches = self.synchronize_datasets(dataloader1, dataloader2)
        
        print("Preparing meta-features...")
        meta_features, targets = self.prepare_meta_features(synced_batches)

        best_val_acc = 0
        
        final_predictions = (self.meta_model.predict(meta_features))
        prob_predictions = self.meta_model.predict_proba(meta_features)
        
        
        
        return prob_predictions, final_predictions, targets
        


    def forward(self, audio_data, video_data, eeg_data):
        device = next(self.model1.parameters()).device
        
        # Move data to device
        audio_data = audio_data.to(device)
        video_data = video_data.to(device)
        eeg_data = eeg_data.to(device)
        
        # Get predictions from both models
        with torch.no_grad():
            preds1 = self.model1(audio_data, video_data).cpu().numpy()
            preds2 = self.model2(eeg_data).cpu().numpy()
        
        # Combine predictions and use meta-model
        meta_features = np.hstack((preds1, preds2))
        final_predictions = self.meta_model.predict(meta_features)
        
        return final_predictions