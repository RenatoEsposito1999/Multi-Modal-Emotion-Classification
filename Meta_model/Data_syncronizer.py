import numpy as np
from collections import defaultdict
import random
from sklearn.model_selection import train_test_split

class EmotionDatasetSynchronizer:
    def __init__(self, random_seed=42):
        self.random_seed = random_seed
        random.seed(random_seed)
        np.random.seed(random_seed)
        
    def organize_by_emotion(self, X, y):
        """
        Organizes samples by emotion label
        """
        emotion_dict = defaultdict(list)
        for idx, emotion in enumerate(y):
            emotion_dict[emotion].append(idx)
        return emotion_dict
    
    def synchronize_datasets(self, X1, y1, X2, y2):
        """
        Synchronizes two datasets by matching emotion labels
        
        Parameters:
        -----------
        X1: np.array
            Features from first dataset
        y1: np.array
            Labels from first dataset
        X2: np.array
            Features from second dataset
        y2: np.array
            Labels from second dataset
            
        Returns:
        --------
        X1_sync, X2_sync, y_sync: synchronized datasets and labels
        """
        # Organize samples by emotion for both datasets
        emotions_dict1 = self.organize_by_emotion(X1, y1)
        emotions_dict2 = self.organize_by_emotion(X2, y2)
        
        # Find common emotions
        common_emotions = set(emotions_dict1.keys()) & set(emotions_dict2.keys())
        print(f"Common emotions found: {common_emotions}")
        
        # Initialize synchronized datasets
        X1_synced = []
        X2_synced = []
        y_synced = []
        
        # For each common emotion
        for emotion in common_emotions:
            indices1 = emotions_dict1[emotion]
            indices2 = emotions_dict2[emotion]
            
            # Get number of samples for this emotion
            n_samples = min(len(indices1), len(indices2))
            
            # Randomly select samples from both datasets
            selected_indices1 = random.sample(indices1, n_samples)
            selected_indices2 = random.sample(indices2, n_samples)
            
            # Add selected samples to synchronized datasets
            X1_synced.extend(X1[selected_indices1])
            X2_synced.extend(X2[selected_indices2])
            y_synced.extend([emotion] * n_samples)
            
        # Convert to numpy arrays
        X1_synced = np.array(X1_synced)
        X2_synced = np.array(X2_synced)
        y_synced = np.array(y_synced)
        
        return X1_synced, X2_synced, y_synced
    
    def synchronize_meta_features(self, pred1, y1, pred2, y2):
        """
        Synchronize meta-features based on predictions from both models
        """
        emotions_dict1 = self.organize_by_emotion(pred1, y1)
        emotions_dict2 = self.organize_by_emotion(pred2, y2)
        
        common_emotions = set(emotions_dict1.keys()) & set(emotions_dict2.keys())
        
        pred1_synced, pred2_synced, y_synced = [], [], []
        
        for emotion in common_emotions:
            indices1 = emotions_dict1[emotion]
            indices2 = emotions_dict2[emotion]
            
            n_samples = min(len(indices1), len(indices2))
            
            selected_indices1 = random.sample(indices1, n_samples)
            selected_indices2 = random.sample(indices2, n_samples)
            
            pred1_synced.extend(pred1[selected_indices1])
            pred2_synced.extend(pred2[selected_indices2])
            y_synced.extend([emotion] * n_samples)
            
        return np.array(pred1_synced), np.array(pred2_synced), np.array(y_synced)