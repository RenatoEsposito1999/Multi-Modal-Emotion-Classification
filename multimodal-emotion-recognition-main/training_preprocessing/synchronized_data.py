import torch


class Synchronized_data():
    def __init__(self, dataset):
        self.complete_dataset = self.split_dataset(dataset) #return list of datasets splitted by labels
        
        
    def split_dataset(self, dataset):
        """
        Splits a dataset into separate datasets based on labels.
    
        Args:
            dataset (Dataset): A dataset object with labeled data.
    
        Returns:
            tuple: Four datasets corresponding to labels 0, 1, 2, and 3.
        """
        # Initialize a dictionary to group indices by labels
        label_dict = {0: [], 1: [], 2: [], 3: []}
    
        # Populate the dictionary with indices corresponding to each label
        for index in range(len(dataset)):
            _, label = dataset[index]
            label_dict[label.item()].append(index)  # Use .item() to get the integer value from the tensor
    
        # Create separate datasets for each label
        dataset_0 = torch.utils.data.Subset(dataset, label_dict[0])
        dataset_1 = torch.utils.data.Subset(dataset, label_dict[1])
        dataset_2 = torch.utils.data.Subset(dataset, label_dict[2])
        dataset_3 = torch.utils.data.Subset(dataset, label_dict[3])
        
        complete_dataset = [dataset_0, dataset_1, dataset_2, dataset_3]
        
        return complete_dataset 
    
    def generate_artificial_batch(self, labels):
        batch = []
        for i in labels:
            selection_list = self.complete_dataset[i]
            random_index = torch.randint(0, len(selection_list), (1,)).item()  # Get a random index
            random_element = selection_list[random_index][0]# Access the random element
            batch.append(random_element)
        return batch