import torch
import random
from torcheeg.models import FBCCNN
from torcheeg.datasets import SEEDIVDataset
from torcheeg.datasets.constants import SEED_IV_CHANNEL_LIST, SEED_IV_CHANNEL_LOCATION_DICT
from torcheeg import transforms

# Define paths and configurations
cached_save = "/home/v.mele/cognitive_robotics/cognitive-robotics-project/EEG_model/.torcheeg/datasets_1735832345892_i0VpE"
path = "/home/v.mele/cognitive_robotics/datasets/SEED_IV/SEED_IV/eeg_raw_data"
epoc_plus_channels = ['AF3', 'AF4', 'F3', 'F4', 'F7', 'F8', 'T7', 'T8', 'P7', 'P8', 'O1', 'O2', 'FC5', 'FC6']
Filtered_channels = transforms.PickElectrode.to_index_list(epoc_plus_channels, SEED_IV_CHANNEL_LIST)
filtered_location_dict = {ch: SEED_IV_CHANNEL_LOCATION_DICT[ch] for ch in epoc_plus_channels}
device = "cuda" if torch.cuda.is_available() else "cpu"


# Original SEED_IV labels: neutral (0),sad(1),fear(2),happy(3)
# MM emotion rec labels: neutral(0),happy(1),angry(2),sad(3)
label_mapping = {
    0: 0,
    1: 3,
    2: 2,
    3: 1
}

def import_data_model():
    dataset = SEEDIVDataset(
        io_path=cached_save,
        root_path=path,
        offline_transform=transforms.Compose([
            transforms.PickElectrode(Filtered_channels),
            transforms.BandDifferentialEntropy(),
            transforms.ToGrid(filtered_location_dict)]),
        online_transform=transforms.Compose([
            transforms.ToTensor()]),
        label_transform=transforms.Compose([
            transforms.Select('emotion')
        ]),
        num_worker=16,
        io_size=167772160
    )

    # Configure FBCNN with matching dimensions
    model = FBCCNN(num_classes=4, in_channels=4, grid_size=(9, 9)).to(device)
    return model, dataset


def predict(model,test_split):

    batches = list(test_split)
    random_batch = random.choice(batches)

    data_batch, labels_batch = random_batch

    random_sample_idx = random.randint(0, data_batch.size(0) - 1)
    random_sample = data_batch[random_sample_idx]
    random_label = labels_batch[random_sample_idx]

    torch.cuda.empty_cache()
    torch.set_printoptions(precision=10)

    best_state = torch.load('best_state.pth')
    model.load_state_dict(best_state['state_dict'])

    # Ensure the model and random_sample are on the same device
    random_sample = random_sample.to(device)
    model = model.to(device)
    
    # Perform inference
    with torch.no_grad():
        logits = model(random_sample.unsqueeze(0))  # Add batch dimension if needed
        predicted_label = torch.argmax(logits, dim=1).item()  # Get the predicted label
    
    print(f"Predicted label: {predicted_label}")
    print(f"True label: {random_label.item()}")  # Convert label tensor to a scalar if needed