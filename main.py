import h5py
import json
import os
import pandas as pd
import numpy as np
import torch
from torch import nn
import logging
from tqdm import tqdm
from matplotlib import pyplot as plt
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
logger = logging.getLogger(__name__)

datapath = "/cluster/work/igp_psr/arrueegg/GNSS_STEC_DB/2024/322/ccl_2024322_30_5.h5"

def get_data(datapath: str) -> pd.DataFrame:
    file = h5py.File(datapath, 'r')
    year = list(file.keys())[0]
    day = list(file[year].keys())[0]
    data = file[year][day]["all_data"]
    # TODO: figure out a better way than pandas to manipulate the data.
    data = pd.DataFrame(data[:])
    data["station"] = data["station"].str.decode("utf8")
    data["sat"] = data["sat"].str.decode("utf8")

    logger.info("Loaded data from file.")
    return data


def split_train_val_test(data: pd.DataFrame) -> pd.DataFrame:
    """ If file with splits alrady exists, load it. Otherwise generate the splits.
    I do it this way to ensure that the split is reproducible. Other options considered:
    - Sort the stations and randomly shuffle them using a fixed seed. After shuffling, keep first section
      for training, second for validation and third for testing. This would fail if depending on the day
      some stations are inactive.
    - Hash the stations by name into train, val and test buckets. My concern is that the hashing function
      may not be reproducible if, for instance, OS changes.
    """

    split_file_path = "/cluster/work/igp_psr/dslab_FS25/split.json"
    if os.path.exists(split_file_path):
        with open(split_file_path, 'r') as f:
            stations_map = json.load(f)
        logger.info("Loading train, val, test split...")
        
        
    else:
        logger.info("No existing split found...")
        logger.info("Splitting data into train, val, and test sets...")
        stations_df = data[["station", "lat_sta", "lon_sta"]].groupby("station").agg("first").sort_values(by="station").reset_index()
        stations = stations_df["station"].tolist()
        np.random.seed(10)  
        np.random.shuffle(stations)
        nstations = len(stations)
        stations_map = {station: (i / nstations > 0.6) + (i / nstations > 0.8) for i, station in enumerate(stations)}
    
        fig, ax = plt.subplots()
        stations_df["split"] = stations_df["station"].map(stations_map)
           
        ax.scatter(stations_df["lon_sta"], stations_df["lat_sta"], c=stations_df["split"] / 2, cmap="viridis")
        fig.savefig("outputs/split.png")
        
        with open(split_file_path, 'w') as f:
            json.dump(stations_map, f)
    
    
    data['split'] = data["station"].map(stations_map)   
    logger.info("Train, val, and test sets ready.") 
    
    return data

def extract_features(data: pd.DataFrame) -> pd.DataFrame:
    logger.info("Extracting sin/cos features...")
    data['sm_lon_ipp_s'] = np.sin(2 * np.pi * data['sm_lon_ipp'] / 360)
    data['sm_lon_ipp_c'] = np.cos(2 * np.pi * data['sm_lon_ipp'] / 360)
    data['sod_s'] = np.sin(2 * np.pi * data['sod'] / 86400) 
    data['sod_c'] = np.cos(2 * np.pi * data['sod'] / 86400) 
    data['satazi_s'] = np.sin(2 * np.pi * data['satazi'] / 360)
    data['satazi_c'] = np.cos(2 * np.pi * data['satazi'] / 360)

    logger.info("Extracted sin/cos features.")
    return data
    
class FCN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(8, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)
        self.double()

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        return x
    
def train(dataloader, model, loss_fct, optimizer, device):
    size = len(dataloader.dataset)
    # Set the model to training mode - important for batch normalization and dropout layers
    model.train()
    running_loss = 0.0
    for batch, (X, y) in tqdm(enumerate(dataloader), total=len(dataloader)):
        # Compute prediction and loss
        X, y = X.to(device), y.to(device)
        
        optimizer.zero_grad()

        pred = model(X)
        loss = loss_fct(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if batch % 3000 == 2999:    
            logger.info(f'[{batch + 1:5d}/{len(dataloader):>5d}] loss: {running_loss / 2000:.3f}')
            running_loss = 0.0


def test(dataloader, model, loss_fct, device):
    # Set the model to evaluation mode - important for batch normalization and dropout layers
    model.eval()
    num_batches = len(dataloader)
    test_loss = 0

    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fct(pred, y).item()

    test_loss /= num_batches

    return test_loss





if __name__ == "__main__":
    if not os.path.exists("outputs"): 
        os.makedirs("outputs")
                    
    torch.manual_seed(10)
    logging.basicConfig(filename='outputs/FCN.log', level=logging.INFO, format='%(asctime)s | %(message)s', datefmt='%H:%M')

    device = torch.accelerator.current_accelerator().type if torch.cuda.is_available() else "cpu"
    logger.info("Using %s device", device)

    learning_rate = 1e-3
    batch_size = 64
    epochs = 10

    logger.info("Loading Data...")
    data = get_data(datapath)
    data = split_train_val_test(data)
    data = extract_features(data)
    features = ['sm_lat_ipp', 'sm_lon_ipp_s', 'sm_lon_ipp_c', 'sod_s', 'sod_c', 'satele','satazi_s', 'satazi_c']

    logger.info("Loading Data into tensors...")
    X = torch.tensor(data[features].values, device=device, dtype=torch.float64)
    y = torch.tensor(data['stec'].values, device=device, dtype=torch.float64).unsqueeze_(-1)

    logger.info("Creating Tensor Datasets...")
    dataset_train = TensorDataset(X[data['split'] == 0], y[data['split'] == 0])
    dataset_val = TensorDataset(X[data['split'] == 1], y[data['split'] == 1])
    dataset_test = TensorDataset(X[data['split'] == 2], y[data['split'] == 2])
    
    logger.info("Preparing DataLoaders...")
    dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    dataloader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=True)
    dataloader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False)


    logger.info("Setting up Model...")
    model = FCN().to(device)
    logger.info("Model: %s", model)

    loss = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    logger.info("Starting training...")
    best_val_loss = float('inf')
    for t in range(epochs):
        logger.info("-------------------------------\nEpoch %s\n-------------------------------", t+1)
        train(dataloader_train, model, loss, optimizer, device)
        val_loss = test(dataloader_val, model, loss, device)
        logger.info(f"Validation Loss: {val_loss:>7f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "model.pth")
        else:
            # validation loss is increasing, so we stop training
            logger.info("Validation loss increased. Stopping training.")
            break
        
    logger.info("Training completed.")


    logger.info("Starting evaluation...")
    eval_loss_fct = nn.MAELoss()
    test_loss = test(dataloader_test, model, eval_loss_fct, device)
    logger.info(f"Evaluation MAE Loss: {test_loss:>7f}")
    logger.info("Completed.")
    
