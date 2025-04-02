import h5py
import pandas as pd
import numpy as np
import torch
from torch import nn
import logging
from tqdm import tqdm
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
logger = logging.getLogger(__name__)

# datapath = "/cluster/work/igp_psr/arrueegg/GNSS_STEC_DB/2024/322/ccl_2024322_30_5.h5"
datapath = "V:/courses/dslab/team16/data/2023/020/ccl_2023020_30_5.h5"

def get_data(datapath: str) -> pd.DataFrame:
    file = h5py.File(datapath, 'r')
    year = list(file.keys())[0]
    day = list(file[year].keys())[0]
    data = file[year][day]["all_data"]
    # TODO: figure out a better way than pandas to manipulate the data.
    data = pd.DataFrame(data[:])
    # print("Loaded data from file.")
    logger.info("Loaded data from file.")
    return data


def split_train_val_test(data: pd.DataFrame) -> pd.DataFrame:
    # print("Splitting data into train, val, and test sets...")
    logger.info("Splitting data into train, val, and test sets...")
    stations = data["station"].unique()
    np.random.seed(10)  
    np.random.shuffle(stations)
    nstations = len(stations)
    stations_map = {station: (i / nstations > 0.6) + (i / nstations > 0.8) for i, station in enumerate(stations)}
    
    data['split'] = data["station"].map(stations_map)   
    # print("Train, val, and test sets created.")
    logger.info("Train, val, and test sets created.")
    return data

def extract_features(data: pd.DataFrame) -> pd.DataFrame:
    # print("Extracting sin/cos features...")
    logger.info("Extracting sin/cos features...")
    data['sm_lon_ipp_s'] = np.sin(2 * np.pi * data['sm_lon_ipp'] / 360)
    data['sm_lon_ipp_c'] = np.cos(2 * np.pi * data['sm_lon_ipp'] / 360)
    data['sod_s'] = np.sin(2 * np.pi * data['sod'] / 86400) 
    data['sod_c'] = np.cos(2 * np.pi * data['sod'] / 86400) 
    data['satazi_s'] = np.sin(2 * np.pi * data['satazi'] / 360)
    data['satazi_c'] = np.cos(2 * np.pi * data['satazi'] / 360)
    # print("Extracted sin/cos features.")
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
        

        # if batch % 3000 == 0:
        #     loss, current = loss.item(), batch * batch_size + len(X)
        #     # print(f"Training Loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
        #     logger.info(f"Training Loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
        #     # logger.info("Training Loss: %(L)s   [%(C)s/%(S)s]", {'L': loss, 'C': current, 'S': size})

        # # print statistics
        running_loss += loss.item()
        if batch % 6000 == 5999:    
            # print(f'[{batch + 1:5d}/{len(dataloader):>5d}] loss: {running_loss / 2000:.3f}')
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
    # # print(f"Test Error: \n Avg loss: {test_loss:>8f} \n")
    # logger.info(f"Test Error: \n Avg loss: {test_loss:>8f} \n")
    # # logger.info("Test Error: \n Avg loss: %(L) \n", {'L': test_loss})
    return test_loss





if __name__ == "__main__":
    torch.manual_seed(10)
    logging.basicConfig(filename='FCN.log', level=logging.INFO, format='%(asctime)s | %(message)s', datefmt='%H:%M')
    logger.info("-------------------------------------------------------\nStarting Script\n-------------------------------------------------------")

    # device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu" #only work in pytorch 2.6
    device = torch.accelerator.current_accelerator().type if torch.cuda.is_available() else "cpu"
    # print(f"Using {device} device")
    logger.info("Using %s device", device)

    learning_rate = 1e-3
    batch_size = 64
    epochs = 10

    # print("Loading Data...")
    logger.info("Loading Data...")
    data = get_data(datapath)
    data = split_train_val_test(data)
    data = extract_features(data)
    features = ['sm_lat_ipp', 'sm_lon_ipp_s', 'sm_lon_ipp_c', 'sod_s', 'sod_c', 'satele','satazi_s', 'satazi_c']

    # print("Loading Data into tensors...")
    logger.info("Loading Data into tensors...")
    X = torch.tensor(data[features].values, device=device)
    y = torch.tensor(data['stec'].values, device=device, dtype=torch.float64).unsqueeze_(-1)

    # print("Creating Tensor Datasets...")
    logger.info("Creating Tensor Datasets...")
    dataset_train = TensorDataset(X[data['split'] == 0], y[data['split'] == 0])
    dataset_val = TensorDataset(X[data['split'] == 1], y[data['split'] == 1])
    dataset_test = TensorDataset(X[data['split'] == 2], y[data['split'] == 2])
    
    # print("Preparing DataLoaders...")
    logger.info("Preparing DataLoaders...")
    dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    dataloader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=True)
    dataloader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False)


    
    # print("Setting up Model...")
    logger.info("Setting up Model...")
    model = FCN().to(device)
    # print(model)
    logger.info("Model: %s", model)

    loss = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # print("Starting training...")
    logger.info("Starting training...")
    best_val_loss = float('inf')
    for t in range(epochs):
        # print(f"-------------------------------\nEpoch {t+1}\n-------------------------------")
        logger.info("-------------------------------\nEpoch %s\n-------------------------------", t+1)
        train(dataloader_train, model, loss, optimizer, device)
        val_loss = test(dataloader_val, model, loss, device)
        # print(f"Validation Loss: {val_loss:>7f}")
        logger.info(f"Validation Loss: {val_loss:>7f}")
        # logger.info("Validation Loss: %s \n", val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # torch.save(model.state_dict(), "model.pth")
        else:
            # validation loss is increasing, so we stop training
            # print("Validation loss increased. Stopping training.")
            logger.info("Validation loss increased. Stopping training.")
            break
        
    # print("Training completed.")
    logger.info("Training completed.")


    # print("Starting evaluation...")
    logger.info("Starting evaluation...")
    eval_loss_fct = nn.L1loss()
    test_loss = test(dataloader_test, model, eval_loss_fct, device)
    # print(f"Evaluation MAE Loss: {test_loss:>7f}")
    logger.info(f"Evaluation MAE Loss: {test_loss:>7f}")
    
    # print("Completed.")
    logger.info("Completed.")
    
    
    # TODO: Define a fully connected network model
    # TODO: Add some training code
    # TODO: Add some eval code