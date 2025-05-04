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
from spacepy.coordinates import Coords
from spacepy.time import Ticktock
from datetime import datetime, timedelta
logger = logging.getLogger(__name__)

datapath = "/cluster/work/igp_psr/arrueegg/GNSS_STEC_DB/2024/322/ccl_2024322_30_5.h5"
checkpoint_path = "/cluster/work/igp_psr/dslab_FS25_data_and_weights/model_1_4.pth"
#datapath = "ccl_2024322_30_5.h5"

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

    split_file_path = "/cluster/work/igp_psr/dslab_FS25_data_and_weights/split.json"
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

def extract_features_vtec(data: pd.DataFrame) -> pd.DataFrame:
    logger.info("Extracting sin/cos features...")
    
    #data['sm_lat_ipp'] = data["sm_lat_sta"]
    #data['sm_lon_ipp'] = data["sm_lon_sta"]
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
    



def test(dataloader, model, device):
    # Set the model to evaluation mode - important for batch normalization and dropout layers
    model.eval()
    num_batches = len(dataloader)
    test_loss = 0

    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    
    lat_list = []
    lon_list = []
    sod_list = []
    pred_list = []
    with torch.no_grad():
        for X in dataloader:
            X = X[0].to(device)
            pred = model(X[:,:-2]).squeeze(1)
            
            lat_list += X[:,0].tolist()
            lon_list += X[:,-2].tolist()
            sod_list += X[:,-1].tolist()
            pred_list += pred.tolist()
    
    df = pd.DataFrame()
    df["sta_lat"] = lat_list
    df["sta_lon"] = lon_list
    df["sod"] = sod_list
    df["pred_stec"] = pred_list
    return df





#if __name__ == "__main__":
if not os.path.exists("outputs"): 
    os.makedirs("outputs")

torch.manual_seed(10)
logging.basicConfig(filename='outputs/FCN.log', level=logging.INFO, format='%(asctime)s | %(message)s', datefmt='%H:%M')
logger.info("-------------------------------------------------------\nStarting Script\n-------------------------------------------------------")

device = torch.accelerator.current_accelerator().type if torch.cuda.is_available() else "cpu"
logger.info("Using %s device", device)

learning_rate = 1e-3
batch_size = 64
epochs = 10

logger.info("Loading Data...")
og_data = get_data(datapath)


geo_lat = []
geo_lon = []
new_sod = []
for lat in range(-89,90):
    for lon in range(-179,180):
        for s in range(0,86370,60):
            geo_lat.append(lat)
            geo_lon.append(lon)
            new_sod.append(s)

def coord_transform(input_type, output_type, lats, lons, epochs):
    coords = np.array([[1 + 450 / 6371, lat, lon] for lat, lon in zip(lats, lons)], dtype=np.float64)
    geo_coords = Coords(coords, input_type, 'sph')
    geo_coords.ticks = Ticktock(epochs, 'UTC')
    return geo_coords.convert(output_type, 'sph')

date = datetime.strptime("2024-01-01", "%Y-%m-%d") + timedelta(days=82 - 1)
epochs = [date + timedelta(seconds=int(sod)) for sod in new_sod]
# Step 2: Transform to SM coordinates
sm_coords = coord_transform('GEO', 'SM', geo_lat, geo_lon, epochs)

out_coords = sm_coords.data



data = pd.DataFrame()
data["sm_lat_sta"] = out_coords[:,1]
data["sm_lon_sta"] = out_coords[:,2]
data["sm_lat_ipp"] = out_coords[:,1]
data["sm_lon_ipp"] = out_coords[:,2]
data["sod"] = new_sod  

data["satazi"] = np.mean(og_data["satazi"])
data["satele"] = np.mean(og_data["satele"])

data = extract_features_vtec(data)
features = ['sm_lat_ipp', 'sm_lon_ipp_s', 'sm_lon_ipp_c', 'sod_s', 'sod_c', 'satele','satazi_s', 'satazi_c',"sm_lon_ipp","sod"]

logger.info("Loading Data into tensors...")
X = torch.tensor(data[features].values, device=device, dtype=torch.float64)

logger.info("Creating Tensor Dataset...")
dataset = TensorDataset(X)

logger.info("Preparing DataLoaders...")
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

logger.info("Setting up Model...")
    
model = FCN().to(device)
model.load_state_dict(torch.load(checkpoint_path, weights_only=True))
model.eval()

logger.info("Model: %s", model)

logger.info("Running inference")
out_df = test(dataloader, model, device)

out_df["geo_lat"] = geo_lat
out_df["geo_lon"] = geo_lon
out_df.to_csv("outputs/vtec_grid_with_geo.csv",index=False)
logger.info("Done")