import os
import pandas as pd
import torch
import logging
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from global_dataset import DatasetArtificial
from datetime import datetime
from models import load_model
import yaml
import numpy as np
from tqdm import tqdm
import shutil
from torch import nn
from datetime import datetime, timedelta
from spacepy.coordinates import Coords
from spacepy.time import Ticktock

inferences_config_path = "config/inferences_config.yaml"
solar_indices_path = "/cluster/work/igp_psr/dslab_FS25_data_and_weights/"

with open(inferences_config_path, 'r') as file:
    inferences_config = yaml.load(file, Loader=yaml.FullLoader)

model_config_path = inferences_config["model_path"] + "trainig_config.yaml"

with open(model_config_path, 'r') as file:
    model_config = yaml.load(file, Loader=yaml.FullLoader)

logger = logging.getLogger(__name__)

timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
print("Started ", timestamp)

dslab_path = model_config["dslab_path"]
models_path = dslab_path + model_config["models_dir_name"] + "/"

batch_size = model_config["batch_size"]
num_workers = model_config["num_workers"]
pytables = model_config["pytables"]

if not "use_spheric_coords" in model_config.keys():
    model_config["use_spheric_coords"] = False
if not "normalize_features" in model_config.keys():
    model_config["normalize_features"] = False

device = torch.accelerator.current_accelerator().type if torch.cuda.is_available() else "cpu"
logger.info("Using %s device", device)
logger.info(f"batch_size: {batch_size}")

if inferences_config["same_day_as_training"]:
    doy = model_config["start_doy"]
    year = model_config["year"]
    n = model_config["num_days"]

else:
    doy = inferences_config["start_doy"]
    year = inferences_config["year"]
    n = inferences_config["num_days"]



model = load_model(inferences_config["model_path"])
logger.info("Model: %s", model)

model = model.to(device)
model.eval()

lons = []
lats = []
sods = []
for long in range(-180,180):
    for lat in range(-90,90):
        for sod in np.arange(0,24*60*60+1,60):
            lons.append(long)
            lats.append(lat)
            sods.append(sod)

def coord_transform(input_type, output_type, lats, lons, epochs):
    coords = np.array([[1 + 450 / 6371, lat, lon] for lat, lon in zip(lats, lons)], dtype=np.float64)
    geo_coords = Coords(coords, input_type, 'sph')
    geo_coords.ticks = Ticktock(epochs, 'UTC')
    return geo_coords.convert(output_type, 'sph')

date = datetime.strptime(str(year)+"-01-01", "%Y-%m-%d") + timedelta(days=doy - 1)
epochs = [date + timedelta(seconds=int(sod)) for sod in sods]
# Step 2: Transform to SM coordinates
sm_coords = coord_transform('GEO', 'SM', lats, lons, epochs)
out_coords = sm_coords.data


data_df = pd.DataFrame()

data_df["sm_lat_ipp"] = out_coords[:,1]
data_df["sm_lon_ipp"] = out_coords[:,2]
data_df["sod"] = sods
data_df["satele"] = 90 
data_df["satazi"] = 45

#shuffle_df = data_df.sample(frac=1,random_state=2001).reset_index(drop=True)
#print(shuffle_df.head(50))
#shuffle_df.head(50).to_csv("test.csv")

dataset_test = DatasetArtificial(data_df, doy=doy, year = year, pytables=pytables, solar_indices_path=solar_indices_path, optional_features=model_config['optional_features'], use_spheric_coords=model_config["use_spheric_coords"], normalize_features=model_config["normalize_features"])
dataloader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False, num_workers=num_workers)

num_batches = len(dataloader_test)
test_loss = 0

print("total",len(dataset_test))
pred_list = []
with torch.no_grad():
    for X in tqdm(dataloader_test):
        X= X.to(device)  
        pred = model(X)
        pred_list += pred.squeeze(1).tolist()

out_df = pd.DataFrame()
out_df["Latitude"] = lats
out_df["Longtidue"] = lons
out_df["SOD"] = sods
out_df["STEC"] = pred_list
out_df.to_csv("global_inference.csv",index=False)

