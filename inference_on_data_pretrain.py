import os
import pandas as pd
import torch
from torch import nn
import logging
from tqdm import tqdm
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from dataset.dataset import DatasetIndices, DatasetReorganized
from evaluation.test import test
from datetime import datetime
from models.models import get_model, load_pretrained_model
from training.training import train_single_epoch
import yaml
import shutil
import numpy as np
import glob

model_to_eval_path = "/cluster/work/igp_psr/dslab_FS25_data_and_weights/models/model_2025-05-01-14-35-02/"
model_config_path = model_to_eval_path + "trainig_config.yaml"
with open(model_config_path, 'r') as file:
    model_config = yaml.load(file, Loader=yaml.FullLoader)
model_config

pretrain_model_config_path = model_config["pretrained_model_path"] + "trainig_config.yaml"
with open(pretrain_model_config_path, 'r') as file:
    pretrain_model_config = yaml.load(file, Loader=yaml.FullLoader)

logger = logging.getLogger(__name__)


timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
print("Started ", timestamp)


dslab_path = model_config["dslab_path"]
models_path = dslab_path + model_config["models_dir_name"] + "/"

if not os.path.exists(models_path): 
    os.makedirs(models_path)

training_id = "model_" + timestamp

model_path = models_path + training_id + "/"
assert not os.path.exists(model_path), f"Training ID {training_id} already exists"
        

batch_size = model_config["batch_size"]
num_workers = model_config["num_workers"]
use_reorganized_data = model_config["use_reorganized_data"]
pytables = model_config["pytables"]


torch.manual_seed(10)
logger.info("-------------------------------------------------------\nStarting Script\n-------------------------------------------------------")

device = torch.accelerator.current_accelerator().type if torch.cuda.is_available() else "cpu"
logger.info("Using %s device", device)
logger.info(f"batch_size: {batch_size}")

datapath = model_config["datapath"]
doy = model_config["start_doy"]
year = model_config["year"]
n = model_config["num_days"]
logger.info(f"date range: {doy} until {doy+n-1} of {year}")
assert doy + n <= 366, "Date range reaches end of year. Currently not supported."
datapaths = [datapath + f"{year}/{str(doy+i).zfill(3)}/ccl_{year}{str(doy+i).zfill(3)}_30_5.h5" for i in range(n)]
datapaths_train = datapaths_val = datapaths_test = datapaths
dataset_class = DatasetIndices

print("get datasets")

#dataset_train = dataset_class(datapaths_train, "train", logger, pytables=pytables, optional_features=config['optional_features'])

#dataset_val = dataset_class(datapaths_val, "val", logger, pytables=pytables, optional_features=config['optional_features'])
dataset_test = dataset_class(datapaths_test, "test", logger, pytables=pytables, optional_features=model_config['optional_features'])

print("get dataloaders")
logger.info("Preparing DataLoaders...")
#dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=num_workers)
#dataloader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=True, num_workers=num_workers)
dataloader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False, num_workers=num_workers)
print(f"Total length = {dataset_test.__len__()*1e-6:.2f} Mil")
x, y = dataset_test[0]
input_features = x.shape[0]



model_state_dict = torch.load(model_to_eval_path + "model.pth", weights_only=False, map_location=torch.device('cpu'))
input_size = model_state_dict[list(model_state_dict.keys())[0]].shape[1]

if pretrain_model_config["model_type"] == "TwoStageModel":
    input_size += len(pretrain_model_config['optional_features'])

model = get_model(pretrain_model_config, input_size)
model.load_state_dict(model_state_dict)

if pretrain_model_config["model_type"] == "TwoStageModel":
    for param in model.fcn1.parameters():
        param.requires_grad = False

    for param in model.fcn2.parameters():
        param.requires_grad = True

elif pretrain_model_config["model_type"] == "FCN":
    for param in model.parameters():
        param.requires_grad = True

model = model.to(device)

model.eval()
num_batches = len(dataloader_test)
test_loss = 0

features = np.zeros((len(dataset_test),10))
preds = np.zeros(len(dataset_test))
targets = np.zeros(len(dataset_test))

# Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode

print("total",len(dataset_test))
index_start = 0
with torch.no_grad():
    for X, y in dataloader_test:
        print("start",index_start)
        
        index_end = index_start + X.shape[0]

        X, y = X.to(device), y.to(device)  
        pred = model(X)
        
        features[index_start:index_end] = X.numpy()
        preds[index_start:index_end] = pred.squeeze(1).numpy()
        targets[index_start:index_end] = y.squeeze(1).numpy()

        index_start = index_end

        print("end",index_end)

out_df = pd.DataFrame()
out_df["sm_lat_ipp"] = features[:,0]
out_df["sm_lon_ipp_sin"] = features[:,1]
out_df["sm_lon_ipp_cos"] = features[:,2]
out_df["sod_sin"] = features[:,3]
out_df["sod_cos"] = features[:,4]
out_df["satazi_cos"] = features[:,5]
out_df["satazi_sin"] = features[:,6]
out_df["satele"] = features[:,7]
out_df["doy"] = features[:,8]
out_df["year"] = features[:,9]
out_df["prediction"] = preds
out_df["target"] = targets

model_id = model_to_eval_path.split("/")[-2]
if not os.path.exists("outputs/"): 
    os.makedirs("outputs/")

out_df.to_csv("outputs/"+model_id+".csv",index=False)
