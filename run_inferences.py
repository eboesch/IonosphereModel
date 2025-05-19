import os
import pandas as pd
import torch
import logging
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from dataset.dataset import DatasetIndices, DatasetSA
from datetime import datetime
from models.models import load_model
import yaml
import numpy as np
from tqdm import tqdm
import shutil
from torch import nn

inferences_config_path = "config/inferences_config.yaml"
solar_indices_path = "/cluster/work/igp_psr/dslab_FS25_data_and_weights/"

if __name__ == "__main__":
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

    evaluation_data = inferences_config["evaluation_data"]
    if evaluation_data in ["train", "val", "test"]:
        datapath = model_config["datapath"]
        logger.info(f"date range: {doy} until {doy+n-1} of {year}")
        assert doy + n <= 366, "Date range reaches end of year. Currently not supported."
        datapaths_test = [datapath + f"{year}/{str(doy+i).zfill(3)}/ccl_{year}{str(doy+i).zfill(3)}_30_5.h5" for i in range(n)]
        dataset_test = DatasetIndices(datapaths_test, evaluation_data, logger, pytables=pytables, solar_indices_path=solar_indices_path, optional_features=model_config['optional_features'], use_spheric_coords=model_config["use_spheric_coords"], normalize_features=model_config["normalize_features"])

    elif evaluation_data == "SA":
        sa_path = inferences_config["sa_path"]
        data = pd.read_csv(sa_path)
        data['time'] = pd.to_datetime(data['time'])
        print(data.shape)
        data = data[(data['time'].dt.dayofyear >= doy) & (data['time'].dt.dayofyear < doy + n)]
        print(data.shape)
        data = data[data['time'].dt.year == year]
        print(data.shape)
        dataset_test = DatasetSA(data, model_config['optional_features'], None, use_spheric_coords=model_config["use_spheric_coords"], normalize_features=model_config["normalize_features"])

    else:
        assert False
    
    logger.info(f"Total length of Test Dataset = {dataset_test.__len__()*1e-6:.2f} Mil")
    
    dataloader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    model = load_model(inferences_config["model_path"])
    logger.info("Model: %s", model)

    model = model.to(device)
    model.eval()
    num_batches = len(dataloader_test)
    test_loss = 0

    x, y = dataset_test[0]
    input_features = x.shape[0]
    features = np.zeros((len(dataset_test), input_features))
    preds = np.zeros(len(dataset_test))
    targets = np.zeros(len(dataset_test))
    
    print("total",len(dataset_test))
    index_start = 0
    with torch.no_grad():
        for X, y in tqdm(dataloader_test):
            print("start",index_start)
            
            index_end = index_start + X.shape[0]
            features[index_start:index_end] = X.numpy()
            targets[index_start:index_end] = y.squeeze(1).numpy()

            X, y = X.to(device), y.to(device)  
            pred = model(X)
            
            preds[index_start:index_end] = pred.squeeze(1).cpu().numpy()

            index_start = index_end

            print("end",index_end)

    model_id = inferences_config["model_path"].split("/")[-2]

    out_path = inferences_config["model_path"] + f"inferences_{timestamp}/"
    os.makedirs(out_path, exist_ok = True)
    shutil.copy(inferences_config_path, out_path + "inferences_config.yaml")

    metrics = pd.Series({
        "MSE": np.abs((targets - preds)**2).mean().item(),
        "MAE": np.abs(targets - preds).mean().item()
    })
    print(metrics)
    metrics.to_csv(out_path + "metrics.csv")


    if inferences_config["save_inferences"]:

        out_df = pd.DataFrame()
        out_df["sm_lat_ipp"] = features[:,0]
        out_df["sm_lon_ipp_sin"] = features[:,1]
        out_df["sm_lon_ipp_cos"] = features[:,2]
        out_df["sod_sin"] = features[:,3]
        out_df["sod_cos"] = features[:,4]
        out_df["satazi_sin"] = features[:,5]
        out_df["satazi_cos"] = features[:,6]
        out_df["satele"] = features[:,7]


        # NOTE: older models didn't have the two-tier format of optional_features.
        # for the sake of backwards comptatibility we make this distinction here
        if type(model_config["optional_features"]) == dict:
            initial = model_config["optional_features"]["initial"] or []
            delayed = model_config["optional_features"]["delayed"] or []
            optional_features = initial + delayed
        else:
            optional_features = model_config["optional_features"] or []

        idx = 8
        for feature in optional_features:
            out_df[feature] = features[:,idx]
            idx += 1
            
        # if model_config["optional_features"] is not None:
        #     if "doy" in model_config["optional_features"]:
        #         out_df["doy"] = features[:,idx]
        #         idx += 1

        #     if "year" in model_config["optional_features"]:
        #         out_df["year"] = features[:,idx]

        out_df["prediction"] = preds
        out_df["target"] = targets

        out_df.to_csv(out_path + "inferences.csv",index=False)

        # for debug
        out_df.to_csv("outputs/" + model_id + inferences_config['evaluation_data'] + str(year) + str(doy) + str(n) + ".csv", index=False)