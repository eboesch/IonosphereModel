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
from torch.utils.data import DataLoader
from dataset.dataset import DatasetIndices
from evaluation.test import test
from datetime import datetime
from models.models import get_model_class_from_string
from training.training import train_single_epoch
import yaml
import shutil
import tables

logger = logging.getLogger(__name__)


# datapaths = [f"/cluster/work/igp_psr/arrueegg/GNSS_STEC_DB/2024/{doi}/ccl_2024{doi}_30_5.h5" for doi in range(300, 320)]
dslab_path = "/cluster/work/igp_psr/dslab_FS25_data_and_weights/"



if __name__ == "__main__":
    print("Started ", datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
    if not os.path.exists(dslab_path + "models"): 
        os.makedirs(dslab_path + "models")

    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    training_id = "model_" + timestamp

    model_path = dslab_path + "models/" + training_id + "/"
    assert not os.path.exists(model_path), f"Training ID {training_id} already exists"
          
    os.makedirs(model_path)
    
    config_path = "config/training_config.yaml"
    with open(config_path, 'r') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    shutil.copy(config_path, model_path + "trainig_config.yaml")
    learning_rate = config["learning_rate"]
    batch_size = config["batch_size"]
    epochs = config["epochs"]
    model_type = config["model_type"]
    num_workers = config["num_workers"]


    torch.manual_seed(10)
    logging.basicConfig(handlers=[logging.FileHandler(model_path + 'logs.log'), logging.StreamHandler()], level=logging.INFO, format='%(asctime)s | %(message)s', datefmt='%H:%M')
    logger.info("-------------------------------------------------------\nStarting Script\n-------------------------------------------------------")

    device = torch.accelerator.current_accelerator().type if torch.cuda.is_available() else "cpu"
    logger.info("Using %s device", device)

    logger.info(f"learning_rate: {learning_rate}")
    logger.info(f"batch_size: {batch_size}")
    logger.info(f"epochs: {epochs}")


    doy = config["start_doy"]
    year = config["year"]
    n = config["num_days"]
    logger.info(f"date range: {doy} until {doy+n-1} of {year}")
    assert doy + n <= 366, "Date range reaches end of year. Currently not supported."

    datapaths = [f"/cluster/work/igp_psr/arrueegg/GNSS_STEC_DB/{year}/{str(doy+i).zfill(3)}/ccl_{year}{str(doy+i).zfill(3)}_30_5.h5" for i in range(n)]
    
    print("get datasets")

    dataset_train = DatasetIndices(datapaths, "train", logger, pytables=True)
    print(f"Total length = {dataset_train.__len__()*1e-6:.2f} Mil")
    x, y = dataset_train[0]
    input_features = x.shape[0]
    dataset_val = DatasetIndices(datapaths, "val", logger, pytables=True)
    dataset_test = DatasetIndices(datapaths, "test", logger, pytables=True)

    print("get dataloaders")
    logger.info("Preparing DataLoaders...")
    dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    dataloader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    dataloader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False, num_workers=num_workers)


    logger.info("Setting up Model...")
    model_class = get_model_class_from_string(model_type)
    model = model_class(input_features, config["num_hidden_layers"], config["hidden_size"]).to(device)
    logger.info("Model: %s", model)

    loss = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    print("start training")
    logger.info("Starting training...")
    best_val_loss = float('inf')
    for t in range(epochs):
        logger.info("-------------------------------\nEpoch %s\n-------------------------------", t+1)
        train_single_epoch(dataloader_train, model, loss, optimizer, device, logger,log_interval=3000)
        val_loss = test(dataloader_val, model, loss, device)
        logger.info(f"Validation Loss: {val_loss:>7f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), model_path + "model.pth")
        else:
            # validation loss is increasing, so we stop training
            logger.info("Validation loss increased. Stopping training.")
            break
        
    logger.info("Training completed.")


    logger.info("Starting evaluation...")
    eval_loss_fct = nn.L1Loss()
    test_loss = test(dataloader_test, model, eval_loss_fct, device)
    logger.info(f"Evaluation MAE Loss: {test_loss:>7f}")
    logger.info("Completed.")
    
