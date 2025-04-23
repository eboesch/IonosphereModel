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
from dataset.dataset import DatasetReorganized
from evaluation.test import test
from datetime import datetime
from models.models import get_model_class_from_string
from training.training import train_single_epoch
import yaml
import shutil

logger = logging.getLogger(__name__)


dslab_path = "/cluster/work/igp_psr/dslab_FS25_data_and_weights/"
datapaths_train = [dslab_path + f"reorganized_data/2023-{i}-train.h5" for i in range(1, 13)]
datapaths_val = [dslab_path + f"reorganized_data/2023-{i}-val.h5" for i in range(1, 13)]
datapaths_test = [dslab_path + f"reorganized_data/2023-{i}-test.h5" for i in range(1, 13)]



if __name__ == "__main__":
    print("HELLO")
    if not os.path.exists(dslab_path + "pretrained_models"): 
        os.makedirs(dslab_path + "pretrained_models")

    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    training_id = "model_" + timestamp

    model_path = dslab_path + "pretrained_models/" + training_id + "/"
    assert not os.path.exists(model_path), f"Training ID {training_id} already exists"
          
    os.makedirs(model_path)
    
    config_path = "config/training_config.yaml"
    with open(config_path, 'r') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    shutil.copy(config_path, model_path + "training_config.yaml")
    learning_rate = config["learning_rate"]
    batch_size = config["batch_size"]
    epochs = config["epochs"]
    model_type = config["model_type"]
    num_workers = config["num_workers"]

    torch.manual_seed(10)
    logging.basicConfig(filename=model_path + 'logs.log', level=logging.INFO, format='%(asctime)s | %(message)s', datefmt='%H:%M')
    logger.info("-------------------------------------------------------\nStarting Script\n-------------------------------------------------------")

    device = torch.accelerator.current_accelerator().type if torch.cuda.is_available() else "cpu"
    logger.info("Using %s device", device)

    logger.info(f"learning_rate: {learning_rate}")
    logger.info(f"batch_size: {batch_size}")
    logger.info(f"epochs: {epochs}")
    logger.info(f"datapaths: {datapaths_train}")
  
    dataset_train = DatasetReorganized(datapaths_train, logger)
    x, y = dataset_train[0]
    input_features = x.shape[0]
    dataset_val = DatasetReorganized(datapaths_val, logger)
    dataset_test = DatasetReorganized(datapaths_test, logger)

    
    logger.info("Preparing DataLoaders...")
    dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    dataloader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    dataloader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False, num_workers=num_workers)


    logger.info("Setting up Model...")
    model_class = get_model_class_from_string(model_type)
    model = model_class(input_features).to(device)
    logger.info("Model: %s", model)

    loss = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

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
    