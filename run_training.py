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
from dataset.dataset import DatasetGNSS
from evaluation.test import test

from models.models import FCN
from training.training import train_single_epoch

logger = logging.getLogger(__name__)


datapaths = [f"/cluster/work/igp_psr/arrueegg/GNSS_STEC_DB/2024/{doi}/ccl_2024{doi}_30_5.h5" for doi in range(300, 301)]
splits_file = "/cluster/work/igp_psr/dslab_FS25_data_and_weights/split_exp.json"
# datapath = "V:/courses/dslab/team16/data/2023/020/ccl_2023020_30_5.h5"



if __name__ == "__main__":
    if not os.path.exists("outputs"): 
        os.makedirs("outputs")
                    
    torch.manual_seed(10)
    logging.basicConfig(filename='outputs/FCN.log', level=logging.INFO, format='%(asctime)s | %(message)s', datefmt='%H:%M')
    logger.info("-------------------------------------------------------\nStarting Script\n-------------------------------------------------------")

    device = torch.accelerator.current_accelerator().type if torch.cuda.is_available() else "cpu"
    logger.info("Using %s device", device)

    learning_rate = 1e-3
    batch_size = 64
    epochs = 1
    
    dataset_train = DatasetGNSS(datapaths, 0, splits_file)
    dataset_val = DatasetGNSS(datapaths, 1, splits_file)
    dataset_test = DatasetGNSS(datapaths, 2, splits_file)

    
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
        train_single_epoch(dataloader_train, model, loss, optimizer, device, logger,log_interval=6000)
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
    eval_loss_fct = nn.L1Loss()
    test_loss = test(dataloader_test, model, eval_loss_fct, device)
    logger.info(f"Evaluation MAE Loss: {test_loss:>7f}")
    logger.info("Completed.")
    
