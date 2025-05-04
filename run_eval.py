"""Script to load a pre-trained model and run evaluation and a test set"""
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


datapaths = [f"/cluster/work/igp_psr/arrueegg/GNSS_STEC_DB/2024/{doi}/ccl_2024{doi}_30_5.h5" for doi in range(300, 302)]
splits_file = "/cluster/work/igp_psr/dslab_FS25_data_and_weights/split_exp.json"
# datapath = "V:/courses/dslab/team16/data/2023/020/ccl_2023020_30_5.h5"



if __name__ == "__main__":
    if not os.path.exists("outputs"): 
        os.makedirs("outputs")
                    
    torch.manual_seed(10)
    logging.basicConfig(filename='outputs/FCN_eval.log', level=logging.INFO, format='%(asctime)s | %(message)s', datefmt='%H:%M')
    logger.info("-------------------------------------------------------\nStarting Script\n-------------------------------------------------------")

    device = torch.accelerator.current_accelerator().type if torch.cuda.is_available() else "cpu"
    logger.info("Using %s device", device)

    batch_size = 64
    
    dataset_test = DatasetGNSS(datapaths, 2, splits_file)

    
    logger.info("Preparing DataLoaders...")
    dataloader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False)


    logger.info("Setting up Model...")
    model = FCN().to(device)
    model_path = "model.pth"
    model.load_state_dict(torch.load(model_path, weights_only=True))
    # model.eval()

    logger.info("Model: %s", model)


    logger.info("Starting evaluation...")
    eval_loss_fct = nn.L1Loss()
    test_loss = test(dataloader_test, model, eval_loss_fct, device)
    logger.info(f"Evaluation MAE Loss: {test_loss:>7f}")
    logger.info("Completed.")
    
