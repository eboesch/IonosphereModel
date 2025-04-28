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
import glob


logger = logging.getLogger(__name__)


if __name__ == "__main__":
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    print("Started ", timestamp)

    config_path = "config/training_config.yaml"
    with open(config_path, 'r') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    dslab_path = config["dslab_path"]
    models_path = dslab_path + config["models_dir_name"] + "/"

    if not os.path.exists(models_path): 
        os.makedirs(models_path)

    training_id = "model_" + timestamp

    model_path = models_path + training_id + "/"
    assert not os.path.exists(model_path), f"Training ID {training_id} already exists"
          
    os.makedirs(model_path)
    shutil.copy(config_path, model_path + "trainig_config.yaml")

    learning_rate = config["learning_rate"]
    batch_size = config["batch_size"]
    epochs = config["epochs"]
    num_workers = config["num_workers"]
    use_reorganized_data = config["use_reorganized_data"]
    pytables = config["pytables"]


    torch.manual_seed(10)
    logging.basicConfig(handlers=[logging.FileHandler(model_path + 'logs.log'), logging.StreamHandler()], level=logging.INFO, format='%(asctime)s | %(message)s', datefmt='%H:%M')
    logger.info("-------------------------------------------------------\nStarting Script\n-------------------------------------------------------")

    device = torch.accelerator.current_accelerator().type if torch.cuda.is_available() else "cpu"
    logger.info("Using %s device", device)

    logger.info(f"learning_rate: {learning_rate}")
    logger.info(f"batch_size: {batch_size}")
    logger.info(f"epochs: {epochs}")

    if use_reorganized_data:
        reorganized_datapath = config["dslab_path"] + config["reorganized_data_dir_name"] + "/"
        years_dict = config["years"]
        datapaths_train = []
        datapaths_val = []
        datapaths_test = []

        for year, months_in_year in years_dict.items():
            if months_in_year != "all":
                # TODO implement logic in case we dont want to select all months
                raise NotImplementedError
            
            datapaths_train += glob.glob(reorganized_datapath + f"{year}*train.h5")
            datapaths_val += glob.glob(reorganized_datapath + f"{year}*val.h5")
            datapaths_test += glob.glob(reorganized_datapath + f"{year}*test.h5")
        dataset_class = DatasetReorganized

    else:
        datapath = config["datapath"]
        doy = config["start_doy"]
        year = config["year"]
        n = config["num_days"]
        logger.info(f"date range: {doy} until {doy+n-1} of {year}")
        assert doy + n <= 366, "Date range reaches end of year. Currently not supported."
        datapaths = [datapath + f"{year}/{str(doy+i).zfill(3)}/ccl_{year}{str(doy+i).zfill(3)}_30_5.h5" for i in range(n)]
        datapaths_train = datapaths_val = datapaths_test = datapaths
        dataset_class = DatasetIndices

    print("get datasets")

    dataset_train = dataset_class(datapaths_train, "train", logger, pytables=pytables, optional_features=config['optional_features'])
    print(f"Total length = {dataset_train.__len__()*1e-6:.2f} Mil")
    x, y = dataset_train[0]
    input_features = x.shape[0]
    dataset_val = dataset_class(datapaths_val, "val", logger, pytables=pytables, optional_features=config['optional_features'])
    dataset_test = dataset_class(datapaths_test, "test", logger, pytables=pytables, optional_features=config['optional_features'])

    print("get dataloaders")
    logger.info("Preparing DataLoaders...")
    dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    dataloader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    dataloader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False, num_workers=num_workers)


    logger.info("Setting up Model...")
    pretrained_model_path = config["pretrained_model_path"]
    if pretrained_model_path is None:
        model_type = config["model_type"]
        model = get_model(config, input_features)
        model = model.to(device)
        logger.info("Model: %s", model)
    else:
        model = load_pretrained_model(pretrained_model_path)

    loss = nn.MSELoss()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)

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
        # else:
        #     # validation loss is increasing, so we stop training
        #     logger.info("Validation loss increased. Stopping training.")
        #     break
        
    logger.info("Training completed.")


    logger.info("Starting evaluation...")
    eval_loss_fct = nn.L1Loss()
    test_loss = test(dataloader_test, model, eval_loss_fct, device)
    logger.info(f"Evaluation MAE Loss: {test_loss:>7f}")
    logger.info("Completed.")
    
