import h5py
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

datapath = "/cluster/work/igp_psr/arrueegg/GNSS_STEC_DB/2024/322/ccl_2024322_30_5.h5"

def get_data(datapath: str) -> pd.DataFrame:
    file = h5py.File(datapath, 'r')
    year = list(file.keys())[0]
    day = list(file[year].keys())[0]
    data = file[year][day]["all_data"]
    # TODO: figure out a better way than pandas to manipulate the data.
    data = pd.DataFrame(data[:])
    return data


def split_train_val_test(data: pd.DataFrame) -> pd.DataFrame:
    stations = data["station"].unique()
    np.random.seed(10)  
    np.random.shuffle(stations)
    nstations = len(stations)
    stations_map = {station: (i / nstations > 0.6) + (i / nstations > 0.8) for i, station in enumerate(stations)}
    
    data['split'] = data["station"].map(stations_map)   
    return data

def extract_features(data: pd.DataFrame) -> pd.DataFrame:
    data['sm_lon_ipp_s'] = np.sin(2 * np.pi * data['sm_lon_ipp'] / 360)
    data['sm_lon_ipp_c'] = np.cos(2 * np.pi * data['sm_lon_ipp'] / 360)
    data['sod_s'] = np.sin(2 * np.pi * data['sod'] / 86400) 
    data['sod_c'] = np.cos(2 * np.pi * data['sod'] / 86400) 
    data['satazi_s'] = np.sin(2 * np.pi * data['satazi'] / 360)
    data['satazi_c'] = np.cos(2 * np.pi * data['satazi'] / 360)
    return data
    


if __name__ == "__main__":
    torch.seed(10)
    data = get_data(datapath)
    data = split_train_val_test(data)
    data = extract_features(data)
    features = ['sm_lat_ipp', 'sm_lon_ipp_s', 'sm_lon_ipp_c', 'sod_s', 'sod_c', 'satele','satazi_s', 'satazi_c']
    X = torch.tensor(data[[features]].values)
    y = torch.tensor(data['stec'].values)
    
    dataset_train = TensorDataset(X[data['split'] == 0], y[data['split'] == 0])
    dataset_val = TensorDataset(X[data['split'] == 1], y[data['split'] == 1])
    dataset_test = TensorDataset(X[data['split'] == 2], y[data['split'] == 2])
    
    dataloader_train = DataLoader(dataset_train, batch_size=64, shuffle=True)
    dataloader_val = DataLoader(dataset_val, batch_size=64, shuffle=True)
    dataloader_test = DataLoader(dataset_test, batch_size=64, shuffle=False)
    
    # TODO: Define a fully connected network model
    # TODO: Add some training code
    # TODO: Add some eval code