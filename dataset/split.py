import pandas as pd
import json
import numpy as np
from matplotlib import pyplot as plt
import h5py

def generate_split_train_val_test(data: pd.DataFrame, splits_path: str, train_frac=0.8) -> None:
    """
    Assigns each station a data category (train/val/test). Plots stations and corresponding category.
    Creates a JSON dictionary containing the data split of the stations.
    """
    test_frac = (1 - train_frac) / 2
    train_val_frac = 1 - test_frac

    # NOTE: pandas is only used to keep track of each station lon and lat, which are used for plotting reasons.
    # data["station"] = data["station"].str.decode("utf8")
    stations_df = data[["station", "lat_sta", "lon_sta"]].groupby("station").agg("first").sort_values(by="station").reset_index()
    # create a list of all stations
    stations = stations_df["station"].tolist()

    # shuffle stations list assign a data category (train/val/test) to each entry
    seed = 2
    np.random.seed(seed)  
    np.random.shuffle(stations)
    nstations = len(stations)
    train_stations = stations[:int(np.floor(nstations*train_frac))]
    val_stations = stations[int(np.floor(nstations*train_frac)):int(np.floor(nstations*train_val_frac))]
    test_stations = stations[int(np.floor(nstations*train_val_frac)):]
    
    assert len(train_stations) + len(val_stations) + len(test_stations) == nstations
    
    splits_dict = {station: 0.5*(station in val_stations) + (station in test_stations) for i, station in enumerate(stations)}

    fig, ax = plt.subplots()
    stations_df["split"] = stations_df["station"].map(splits_dict)
        
    ax.scatter(stations_df["lon_sta"], stations_df["lat_sta"], c=stations_df["split"], cmap="viridis")
    fig.savefig(f"outputs/splits_{seed}.png")
    
    np.save(splits_path + "train.npy", train_stations)
    np.save(splits_path + "val.npy", val_stations)
    np.save(splits_path + "test.npy", test_stations)


def plot_predefined_splits(data: pd.DataFrame) -> None:
    """
    Plots the stations according to data split.
    """
    stations_df = data[["station", "lat_sta", "lon_sta"]].groupby("station").agg("first").sort_values(by="station").reset_index()
    stations = stations_df["station"].tolist()

    with open("dataset/train.list", "r") as file:
        train_stations = [line.strip() for line in file]
        
    with open("dataset/val.list", "r") as file:
        val_stations = [line.strip() for line in file]

    with open("dataset/test.list", "r") as file:
        test_stations = [line.strip() for line in file]    
        
    splits_dict = {station.encode('utf8'): 0 for station in train_stations}
    splits_dict.update({station.encode('utf8'): 0.5 for station in val_stations})
    splits_dict.update({station.encode('utf8'): 1 for station in test_stations})

    fig, ax = plt.subplots()
    stations_df["split"] = stations_df["station"].map(splits_dict)
        
    ax.scatter(stations_df["lon_sta"], stations_df["lat_sta"], c=stations_df["split"], cmap="viridis")
    fig.savefig(f"outputs/splits_arno.png")

if __name__ == "__main__":
    datapath = "/cluster/work/igp_psr/arrueegg/GNSS_STEC_DB/2024/300/ccl_2024300_30_5.h5"
    file = h5py.File(datapath, 'r')
    year = list(file.keys())[0]
    day = list(file[year].keys())[0]
    data = file[year][day]["all_data"]
    data = pd.DataFrame(data['station', 'lon_sta', 'lat_sta'])  
    
    # generate_split_train_val_test(data, "/cluster/work/igp_psr/dslab_FS25_data_and_weights/")
    plot_predefined_splits(data)