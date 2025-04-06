import pandas as pd
import json
import numpy as np
from matplotlib import pyplot as plt
import h5py

def generate_split_train_val_test(data: pd.DataFrame, split_file_path: str):
    """
    Assigns each station a data category (train/val/test). Plots stations and corresponding category.
    Creates a JSON dictionary containing the data split of the stations.
    """

    # NOTE: pandas is only used to keep track of each station lon and lat, which are used for plotting reasons.
    data["station"] = data["station"].str.decode("utf8")
    stations_df = data[["station", "lat_sta", "lon_sta"]].groupby("station").agg("first").sort_values(by="station").reset_index()
    # create a list of all stations
    stations = stations_df["station"].tolist()

    # shuffle stations list assign a data category (train/val/test) to each entry
    np.random.seed(10)  
    np.random.shuffle(stations)
    nstations = len(stations)
    splits_dict = {station: (i / nstations > 0.6) + (i / nstations > 0.8) for i, station in enumerate(stations)}

    fig, ax = plt.subplots()
    stations_df["split"] = stations_df["station"].map(splits_dict)
        
    ax.scatter(stations_df["lon_sta"], stations_df["lat_sta"], c=stations_df["split"] / 2, cmap="viridis")
    fig.savefig("outputs/split.png")
    
    with open(split_file_path, 'w') as f:
        json.dump(splits_dict, f)
    

if __name__ == "__main__":
    datapath = "/cluster/work/igp_psr/arrueegg/GNSS_STEC_DB/2024/322/ccl_2024322_30_5.h5"
    file = h5py.File(datapath, 'r')
    year = list(file.keys())[0]
    day = list(file[year].keys())[0]
    data = file[year][day]["all_data"]
    data = pd.DataFrame(data['station', 'lon_sta', 'lat_sta'])  
    
    generate_split_train_val_test(data, "/cluster/work/igp_psr/dslab_FS25_data_and_weights/split_exp.json")