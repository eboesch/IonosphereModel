from torch.utils.data import Dataset
import torch
import tables
import os
import numpy as np
from numpy.typing import NDArray
from logging import Logger
import logging
import h5py
import time
import psutil
import os
import pandas as pd
import pandas as pd


def monitor_access(data, index):
    # Record time and CPU info before access
    process = psutil.Process(os.getpid())
    before_io = psutil.disk_io_counters()
    before_cpu = process.cpu_times()
    before_mem = process.memory_info()
    start = time.perf_counter()
    before_net = psutil.net_io_counters()

    # Actual data access
    row = data[index]

    end = time.perf_counter()
    after_io = psutil.disk_io_counters()
    after_cpu = process.cpu_times()
    after_mem = process.memory_info()
    
    # ...
    after_net = psutil.net_io_counters()
    net_bytes = after_net.bytes_recv - before_net.bytes_recv
    net_kb = net_bytes / 1024

    # Metrics
    duration = end - start
    io_read_bytes = after_io.read_bytes - before_io.read_bytes
    cpu_time = (after_cpu.user + after_cpu.system) - (before_cpu.user + before_cpu.system)
    memory_used_mb = after_mem.rss / 1024**2
    delta_mem = (after_mem.rss - before_mem.rss) / 1024**2  # In MB

    print(f"Index: {index}")
    print(f"  Access time: {duration:.4f} sec")
    print(f"  Disk read: {io_read_bytes / 1024:.1f} KB")
    print(f"  CPU time: {cpu_time:.4f} sec")
    print(f"  RAM used: {memory_used_mb:.1f} MB")
    print(f"  RAM delta: {delta_mem:+.1f} MB (RSS)")
    print(f"  network: {net_kb}")
    print("---")

    return row


def get_features_from_row(row: NDArray, optional_features_values: list, use_spheric_coords: bool = False, normalize_features: bool = False):
    if use_spheric_coords:
        angle_coords = [
            np.cos(2 * np.pi * row['satele'] / 360) * np.cos(2 * np.pi * row['satazi'] / 360),
            np.cos(2 * np.pi * row['satele'] / 360) * np.sin(2 * np.pi * row['satazi'] / 360),
            np.sin(2 * np.pi * row['satele'] / 360)
        ]

    else:
        angle_coords = [
            np.sin(2 * np.pi * row['satazi'] / 360),
            np.cos(2 * np.pi * row['satazi'] / 360),
            row['satele'] if not normalize_features else row['satele'] / 45 - 1.0,
        ]
    x = torch.tensor(
            [
                row['sm_lat_ipp'] if not normalize_features else row['sm_lat_ipp'] / 90,
                np.sin(2 * np.pi * row['sm_lon_ipp'] / 360),
                np.cos(2 * np.pi * row['sm_lon_ipp'] / 360),
                np.sin(2 * np.pi * row['sod'] / 86400),
                np.cos(2 * np.pi * row['sod'] / 86400), 
            ] + angle_coords + optional_features_values,
            dtype=torch.float32
        )
    y = torch.tensor([row['stec']]) # label
    return x, y


def get_file_and_data(filepath: str, nodepath: str, pytables: bool):
    if pytables:
        file = tables.open_file(filepath, mode='r', driver='H5FD_SEC2')
        data = file.get_node(nodepath)
    else:
        file = h5py.File(filepath, 'r')
        data = file[nodepath][:]

    return file, data


def get_daily_solar_indices(year, doy, solar_indices_df):
    """
    Returns an array of the daily average solar indices of the given day in the given year.
    """
    row = solar_indices_df.loc[(solar_indices_df["year"] == year) & (solar_indices_df["doy"] == doy)] # filter for the correct row
    solar_indices = row[["kp_index", "r_index", "dst_index", "f_index"]].to_numpy().reshape(-1) # extract the desired indices 
    return solar_indices

def get_hourly_solar_indices(year, doy, sod, solar_indices_df):
    """
    Returns a numpy array of the hourly solar indices of the given day in the given year.
    """
    hour = np.round(sod/3600).astype(int)
    if hour < 24:
        row = solar_indices_df.loc[(solar_indices_df["year"] == year) & (solar_indices_df["doy"] == doy) & (solar_indices_df["hour"] == hour)]
    else: # retrieve values of midnight of next day
        last_doy = {2020: 366, 2021: 365, 2022: 365, 2023: 365, 2024: 366, 2025: 365}
        if doy == last_doy[year]:
            # if we're at the last day of the year, we need get midnight of the first day of the next year
            row = solar_indices_df.loc[(solar_indices_df["year"] == year+1) & (solar_indices_df["doy"] == 1) & (solar_indices_df["hour"] == 0)].copy()
        else:
            row = solar_indices_df.loc[(solar_indices_df["year"] == year) & (solar_indices_df["doy"] == doy+1) & (solar_indices_df["hour"] == 0)].copy() # get midnight of the next day
    
        prev_row = row = solar_indices_df.loc[(solar_indices_df["year"] == year) & (solar_indices_df["doy"] == doy) & (solar_indices_df["hour"] == 23)].copy()
        row['r_index'] = prev_row['r_index'] # r_index and f_index are the same all day -> keep the values of the current day
        row['f_index'] = prev_row['f_index']

    solar_indices = row[["kp_index", "r_index", "dst_index", "f_index"]].to_numpy().reshape(-1) # extract the desired indices 
    return solar_indices

class DatasetIndices(Dataset):
    # An adaptation of https://github.com/arrueegg/STEC_pretrained/blob/main/src/utils/data_SH.py
    def __init__(self, datapaths: list[str], split: str, logger: Logger, pytables: bool, solar_indices_path, optional_features = ['doy', 'year'], use_spheric_coords=False, normalize_features=False):
        """
        Creates an instance of DatasetGNSS. 
        
        Attributes:
            datapaths_info :    List of dictionaries. Each dictionary contains: datapath, year, doy, indices 
                (of datapoints whose stations or in the given split), current_start_point (start position in overall dataset)
                of a file in the dataset.
            stations :  list of all stations matching the given split type, encoded to bytes.
            lenght :    length of the Dataset
        """

        self.optional_features = optional_features or []
        if type(self.optional_features) == dict:
            for tier in self.optional_features:
                self.optional_features[tier] = self.optional_features[tier] or []


        
        
        self.use_spheric_coords = use_spheric_coords
        self.normalize_features = normalize_features

        self.datapaths_info = []
        with open(f"dataset/{split}.list", "r") as file:
            self.stations = [line.strip().encode('utf8') for line in file]

        current_start_point = 0

        for datapath in datapaths:            
            if not(os.path.isfile(datapath)):
                # delete this day from datapaths, as there is no data available for that day 
                logger.info(f"Skipping {datapath}")
                continue
            
            # extract year and doy from datapath
            year = datapath.split('/')[-3]
            doy = datapath.split('/')[-2]
            
            file, data = get_file_and_data(datapath, f"/{year}/{doy}/all_data", pytables)
            indices = self._get_indices(data, pytables) #TODO: need to be sure that the given file actually exists
            
            self.datapaths_info.append(
                {
                    "datapath": datapath,
                    "file": file,
                    "data": data,
                    "year": year,
                    "doy": doy,
                    "indices": indices,
                    "start_point": current_start_point
                }
            )
            current_start_point += len(indices)
            # logger.info(f"Completed {datapath}")

        hourly_path = solar_indices_path + "omni2_solar_indices_hourly.lst"
        labels = ["year", "doy", "hour", "kp_index", "r_index", "dst_index", "f_index"]
        self.solar_indices_hourly = pd.read_csv(hourly_path, sep='\s+', names=labels)
        # self.solar_indices = pd.read_csv(hourly_path, delim_whitespace=True, names=labels)

        daily_path = solar_indices_path + "omni2_solar_indices_daily.lst"
        self.solar_indices_daily = pd.read_csv(daily_path, sep='\s+', names=labels)
        # self.solar_indices_daily = pd.read_csv(daily_path, delim_whitespace=True, names=labels)
            
        self.length = current_start_point
            
    def _get_indices(self, data, pytables) -> NDArray:
        """
        Returns the indices of all datapoints whose stations is in the current datasplit.
        """
        # select indices of datapoints whose stations that are in the current split
        if pytables:
            mask_for_split = np.isin(data.col('station'), self.stations) 
        else:
            mask_for_split = np.isin(data['station'], self.stations)

        indices = np.arange(0, len(mask_for_split), 1)[mask_for_split]
        del mask_for_split
        return indices
    
    def __getitem__(self, index: int) -> tuple[torch.tensor, torch.tensor]:
        """
        Loads and returns a sample from the dataset at the given index as a tensor. 
        Features are transformed as necessary. 
        """
        # iterate through all datapaths to find the file that contains our desired index
        for curr_datapath_info, next_datapath_info in zip(self.datapaths_info[:-1], self.datapaths_info[1:]):
            if next_datapath_info['start_point'] > index:
                datapath_info = curr_datapath_info
                break
        else:
            datapath_info = self.datapaths_info[-1]

        doy = float(datapath_info['doy'])
        year = float(datapath_info['year'])
        indices = datapath_info['indices']
        start_point = datapath_info['start_point']

        data = datapath_info['data']
        row = data[indices[index - start_point]]

        sod = row['sod']
        
        daily_solar_indices = get_daily_solar_indices(year, int(doy), self.solar_indices_daily)
        hourly_solar_indices = get_hourly_solar_indices(year, int(doy), sod, self.solar_indices_hourly)
        
        optional_features_dict = {
            'kp_index_hourly': hourly_solar_indices[0],
            'r_index_hourly': hourly_solar_indices[1],
            'dst_index_hourly': hourly_solar_indices[2],
            'f_index_hourly': hourly_solar_indices[3],
            'doy': doy if not self.normalize_features else doy / 183 - 1.0,
            'year': year,
            'kp_index_daily': daily_solar_indices[0],
            'r_index_daily': daily_solar_indices[1],
            'dst_index_daily': daily_solar_indices[2],
            'f_index_daily': daily_solar_indices[3],
        }

        optional_features_values = []
        if type(self.optional_features) == dict:
            for tier in self.optional_features:
                for feature in self.optional_features[tier]:
                    optional_features_values.append(optional_features_dict[feature])
        else:
            for feature in self.optional_features:
                optional_features_values.append(optional_features_dict[feature])

        x, y = get_features_from_row(row, optional_features_values, self.use_spheric_coords, self.normalize_features)
        
        return x, y
        
    def __len__(self):
        """
        Returns length of the dataset in use.
        """
        return self.length
    
    def __del__(self):
        for datapath_info in self.datapaths_info:
            datapath_info["file"].close()



class DatasetReorganized(Dataset):
    # NOTE: split is only passed to match the same signature as DatasetIndices
    def __init__(self, datapaths: list[str], split: str, logger: Logger, pytables: bool, solar_indices_path, optional_features = ['doy', 'year'], use_spheric_coords=False, normalize_features=False):
        
        self.optional_features = optional_features or []
        for tier in self.optional_features:
            self.optional_features[tier] = self.optional_features[tier] or []
        self.use_spheric_coords = use_spheric_coords
        self.normalize_features = normalize_features

        current_start_point = 0
        self.datapaths_info = []

        for datapath in datapaths:            
            if not(os.path.isfile(datapath)):
                logger.info(f"Skipping {datapath}")
                continue
            
            filename =  datapath.split('/')[-1]
            year = filename.split('-')[0]
            month = datapath.split('-')[1]

            file, data = get_file_and_data(datapath, "/all_data", pytables)

            self.datapaths_info.append(
                {
                    "datapath": datapath,
                    "file": file,
                    "data": data,
                    "start_point": current_start_point,
                    "year": year,
                    "month": month
                }
            )
            current_start_point += data.shape[0]
            logger.info(f"Completed {datapath}")

        hourly_path = solar_indices_path + "omni2_solar_indices_hourly.lst"
        labels = ["year", "doy", "hour", "kp_index", "r_index", "dst_index", "f_index"]
        self.solar_indices_hourly = pd.read_csv(hourly_path, sep='\s+', names=labels)
        # self.solar_indices = pd.read_csv(hourly_path, delim_whitespace=True, names=labels)

        daily_path = solar_indices_path + "omni2_solar_indices_daily.lst"
        self.solar_indices_daily = pd.read_csv(daily_path, sep='\s+', names=labels)
        # self.solar_indices_daily = pd.read_csv(daily_path, delim_whitespace=True, names=labels)
            
        self.length = current_start_point

    def __getitem__(self, index: int) -> tuple[torch.tensor, torch.tensor]:
        """
        Loads and returns a sample from the dataset at the given index as a tensor. 
        Features are transformed as necessary. 
        """
        # iterate through all datapaths to find the file that contains our desired index
        for curr_datapath_info, next_datapath_info in zip(self.datapaths_info[:-1], self.datapaths_info[1:]):
            if next_datapath_info['start_point'] > index:
                datapath_info = curr_datapath_info
                break
        else:
            datapath_info = self.datapaths_info[-1]

        start_point = datapath_info['start_point']
        data = datapath_info['data']

        year = float(datapath_info['year'])
        row = data[index - start_point]

        # NOTE: When reorganizing the data it got annoying to create a new column for day but overwriting an
        # existing column was straightforward, so I saved the doy in gphase
        doy = row['gfphase']
        # doy = row['doy']
        sod = row['sod']
        
        daily_solar_indices = get_daily_solar_indices(year, int(doy), self.solar_indices_daily)
        hourly_solar_indices = get_hourly_solar_indices(year, int(doy), sod, self.solar_indices_hourly)
        
        optional_features_dict = {
            'kp_index_hourly': hourly_solar_indices[0],
            'r_index_hourly': hourly_solar_indices[1],
            'dst_index_hourly': hourly_solar_indices[2],
            'f_index_hourly': hourly_solar_indices[3],
            'doy': doy if not self.normalize_features else doy / 183 - 1.0,
            'year': year,
            'kp_index_daily': daily_solar_indices[0],
            'r_index_daily': daily_solar_indices[1],
            'dst_index_daily': daily_solar_indices[2],
            'f_index_daily': daily_solar_indices[3],
        }

        optional_features_values = []
        for tier in self.optional_features:
            for feature in self.optional_features[tier]:
                optional_features_values.append(optional_features_dict[feature])
                
        x, y = get_features_from_row(row, optional_features_values)
        
        return x, y


    def __len__(self):
        """
        Returns length of the dataset in use.
        """
        return self.length
    
    def __del__(self):
        for datapath_info in self.datapaths_info:
            datapath_info["file"].close()


class DatasetSA(Dataset):
    def __init__(self, df, optional_features = ['doi', 'year'], satazi=None, use_spheric_coords=False, normalize_features=False):
        df = df.rename(columns={'sm_lat': 'sm_lat_ipp', 'sm_lon': 'sm_lon_ipp'})
        print(df.columns)
        df['time'] = pd.to_datetime(df['time'])
        df['sod'] = df['time'].dt.second + 60*df['time'].dt.minute + 3600*df['time'].dt.hour
        df['doy'] = df['time'].dt.dayofyear
        df['year'] = df['time'].dt.year
        df['satele'] = 90
        df['stec'] = df['vtec']
        self.df = df
        self.satazi = satazi
        if optional_features is None:
            self.optional_features = []
        else:
            self.optional_features = optional_features
        self.use_spheric_coords = use_spheric_coords
        self.normalize_features = normalize_features


    def __getitem__(self, index):
        row = self.df.iloc[index].copy()
        year = float(row['year'])
        doy = float(row['doy'])
        optional_features_dict = {
            'doy': doy if not self.normalize_features else doy / 183 - 1.0,
            'year': year
        }
        if self.satazi is None:
            row["satazi"] = 360*np.random.uniform()
        else:
            row["satazi"] = self.satazi

        optional_features_values = [val for key, val in optional_features_dict.items() if key in self.optional_features]
        x, y = get_features_from_row(row, optional_features_values, self.use_spheric_coords, self.normalize_features)
        return x, y

    def __len__(self):
        return self.df.shape[0]
        


if __name__ == "__main__":
    logger = logging.getLogger(__name__)

    # datapaths = [f"/cluster/work/igp_psr/arrueegg/GNSS_STEC_DB/2024/{str(doy).zfill(3)}/ccl_2024{str(doy).zfill(3)}_30_5.h5" for doy in range(1, 3)]
    # train_dataset = DatasetGNSS(datapaths, "train", logger)
    # for i in range(100):
    #     print(train_dataset[i])
    # train_dataset.__del__()

    import yaml
    # config_path = "config/pretraining_config.yaml"
    config_path = "config/training_config.yaml"
    with open(config_path, 'r') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)


    datapath = "/cluster/work/igp_psr/arrueegg/GNSS_STEC_DB/"
    dslab_path = "/cluster/work/igp_psr/dslab_FS25_data_and_weights/"
    # datapaths_train = [dslab_path + f"reorganized_data_2/2023-{i}-train.h5" for i in range(1, 12)]
    # train_dataset = DatasetReorganized(datapaths_train, 'train', logger, True, config['solar_indices_path'], optional_features = config['optional_features'])
    datapaths_train = [datapath + f"2024/{str(183+i).zfill(3)}/ccl_2024{str(183+i).zfill(3)}_30_5.h5" for i in range(2)]
    train_dataset = DatasetIndices(datapaths_train, 'train', logger, True, config['solar_indices_path'], optional_features = config['optional_features'])
    train_dataset[10]
    train_dataset.__del__()