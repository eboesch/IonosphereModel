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


def get_solar_indices(solar_indices_path):
    hourly_path = solar_indices_path + "omni2_solar_indices_hourly.lst"
    labels = ["year", "doy", "hour", "kp_index", "r_index", "dst_index", "f_index"]
    solar_indices_hourly = pd.read_csv(hourly_path, sep='\s+', names=labels)
    last_doy = {2020: 366, 2021: 365, 2022: 365, 2023: 365, 2024: 366, 2025: 365}
    solar_indices_lookup_hourly = {
        (row["year"], row["doy"], row["hour"]): row[["kp_index", "r_index", "dst_index", "f_index"]].to_numpy().reshape(-1) 
        for _, row in solar_indices_hourly.iterrows()
    }
    for _, row in solar_indices_hourly.iterrows():
        if row["hour"] == 23:
            if last_doy[row["year"]] == row["doy"]:
                if row["year"] < 2024:
                    solar_indices_lookup_hourly.update(
                        {(row["year"], row["doy"], 24): solar_indices_lookup_hourly[(row["year"]+1, 1, 0)]}
                    )
                else:
                    solar_indices_lookup_hourly.update(
                        {(row["year"], row["doy"], 24): solar_indices_lookup_hourly[(row["year"], row["doy"], 23)]}
                    )
            else:
                solar_indices_lookup_hourly.update(
                    {(row["year"], row["doy"], 24): solar_indices_lookup_hourly[(row["year"], row["doy"] + 1, 0)]}
                )
        
    daily_path = solar_indices_path + "omni2_solar_indices_daily.lst"
    solar_indices_daily = pd.read_csv(daily_path, sep='\s+', names=labels)
    solar_indices_lookup_daily = {
        (row["year"], row["doy"]): row[["kp_index", "r_index", "dst_index", "f_index"]].to_numpy().reshape(-1) 
        for _, row in solar_indices_daily.iterrows()
    }

    return solar_indices_lookup_daily, solar_indices_lookup_hourly


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

def get_only_features_from_row(row: NDArray, optional_features_values: list, use_spheric_coords: bool = False, normalize_features: bool = False):
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
    return x

def get_file_and_data(filepath: str, nodepath: str, pytables: bool):
    if pytables:
        file = tables.open_file(filepath, mode='r', driver='H5FD_SEC2')
        data = file.get_node(nodepath)
    else:
        file = h5py.File(filepath, 'r')
        data = file[nodepath][:]

    return file, data


class DatasetGNSS(Dataset):
    solar_indices_daily: dict
    solar_indices_hourly: dict
    optional_features: list | dict
    normalize_features: bool

    def get_optional_features(self, year, doy, sod):
        optional_features_dict = {
            'doy': doy if not self.normalize_features else doy / 183 - 1.0,
            'year': year,
        }
        if isinstance(self.optional_features, list):
            optional_features_names = self.optional_features
        else:
            optional_features_names = self.optional_features["delayed"]
            assert len(self.optional_features["initial"]) == 0

        if np.any(["daily" in f for f in optional_features_names]):
            daily_solar_indices = self.solar_indices_daily[(int(year), int(doy))]
            optional_features_dict.update(
                {
                    'kp_index_daily': daily_solar_indices[0],
                    'r_index_daily': daily_solar_indices[1],
                    'dst_index_daily': daily_solar_indices[2],
                    'f_index_daily': daily_solar_indices[3],
                }
            )

        if np.any(["hourly" in f for f in optional_features_names]):
            hour = np.round(sod/3600).astype(int)
            hourly_solar_indices = self.solar_indices_hourly[(int(year), int(doy), int(hour))]
            optional_features_dict.update(
                {
                    'kp_index_hourly': hourly_solar_indices[0],
                    'r_index_hourly': hourly_solar_indices[1],
                    'dst_index_hourly': hourly_solar_indices[2],
                    'f_index_hourly': hourly_solar_indices[3],
                }
            )

        optional_features_values = []
        if type(self.optional_features) == dict:
            for tier in self.optional_features:
                for feature in self.optional_features[tier]:
                    optional_features_values.append(optional_features_dict[feature])
        else:
            for feature in self.optional_features:
                optional_features_values.append(optional_features_dict[feature])

        return optional_features_values

class DatasetIndices(DatasetGNSS):
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

        self.solar_indices_daily, self.solar_indices_hourly = get_solar_indices(solar_indices_path)
            
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
        optional_features_values = self.get_optional_features(year, doy, sod)
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



class DatasetReorganized(DatasetGNSS):
    # NOTE: split is only passed to match the same signature as DatasetIndices
    def __init__(self, datapaths: list[str], split: str, logger: Logger, pytables: bool, solar_indices_path, optional_features = ['doy', 'year'], use_spheric_coords=False, normalize_features=False):

        self.optional_features = optional_features or []
        if type(self.optional_features) == dict:
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

        self.solar_indices_daily, self.solar_indices_hourly = get_solar_indices(solar_indices_path)
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
        sod = row['sod']
        
        optional_features_values = self.get_optional_features(year, doy, sod)
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


class DatasetSA(DatasetGNSS):
    def __init__(self, df, solar_indices_path, optional_features = ['doi', 'year'], satazi=None, use_spheric_coords=False, normalize_features=False):
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
        self.solar_indices_daily, self.solar_indices_hourly = get_solar_indices(solar_indices_path)


    def __getitem__(self, index):
        row = self.df.iloc[index].copy()
        year = float(row['year'])
        doy = float(row['doy'])
        sod = float(row['sod'])
        optional_features_values = self.get_optional_features(year, doy, sod)
        if self.satazi is None:
            row["satazi"] = 360*np.random.uniform()
        else:
            row["satazi"] = self.satazi

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


class DatasetArtificial(Dataset):
    # An adaptation of https://github.com/arrueegg/STEC_pretrained/blob/main/src/utils/data_SH.py
    def __init__(self, artificial_df, doy, year, pytables: bool, solar_indices_path, optional_features = ['doy', 'year'], use_spheric_coords=False, normalize_features=False):
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

        self.data = artificial_df
        
        
        self.use_spheric_coords = use_spheric_coords
        self.normalize_features = normalize_features

        hourly_path = solar_indices_path + "omni2_solar_indices_hourly.lst"
        labels = ["year", "doy", "hour", "kp_index", "r_index", "dst_index", "f_index"]
        self.solar_indices_hourly = pd.read_csv(hourly_path, sep='\s+', names=labels)
        # self.solar_indices = pd.read_csv(hourly_path, delim_whitespace=True, names=labels)

        daily_path = solar_indices_path + "omni2_solar_indices_daily.lst"
        self.solar_indices_daily = pd.read_csv(daily_path, sep='\s+', names=labels)
        # self.solar_indices_daily = pd.read_csv(daily_path, delim_whitespace=True, names=labels)
        self.doy = doy
        self.year = year
            
            

    def __getitem__(self, index: int) -> tuple[torch.tensor, torch.tensor]:
        """
        Loads and returns a sample from the dataset at the given index as a tensor. 
        Features are transformed as necessary. 
        """
        # iterate through all datapaths to find the file that contains our desired index

        doy = float(self.doy)
        year = float(self.year)

        row = self.data.iloc[index]
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

        x = get_only_features_from_row(row, optional_features_values, self.use_spheric_coords, self.normalize_features)
        
        return x
        
    def __len__(self):
        """
        Returns length of the dataset in use.
        """
        return len(self.data)
    