"""Comprises all the classes used to load our datasets."""

from torch.utils.data import Dataset
import torch
import tables
import os
import numpy as np
from numpy.typing import NDArray
from logging import Logger
import logging
import h5py
import os
import pandas as pd
from typing import Any


def get_solar_indices(solar_indices_path: str) -> tuple[dict, dict]:
    """
    Fetches the solar index data at `solar_indices_path` and creates a dictionary for daily and one for hourly solar index data.
    For the daily solar indices, the keys are tuples ({year}, {doy}).
    For the hourly solar indices, the keys are tuples ({year}, {doy}, {hour}).
    """

    # fetch hourly solar index data
    hourly_path = solar_indices_path + "omni2_solar_indices_hourly.lst"
    labels = ["year", "doy", "hour", "kp_index", "r_index", "dst_index", "f_index"]
    solar_indices_hourly = pd.read_csv(hourly_path, sep=r"\s+", names=labels)
    last_doy = {2020: 366, 2021: 365, 2022: 365, 2023: 365, 2024: 366, 2025: 365}
    solar_indices_lookup_hourly = {
        (row["year"], row["doy"], row["hour"]): row[["kp_index", "r_index", "dst_index", "f_index"]]
        .to_numpy()
        .reshape(-1)
        for _, row in solar_indices_hourly.iterrows()
    }

    # for each day, add a row with hour=24 with data of midnight of the next day
    for _, row in solar_indices_hourly.iterrows():
        if row["hour"] == 23:
            # if our current doy is the last day of the year, need to fetch data from the first day of the next year
            if last_doy[row["year"]] == row["doy"]:
                if row["year"] < 2024:  # don't have solar index data for 2025
                    solar_indices_lookup_hourly.update(
                        {(row["year"], row["doy"], 24): solar_indices_lookup_hourly[(row["year"] + 1, 1, 0)]}
                    )
                else:
                    solar_indices_lookup_hourly.update(
                        {(row["year"], row["doy"], 24): solar_indices_lookup_hourly[(row["year"], row["doy"], 23)]}
                    )
            else:  # add data of midnight of the next day
                solar_indices_lookup_hourly.update(
                    {(row["year"], row["doy"], 24): solar_indices_lookup_hourly[(row["year"], row["doy"] + 1, 0)]}
                )

    # fetch daily solar index data
    daily_path = solar_indices_path + "omni2_solar_indices_daily.lst"
    solar_indices_daily = pd.read_csv(daily_path, sep=r"\s+", names=labels)
    solar_indices_lookup_daily = {
        (row["year"], row["doy"]): row[["kp_index", "r_index", "dst_index", "f_index"]].to_numpy().reshape(-1)
        for _, row in solar_indices_daily.iterrows()
    }

    return solar_indices_lookup_daily, solar_indices_lookup_hourly


def get_features_from_row(
    row: NDArray, optional_features_values: list, use_spheric_coords: bool = False, normalize_features: bool = False
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Given a row, returns featurs x and label y
    """
    x = get_only_features_from_row(row, optional_features_values, use_spheric_coords, normalize_features)
    y = torch.tensor([row["stec"]])  # label
    return x, y


def get_only_features_from_row(
    row: NDArray, optional_features_values: list, use_spheric_coords: bool = False, normalize_features: bool = False
) -> torch.Tensor:
    """
    Returns features from the given row.

    Args:
        row: a row of the dataset
        optional_features_values: list of the values of all desired optional features
        use_spheric_coords (bool): indicates whether to use spherical coordinates
        normalize_features (bool): indicates whether to normalize features
    """
    if use_spheric_coords:
        angle_coords = [
            np.cos(2 * np.pi * row["satele"] / 360) * np.cos(2 * np.pi * row["satazi"] / 360),
            np.cos(2 * np.pi * row["satele"] / 360) * np.sin(2 * np.pi * row["satazi"] / 360),
            np.sin(2 * np.pi * row["satele"] / 360),
        ]

    else:
        angle_coords = [
            np.sin(2 * np.pi * row["satazi"] / 360),
            np.cos(2 * np.pi * row["satazi"] / 360),
            row["satele"] if not normalize_features else row["satele"] / 45 - 1.0,
        ]
    x = torch.tensor(
        [
            row["sm_lat_ipp"] if not normalize_features else row["sm_lat_ipp"] / 90,
            np.sin(2 * np.pi * row["sm_lon_ipp"] / 360),
            np.cos(2 * np.pi * row["sm_lon_ipp"] / 360),
            np.sin(2 * np.pi * row["sod"] / 86400),
            np.cos(2 * np.pi * row["sod"] / 86400),
        ]
        + angle_coords
        + optional_features_values,
        dtype=torch.float32,
    )
    return x


def get_file_and_data(filepath: str, nodepath: str, pytables: bool) -> tuple[Any, Any]:
    """
    Opens the file at `filepath` and fetches the corresponding data at `nodepath`.
    If pytables is True, uses pytables to handle file and data.
    """
    if pytables:
        file = tables.open_file(filepath, mode="r", driver="H5FD_SEC2")
        data = file.get_node(nodepath)
    else:
        file = h5py.File(filepath, "r")
        data = file[nodepath][:]

    return file, data


class DatasetGNSS(Dataset):
    """
    Base class for all three dataset classes that provides a common class method to handle optional features.
    """

    solar_indices_daily: dict
    solar_indices_hourly: dict
    optional_features: list | dict
    normalize_features: bool

    def get_optional_features(self, year: int, doy: int, sod: int) -> list:
        """
        Returns a list of values of all optional features that are requested, as per self.optional features of the DatasetGNSS instance.
        """
        optional_features_dict = {
            "doy": doy if not self.normalize_features else doy / 183 - 1.0,
            "year": year,
        }

        # NOTE: Some of our older models had a two-tier format of optional_features.
        # For the sake of backwards comptatibility we make this distinction here
        if isinstance(self.optional_features, list):
            optional_features_names = self.optional_features
        else:
            optional_features_names = self.optional_features["delayed"]
            assert len(self.optional_features["initial"]) == 0

        # fetch daily solar index data, if requested
        if np.any(["daily" in f for f in optional_features_names]):
            daily_solar_indices = self.solar_indices_daily[(int(year), int(doy))]
            optional_features_dict.update(
                {
                    "kp_index_daily": daily_solar_indices[0],
                    "r_index_daily": daily_solar_indices[1],
                    "dst_index_daily": daily_solar_indices[2],
                    "f_index_daily": daily_solar_indices[3],
                }
            )

        # fetch hourly solar index data, if requested
        if np.any(["hourly" in f for f in optional_features_names]):
            hour = np.round(sod / 3600).astype(int)
            hourly_solar_indices = self.solar_indices_hourly[(int(year), int(doy), int(hour))]
            optional_features_dict.update(
                {
                    "kp_index_hourly": hourly_solar_indices[0],
                    "r_index_hourly": hourly_solar_indices[1],
                    "dst_index_hourly": hourly_solar_indices[2],
                    "f_index_hourly": hourly_solar_indices[3],
                }
            )

        # create list of values of all optional features that were requested

        optional_features_values = []
        # NOTE: Some of our older models had a two-tier format of optional_features.
        # For the sake of backwards comptatibility we make this distinction here
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
    """
    Dataset class that is used to load the data when it's in the raw dataset format, i.e. without subsampling or reogranization
    Since train, val and test data are mixed, an additional index structure is required to serparate by split.

    Attributes:
        optional_features (list[str]): List of optional features
        use_spheric_coords (bool): indicates whether to use spherical coordinates
        normalize_features (bool): indicates whether to normalize features
        datapaths_info (list[dict]): List of dictionaries. Each dictionary contains: datapath, year, doy, indices
            (of datapoints whose stations or in the given split), current_start_point (start position in overall dataset)
            of a file in the dataset.
        stations: list of all stations matching the given split type, encoded to bytes.
        solar_indices_daily: dictionary of daily solar index data, with keys ({year}, {doy})
        solar_indices_hourly: dictionary of hourly solar index data, with keys ({year}, {doy}, {hour})
        lenght (int): length of the Dataset
    """

    def __init__(
        self,
        datapaths: list[str],
        split: str,
        logger: Logger,
        pytables: bool,
        solar_indices_path: str,
        optional_features: list[str] = ["doy", "year"],
        use_spheric_coords: bool = False,
        normalize_features: bool = False,
    ) -> None:
        """
        Creates an instance of DatasetIndices.

        Args:
            datapaths (list[str]): list of paths to the location of all datasets that should be incorporated
            split (str): determines which datasplit (train, val, test) to process
            logger (Logger): for loggin status reports
            pytables (bool): indicates whether to use pytables
            solar_indices_path (str): path to the folder the solar index are stored in
            optional_features (list[str]): List of desired optional features
            use_spheric_coords (bool): indicates whether to use spherical coordinates
            normalize_features (bool): indicates whether to normalize features
        """

        self.optional_features = optional_features or []
        # NOTE: Some of our older models had a two-tier format of optional_features.
        # For the sake of backwards comptatibility we make this distinction here
        if type(self.optional_features) == dict:
            for tier in self.optional_features:
                self.optional_features[tier] = self.optional_features[tier] or []

        self.use_spheric_coords = use_spheric_coords
        self.normalize_features = normalize_features

        self.datapaths_info = []
        with open(f"dataset/{split}.list", "r") as file:
            self.stations = [line.strip().encode("utf8") for line in file]  # create list of stations

        current_start_point = 0

        # iterating through all datapaths
        for datapath in datapaths:
            if not (os.path.isfile(datapath)):
                logger.info(f"Skipping {datapath}")
                continue

            # extract year and doy from datapath
            year = datapath.split("/")[-3]
            doy = datapath.split("/")[-2]

            # fetch data and indices
            file, data = get_file_and_data(datapath, f"/{year}/{doy}/all_data", pytables)
            indices = self._get_indices(data, pytables)

            self.datapaths_info.append(
                {
                    "datapath": datapath,
                    "file": file,
                    "data": data,
                    "year": year,
                    "doy": doy,
                    "indices": indices,
                    "start_point": current_start_point,
                }
            )
            current_start_point += len(indices)
            # logger.info(f"Completed {datapath}")

        # fetch solar indices data
        self.solar_indices_daily, self.solar_indices_hourly = get_solar_indices(solar_indices_path)

        self.length = current_start_point

    def _get_indices(self, data, pytables: bool) -> NDArray:
        """
        Returns the indices of all datapoints whose station is in the current datasplit.
        """
        # select indices of datapoints whose stations that are in the current split
        if pytables:
            mask_for_split = np.isin(data.col("station"), self.stations)
        else:
            mask_for_split = np.isin(data["station"], self.stations)

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
            if next_datapath_info["start_point"] > index:
                datapath_info = curr_datapath_info
                break
        else:
            datapath_info = self.datapaths_info[-1]

        doy = float(datapath_info["doy"])
        year = float(datapath_info["year"])
        indices = datapath_info["indices"]
        start_point = datapath_info["start_point"]

        data = datapath_info["data"]
        row = data[indices[index - start_point]]

        sod = row["sod"]
        optional_features_values = self.get_optional_features(year, doy, sod)
        x, y = get_features_from_row(row, optional_features_values, self.use_spheric_coords, self.normalize_features)

        return x, y

    def __len__(self) -> int:
        """
        Returns length of the dataset.
        """
        return self.length

    def __del__(self) -> None:
        """
        Closes all open files.
        """
        for datapath_info in self.datapaths_info:
            datapath_info["file"].close()


class DatasetReorganized(DatasetGNSS):
    """
    Dataset class that is used to load the reorganized data.

    Attributes:
        optional_features (list[str]): List of optional features
        use_spheric_coords (bool): indicates whether to use spherical coordinates
        normalize_features (bool): indicates whether to normalize features
        datapaths_info (list[dict]): List of dictionaries. Each dictionary contains: datapath, year, doy, indices
            (of datapoints whose stations or in the given split), current_start_point (start position in overall dataset)
            of a file in the dataset.
        solar_indices_daily: dictionary of daily solar index data, with keys ({year}, {doy})
        solar_indices_hourly: dictionary of hourly solar index data, with keys ({year}, {doy}, {hour})
        lenght (int): length of the Dataset
    """

    def __init__(
        self,
        datapaths: list[str],
        split: str,
        logger: Logger,
        pytables: bool,
        solar_indices_path: str,
        optional_features: list[str] = ["doy", "year"],
        use_spheric_coords: bool = False,
        normalize_features: bool = False,
    ) -> None:
        """
        Creates an instance of DatasetReorganized.

        Args:
            datapaths (list[str]): list of paths to the location of all datasets that should be incorporated
            split (str): split is only passed to match the same signature as DatasetIndices
            logger (Logger): for loggin status reports
            pytables (bool): indicates whether to use pytables
            solar_indices_path (str): path to the folder the solar index are stored in
            optional_features (list[str]): List of desired optional features
            use_spheric_coords (bool): indicates whether to use spherical coordinates
            normalize_features (bool): indicates whether to normalize features
        """

        self.optional_features = optional_features or []
        # NOTE: Some of our older models had a two-tier format of optional_features.
        # For the sake of backwards comptatibility we make this distinction here
        if type(self.optional_features) == dict:
            for tier in self.optional_features:
                self.optional_features[tier] = self.optional_features[tier] or []

        self.use_spheric_coords = use_spheric_coords
        self.normalize_features = normalize_features

        current_start_point = 0
        self.datapaths_info = []

        # iterating through all datapaths
        for datapath in datapaths:
            if not (os.path.isfile(datapath)):
                logger.info(f"Skipping {datapath}")
                continue

            # extract year and doy from datapath
            filename = datapath.split("/")[-1]
            year = filename.split("-")[0]
            month = datapath.split("-")[1]

            file, data = get_file_and_data(datapath, "/all_data", pytables)

            self.datapaths_info.append(
                {
                    "datapath": datapath,
                    "file": file,
                    "data": data,
                    "start_point": current_start_point,
                    "year": year,
                    "month": month,
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
            if next_datapath_info["start_point"] > index:
                datapath_info = curr_datapath_info
                break
        else:
            datapath_info = self.datapaths_info[-1]

        start_point = datapath_info["start_point"]
        data = datapath_info["data"]

        year = float(datapath_info["year"])
        row = data[index - start_point]

        # NOTE: When reorganizing the data it got annoying to create a new column for day but overwriting an
        # existing column was straightforward, so we saved the doy in gphase
        doy = row["gfphase"]
        sod = row["sod"]

        optional_features_values = self.get_optional_features(year, doy, sod)
        x, y = get_features_from_row(row, optional_features_values, self.use_spheric_coords, self.normalize_features)

        return x, y

    def __len__(self) -> int:
        """
        Returns length of the dataset.
        """
        return self.length

    def __del__(self) -> None:
        """
        Closes all open files.
        """
        for datapath_info in self.datapaths_info:
            datapath_info["file"].close()


class DatasetSA(DatasetGNSS):
    """
    Dataset class that is used to load the data reorganized data.

    Attributes:
        df (DataFrame): DataFrame that contains the data
        satazi: Since satellite altimetry data has no azimuth value, supply a fixed value or None for a different random azimuth value in each __getitem__ cal
        optional_features (list[str]): List of optional features
        use_spheric_coords (bool): indicates whether to use spherical coordinates
        normalize_features (bool): indicates whether to normalize features
        solar_indices_daily: dictionary of daily solar index data, with keys ({year}, {doy})
        solar_indices_hourly: dictionary of hourly solar index data, with keys ({year}, {doy}, {hour})
    """

    def __init__(
        self,
        df,
        solar_indices_path: str,
        optional_features: list[str] = ["doy", "year"],
        satazi: float | None = None,
        use_spheric_coords: bool = False,
        normalize_features: bool = False,
    ) -> None:
        """
        Creates an instance of DatasetReorganized.

        Args:
            df (DataFrame): DataFrame that contains the satellite altimetry data
            solar_indices_path (str): path to the folder the solar index are stored in
            satazi: Since satellite altimetry data has no azimuth value, supply a fixed value or None for a different random azimuth value in each __getitem__ cal
            optional_features (list[str]): List of desired optional features
            use_spheric_coords (bool): indicates whether to use spherical coordinates
            normalize_features (bool): indicates whether to normalize features
        """

        df = df.rename(columns={"sm_lat": "sm_lat_ipp", "sm_lon": "sm_lon_ipp"})
        print(df.columns)
        df["time"] = pd.to_datetime(df["time"])
        df["sod"] = df["time"].dt.second + 60 * df["time"].dt.minute + 3600 * df["time"].dt.hour
        df["doy"] = df["time"].dt.dayofyear
        df["year"] = df["time"].dt.year
        df["satele"] = 90
        df["stec"] = df["vtec"]
        self.df = df
        self.satazi = satazi
        if optional_features is None:
            self.optional_features = []
        else:
            self.optional_features = optional_features
        self.use_spheric_coords = use_spheric_coords
        self.normalize_features = normalize_features
        self.solar_indices_daily, self.solar_indices_hourly = get_solar_indices(solar_indices_path)

    def __getitem__(self, index: int):
        """
        Loads and returns a sample from the dataset at the given index as a tensor.
        Features are transformed as necessary.
        """
        row = self.df.iloc[index].copy()
        year = float(row["year"])
        doy = float(row["doy"])
        sod = float(row["sod"])
        optional_features_values = self.get_optional_features(year, doy, sod)
        if self.satazi is None:
            row["satazi"] = 360 * np.random.uniform()
        else:
            row["satazi"] = self.satazi

        x, y = get_features_from_row(row, optional_features_values, self.use_spheric_coords, self.normalize_features)
        return x, y

    def __len__(self) -> int:
        """
        Returns length of the dataset in use.
        """
        return self.df.shape[0]


if __name__ == "__main__":
    logger = logging.getLogger(__name__)

    import yaml

    # config_path = "config/pretraining_config.yaml"
    config_path = "config/training_config.yaml"
    with open(config_path, "r") as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    datapath = "/cluster/work/igp_psr/arrueegg/GNSS_STEC_DB/"
    dslab_path = "/cluster/work/igp_psr/dslab_FS25_data_and_weights/"
    # datapaths_train = [dslab_path + f"reorganized_data_2/2023-{i}-train.h5" for i in range(1, 12)]
    # train_dataset = DatasetReorganized(datapaths_train, 'train', logger, True, config['solar_indices_path'], optional_features = config['optional_features'])
    datapaths_train = [datapath + f"2024/{str(183+i).zfill(3)}/ccl_2024{str(183+i).zfill(3)}_30_5.h5" for i in range(2)]
    train_dataset = DatasetIndices(
        datapaths_train,
        "train",
        logger,
        True,
        config["solar_indices_path"],
        optional_features=config["optional_features"],
    )
    train_dataset[10]
    train_dataset.__del__()
