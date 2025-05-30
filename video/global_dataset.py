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
    solar_indices_hourly = pd.read_csv(hourly_path, sep="\s+", names=labels)
    last_doy = {2020: 366, 2021: 365, 2022: 365, 2023: 365, 2024: 366, 2025: 365}
    solar_indices_lookup_hourly = {
        (row["year"], row["doy"], row["hour"]): row[["kp_index", "r_index", "dst_index", "f_index"]]
        .to_numpy()
        .reshape(-1)
        for _, row in solar_indices_hourly.iterrows()
    }
    for _, row in solar_indices_hourly.iterrows():
        if row["hour"] == 23:
            if last_doy[row["year"]] == row["doy"]:
                if row["year"] < 2024:
                    solar_indices_lookup_hourly.update(
                        {(row["year"], row["doy"], 24): solar_indices_lookup_hourly[(row["year"] + 1, 1, 0)]}
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
    solar_indices_daily = pd.read_csv(daily_path, sep="\s+", names=labels)
    solar_indices_lookup_daily = {
        (row["year"], row["doy"]): row[["kp_index", "r_index", "dst_index", "f_index"]].to_numpy().reshape(-1)
        for _, row in solar_indices_daily.iterrows()
    }

    return solar_indices_lookup_daily, solar_indices_lookup_hourly


def get_features_from_row(
    row: NDArray, optional_features_values: list, use_spheric_coords: bool = False, normalize_features: bool = False
):
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
    y = torch.tensor([row["stec"]])  # label
    return x, y


class DatasetArtificial(Dataset):
    # An adaptation of https://github.com/arrueegg/STEC_pretrained/blob/main/src/utils/data_SH.py
    def __init__(
        self,
        artificial_df,
        doy,
        year,
        pytables: bool,
        solar_indices_path,
        optional_features=["doy", "year"],
        use_spheric_coords=False,
        normalize_features=False,
    ):
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
        self.solar_indices_hourly = pd.read_csv(hourly_path, sep="\s+", names=labels)
        # self.solar_indices = pd.read_csv(hourly_path, delim_whitespace=True, names=labels)

        daily_path = solar_indices_path + "omni2_solar_indices_daily.lst"
        self.solar_indices_daily = pd.read_csv(daily_path, sep="\s+", names=labels)
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
        sod = row["sod"]

        hourly_solar_indices = self.solar_indices_hourly[(int(year), int(doy))]
        daily_solar_indices = self.solar_indices_daily[(int(year), int(doy))]

        optional_features_dict = {
            "kp_index_hourly": hourly_solar_indices[0],
            "r_index_hourly": hourly_solar_indices[1],
            "dst_index_hourly": hourly_solar_indices[2],
            "f_index_hourly": hourly_solar_indices[3],
            "doy": doy if not self.normalize_features else doy / 183 - 1.0,
            "year": year,
            "kp_index_daily": daily_solar_indices[0],
            "r_index_daily": daily_solar_indices[1],
            "dst_index_daily": daily_solar_indices[2],
            "f_index_daily": daily_solar_indices[3],
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
