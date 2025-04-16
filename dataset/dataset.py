from torch.utils.data import Dataset
import torch
import tables
import os
import numpy as np
from numpy.typing import NDArray
from logging import Logger
import time


def get_features_from_row(row: NDArray, doy: str):
    x = torch.tensor(
            [
                row['sm_lat_ipp'],
                np.sin(2 * np.pi * row['sm_lon_ipp'] / 360),
                np.cos(2 * np.pi * row['sm_lon_ipp'] / 360),
                np.sin(2 * np.pi * row['sod'] / 86400),
                np.cos(2 * np.pi * row['sod'] / 86400), 
                np.sin(2 * np.pi * row['satazi'] / 360),
                np.cos(2 * np.pi * row['satazi'] / 360),
                row['satele'],
                float(doy)
            ],
            dtype=torch.float32
        )
    y = torch.tensor([row['stec']]) # label
    return x, y



class DatasetGNSS(Dataset):
    # An adaptation of https://github.com/arrueegg/STEC_pretrained/blob/main/src/utils/data_SH.py
    def __init__(self, datapaths: list[str], split: str, logger: Logger):
        """
        Creates an instance of DatasetGNSS. 
        
        Attributes:
            datapaths_info :    List of dictionaries. Each dictionary contains: datapath, year, doy, indices 
                (of datapoints whose stations or in the given split), current_start_point (start position in overall dataset)
                of a file in the dataset.
            stations :  list of all stations matching the given split type, encoded to bytes.
            lenght :    length of the Dataset
        """
        self.datapaths_info = []
        with open(f"dataset/{split}.list", "r") as file:
            self.stations = [line.strip().encode('utf8') for line in file]
            
        # In order to save the stations to json format, I had to map them from bytes to strings.
        # We now remap them to bytes
        # TODO: Find a better way to do this
        current_start_point = 0
        self.data = []


        for datapath in datapaths:            
            if not(os.path.isfile(datapath)):
                # delete this day from datapaths, as there is no data available for that day 
                logger.info(f"Skipping {datapath}")
                continue
            
            # extract year and doy from datapath
            year = datapath.split('/')[-3]
            doy = datapath.split('/')[-2]
            
            file = tables.open_file(datapath, mode='r', driver='H5FD_SEC2')
            data = file.get_node(f"/{year}/{doy}/all_data")

            indices = self._get_indices(data) #TODO: need to be sure that the given file actually exists
            # add add additional features to data
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
            
        self.length = current_start_point
            
    def _get_indices(self, data) -> NDArray:
        """
        Returns the indices of all datapoints whose stations is in the current datasplit.
        """
        # select indices of datapoints whose stations that are in the current split
        mask_for_split = np.isin(data.col('station'), self.stations) 

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

        # year = datapath_info['year']
        doy = datapath_info['doy']
        indices = datapath_info['indices']
        start_point = datapath_info['start_point']

        data = datapath_info['data']

        start = time.time()    
        row = data[indices[index - start_point]]
        if np.random.rand() > 0.9:
            print(f"Access data by index time: {time.time() - start}s")
        
        # save the data in a tensor and creat
        x, y = get_features_from_row(row, doy)
        
        return x, y
        
    def __len__(self):
        """
        Returns length of the dataset in use.
        """
        return self.length
    
    def __del__(self):
        for datapath_info in self.datapaths_info:
            datapath_info["file"].close()
    
    
if __name__ == "__main__":
    datapaths = [f"/cluster/work/igp_psr/arrueegg/GNSS_STEC_DB/2024/{str(doi).zfill(3)}/ccl_2024{str(doi).zfill(3)}_30_5.h5" for doi in range(1, 3)]
    train_dataset = DatasetGNSS(datapaths, "train")
    print(train_dataset[10])
    train_dataset.__del__()
