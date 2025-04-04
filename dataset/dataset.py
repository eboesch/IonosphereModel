from torch.utils.data import Dataset
import torch
import tables
import json
import numpy as np
from numpy.typing import NDArray


class DatasetGNSS(Dataset):
    # Based in https://github.com/arrueegg/STEC_pretrained/blob/main/src/utils/data_SH.py
    def __init__(self, datapaths: list[str], split: int, splits_file: str):
        self.datapaths_info = []
        self.current_length = 0
        with open(splits_file, 'r') as f:
            splits_dict = json.load(f)
            
        # In order to save the stations to json format, I had to map them from bytes to strings.
        # We now remap them to bytes
        # TODO: Find a better way to do this
        self.stations = [st.encode() for st, spl in splits_dict.items() if spl == split]
        current_start_point = 0
    
        for datapath in datapaths:
            year = datapath.split('/')[-3]
            doy = datapath.split('/')[-2] 
            indices = self._get_indices(datapath, year, doy)
            self.datapaths_info.append(
                {
                    "datapath": datapath,
                    "year": year,
                    "doy": doy,
                    "indices": indices,
                    "start_point": current_start_point
                }
            )
            current_start_point += len(indices)
            
        self.length = current_start_point
            
    def _get_indices(self, datapath: str, year: str, doy: str) -> NDArray:
        with tables.open_file(datapath, mode='r', driver='H5FD_SEC2') as file:
            data = file.get_node(f"/{year}/{doy}/all_data")
            mask_for_split = np.isin(data.col('station'), self.stations)
            del data

        indices = np.arange(0, len(mask_for_split), 1)[mask_for_split]
        del mask_for_split
        return indices
    
    def __getitem__(self, index: int) -> tuple[torch.tensor, torch.tensor]:
        
        for curr_datapath_info, next_datapath_info in zip(self.datapaths_info[:-1], self.datapaths_info[1:]):
            if next_datapath_info['start_point'] > index:
                datapath_info = curr_datapath_info
                break
    
        else:
            datapath_info = self.datapaths_info[-1]

        with tables.open_file(datapath_info['datapath'], mode='r', driver='H5FD_SEC2') as file:
            year = datapath_info['year']
            doy = datapath_info['doy']
            indices = datapath_info['indices']
            start_point = datapath_info['start_point']
            data = file.get_node(f"/{year}/{doy}/all_data")
            row = data[indices[index - start_point]]
            del data
        
        x = torch.tensor(
            [
                row['sm_lat_ipp'],
                np.sin(2 * np.pi * row['sm_lon_ipp'] / 360),
                np.cos(2 * np.pi * row['sm_lon_ipp'] / 360),
                np.sin(2 * np.pi * row['sod'] / 86400),
                np.cos(2 * np.pi * row['sod'] / 86400), 
                np.sin(2 * np.pi * row['satazi'] / 360),
                np.cos(2 * np.pi * row['satazi'] / 360),
                np.cos(2 * np.pi * row['satazi'] / 360),
                row['satele'],
                float(doy)
            ],
            dtype=torch.float32
        ) 
        
        y = torch.tensor([row['stec']])
        
        return x, y
        
    def __len__(self):
        return self.length
    
    
if __name__ == "__main__":
    datapaths = [f"/cluster/work/igp_psr/arrueegg/GNSS_STEC_DB/2024/{doi}/ccl_2024{doi}_30_5.h5" for doi in range(300, 302)]
    splits_file = "/cluster/work/igp_psr/dslab_FS25_data_and_weights/split_exp.json"
    train_dataset = DatasetGNSS(datapaths, 0, splits_file)
    train_dataset[10]