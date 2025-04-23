import os
import tables
import numpy as np
from tqdm import tqdm

np.random.seed(10)


dslab_path = "/cluster/work/igp_psr/dslab_FS25_data_and_weights/"
output_path = dslab_path + "reorganized_data/"


class TableDescription(tables.IsDescription):
    station = tables.StringCol(itemsize=4, shape=(), dflt=np.bytes_(''), pos=0)
    sat = tables.StringCol(itemsize=3, shape=(), dflt=np.bytes_(''), pos=1)
    stec = tables.Float32Col(shape=(), dflt=np.float32(0.0), pos=2)
    vtec = tables.Float32Col(shape=(), dflt=np.float32(0.0), pos=3)
    # vtec_stddev = tables.Float32Col(shape=(), dflt=np.float32(0.0), pos=4)
    # satres = tables.Float32Col(shape=(), dflt=np.float32(0.0), pos=5)
    satele = tables.Float32Col(shape=(), dflt=np.float32(0.0), pos=4)
    satazi = tables.Float32Col(shape=(), dflt=np.float32(0.0), pos=5)
    # dcdbs = tables.Float32Col(shape=(), dflt=np.float32(0.0), pos=8)
    # dcdbrr = tables.Float32Col(shape=(), dflt=np.float32(0.0), pos=9)
    lon_ipp = tables.Float32Col(shape=(), dflt=np.float32(0.0), pos=6)
    lat_ipp = tables.Float32Col(shape=(), dflt=np.float32(0.0), pos=7)
    sm_lat_ipp = tables.Float32Col(shape=(), dflt=np.float32(0.0), pos=8)
    sm_lon_ipp = tables.Float32Col(shape=(), dflt=np.float32(0.0), pos=9)
    sod = tables.Float32Col(shape=(), dflt=np.float32(0.0), pos=10)
    lat_sta = tables.Float32Col(shape=(), dflt=np.float32(0.0), pos=11)
    lon_sta = tables.Float32Col(shape=(), dflt=np.float32(0.0), pos=12)
    sm_lat_sta = tables.Float32Col(shape=(), dflt=np.float32(0.0), pos=13)
    sm_lon_sta = tables.Float32Col(shape=(), dflt=np.float32(0.0), pos=14)
    # doy = tables.Float32Col(shape=(), dflt=np.float32(0.0), pos=15)
    # slipc = tables.Float32Col(shape=(), dflt=np.float32(0.0), pos=19)
    gfphase = tables.Float32Col(shape=(), dflt=np.float32(0.0), pos=15)


def get_expected_rows(group, subsampling_ratio=0.05):
    rows_per_raw_day = 20e6
    row_fraction = 0.7 if group == "train" else (0.2 if group == "test" else 0.1)
    month_max_days = 31
    expectedrows = rows_per_raw_day*row_fraction*subsampling_ratio*month_max_days
    return expectedrows


if __name__ == "__main__":
    os.makedirs(output_path, exist_ok=True)
    subsampling_ratio = 0.05
    year = 2023
    leap_years = [2020, 2024]
    day_count_per_month = [31,28,31,30,31,30,31,31,30,31,30,31]
    if year in leap_years:
        day_count_per_month[1] = 29
    month_indices = np.cumsum(day_count_per_month).tolist()
        
    datapaths_per_month = []
    for last_index, index in zip([0] + month_indices[:-1], month_indices):
        datapaths_per_month.append(
            [f"/cluster/work/igp_psr/arrueegg/GNSS_STEC_DB/{year}/{str(doi + 1).zfill(3)}/ccl_{year}{str(doi + 1).zfill(3)}_30_5.h5" for doi in range(last_index, index)]
        )
        
    
    
    for month_idx, datapaths in enumerate(datapaths_per_month):
        month = month_idx + 1
        print(f"month: {month}")
        for group in ["train", "val", "test"]:
            with open(f"dataset/{group}.list", "r") as file:
                stations = [line.strip().encode('utf8') for line in file]
            
            
            month_output_path = output_path + str(year) + '-' + str(month) + "-" + group + ".h5"
            if os.path.exists(month_output_path):
                print(f"skipping {month_output_path}")
                continue

            try:
                out_h5 = tables.open_file(month_output_path, mode='w')
                expectedrows = get_expected_rows(group, subsampling_ratio)
                out_table = out_h5.create_table('/', 'all_data', TableDescription, expectedrows=expectedrows)
                for datapath in tqdm(datapaths):
                    try:
                        in_h5 = tables.open_file(datapath, mode='r')
                        doy = datapath.split('/')[-2]
                        data = in_h5.get_node(f"/{year}/{doy}/all_data")
                        seconds = [int(30 / subsampling_ratio)*i + 30*np.random.randint(0, 20) for i in range(int(2880*subsampling_ratio))]
                        mask_subsampling = np.isin(data.col('sod'), seconds)
                        mask_stations = np.isin(data.col('station'), stations)
                        filtered_data = data[mask_subsampling*mask_stations]
                        filtered_data['gfphase'] = doy
                        filtered_data = filtered_data[list(TableDescription.__dict__['columns'].keys())]  
                        
                        filtered_data = filtered_data.astype([
                            ('station', 'S4'),
                            ('sat', 'S3'),
                            ('stec', 'f4'),
                            ('vtec', 'f4'),
                            ('satele', 'f4'),
                            ('satazi', 'f4'),
                            ('lon_ipp', 'f4'),
                            ('lat_ipp', 'f4'),
                            ('sm_lat_ipp', 'f4'),
                            ('sm_lon_ipp', 'f4'),
                            ('sod', 'f4'),
                            ('lat_sta', 'f4'),
                            ('lon_sta', 'f4'),
                            ('sm_lat_sta', 'f4'),
                            ('sm_lon_sta', 'f4'),
                            ('gfphase', 'f4')
                        ])  
                        out_table.append(filtered_data)
                        out_table.flush()
                    
                    finally:
                        in_h5.close()
                    
            finally:
                out_h5.close()
        
    