import os
import tables
import numpy as np
import pandas as pd
from tqdm import tqdm
import shutil
import yaml
from joblib import Parallel, delayed

# srun --ntasks=1 --cpus-per-task=6 --mem-per-cpu=4096 -t 600 -o file.out -e file.err python dataset/reorganize_data.py &

class SolarTableDescription(tables.IsDescription):
    """
    Table Schema if either daily or hourly solar indices are requested
    """
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
    # slipc = tables.Float32Col(shape=(), dflt=np.float32(0.0), pos=19)
    # gfphase = tables.Float32Col(shape=(), dflt=np.float32(0.0), pos=20)
    doy = tables.Float32Col(shape=(), dflt=np.float32(0.0), pos=15)
    kp_index = tables.Float32Col(shape=(), dflt=np.float32(0.0), pos=16)
    r_index = tables.Float32Col(shape=(), dflt=np.float32(0.0), pos=17)
    dst_index = tables.Float32Col(shape=(), dflt=np.float32(0.0), pos=18)
    f_index = tables.Float32Col(shape=(), dflt=np.float32(0.0), pos=19)

class PlainTableDescription(tables.IsDescription):
    """
    Table Schema if no solar indices are requested
    """
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
    # slipc = tables.Float32Col(shape=(), dflt=np.float32(0.0), pos=19)
    # gfphase = tables.Float32Col(shape=(), dflt=np.float32(0.0), pos=20)
    doy = tables.Float32Col(shape=(), dflt=np.float32(0.0), pos=15)


def get_expected_rows(group, subsampling_ratio):
    """
    Returns the expected number of rows of the table, based on the subsampling ratio and group (train, test or val)
    """
    rows_per_raw_day = 20e6
    row_fraction = 0.7 if group == "train" else (0.2 if group == "test" else 0.1)
    month_max_days = 31
    expectedrows = rows_per_raw_day*row_fraction*subsampling_ratio*month_max_days
    return expectedrows


def get_daily_solar_indices(year, doy, datapath):
    """
    Returns an array of the daily average solar indices of the given day in the given year.
    """
    path = datapath + "omni2_solar_indices_daily.lst"
    labels = ["year", "doy", "hour", "kp_index", "r_index", "dst_index", "f_index"]
    # df = pd.read_csv(path, sep='\s+', names=labels)
    df = pd.read_csv(path, delim_whitespace=True, names=labels)
    row = df.loc[(df["year"] == year) & (df["doy"] == doy)] # filter for the correct row
    solar_indices = row[["kp_index", "r_index", "dst_index", "f_index"]].to_numpy().reshape(-1) # extract the desired indices 
    return solar_indices

def get_hourly_solar_indices(year, doy, datapath):
    """
    Returns a numpy array of the hourly solar indices of the given day in the given year.
    """
    path = datapath + "omni2_solar_indices_hourly.lst"
    labels = ["year", "doy", "hour", "kp_index", "r_index", "dst_index", "f_index"]
    # df = pd.read_csv(path, sep='\s+', names=labels)
    df = pd.read_csv(path, delim_whitespace=True, names=labels)
    filtered_df = df.loc[(df["year"] == year) & (df["doy"] == doy)]
    filtered_df = filtered_df.drop(columns=['year', 'doy']).to_numpy()
    return filtered_df

def reorganize_month(year, month, datapaths, output_path, subsampling_ratio, solar_indices_mode, solar_indices_path):
    """
    Subsamples the data of the corresponding month and year at the given datapaths and saves it at the indicated location.
    """
    for group in ["train", "val", "test"]:
        with open(f"dataset/{group}.list", "r") as file:
            stations = [line.strip().encode('utf8') for line in file]
        
        month_output_path = output_path + str(year) + '-' + str(month) + "-" + group + ".h5"
        if os.path.exists(month_output_path):
            print(f"Skipping {month_output_path}")
            continue
        
        try:
            print(f"Attempting {month_output_path}")
            # create empty table
            out_h5 = tables.open_file(month_output_path, mode='w')
            expectedrows = get_expected_rows(group, subsampling_ratio)
            if solar_indices_mode == "none":
                TableDescription = PlainTableDescription
            else:
                TableDescription = SolarTableDescription

            out_table = out_h5.create_table('/', 'all_data', TableDescription, expectedrows=expectedrows)
            for datapath in tqdm(datapaths):
                if not os.path.exists(datapath):
                    # if a file doesn't exist, skip it
                    continue

                try:
                    in_h5 = tables.open_file(datapath, mode='r')
                    doy = datapath.split('/')[-2]
                    data = in_h5.get_node(f"/{year}/{doy}/all_data")

                    # subsample based on  second of day. to ensure good coverage, add random perturbation to the sod we want to keep
                    np.random.seed(10 + month + int(doy) + year)
                    seconds = [int(30 / subsampling_ratio)*i + 30*np.random.randint(0, 20) for i in range(int(2880*subsampling_ratio))]
                    mask_subsampling = np.isin(data.col('sod'), seconds)
                    mask_stations = np.isin(data.col('station'), stations)
                    filtered_data = data[mask_subsampling*mask_stations]

                    
                    if solar_indices_mode == "none":
                        new_dtype = np.dtype(filtered_data.dtype.descr + [('doy', 'f4')])
                    else:
                        new_dtype = np.dtype(filtered_data.dtype.descr + [('doy', 'f4'), ('kp_index', 'f4'), ('r_index', 'f4'), ('dst_index', 'f4'), ('f_index', 'f4')])
                    
                    extended_data = np.empty(filtered_data.shape, dtype=new_dtype)
                    for name in filtered_data.dtype.names:
                        extended_data[name] = filtered_data[name]
                    extended_data['doy'] = doy

                    if solar_indices_mode == "daily":
                        solar_indices = get_daily_solar_indices(year, int(doy), solar_indices_path)
                        extended_data['kp_index'] = solar_indices[0]
                        extended_data['r_index'] = solar_indices[1]
                        extended_data['dst_index'] = solar_indices[2]
                        extended_data['f_index'] = solar_indices[3]
                    elif solar_indices_mode == "hourly":
                        solar_indices = get_hourly_solar_indices(year, int(doy), solar_indices_path)
                        
                        sod = extended_data['sod']
                        hour_indices = np.round(sod/3600).astype(int)
                        matched_solar_indices = solar_indices[hour_indices]

                        
                        extended_data['kp_index'] = matched_solar_indices[:,1]
                        extended_data['r_index'] = matched_solar_indices[:,2]
                        extended_data['dst_index'] = matched_solar_indices[:,3]
                        extended_data['f_index'] = matched_solar_indices[:,4]

                    extended_data = extended_data[list(TableDescription.__dict__['columns'].keys())] # drop the columns we don't want


                    if solar_indices_mode == "none":
                        extended_data = extended_data.astype([
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
                        ('doy', 'f4'),
                    ])  
                    else:
                        extended_data = extended_data.astype([
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
                            ('doy', 'f4'),
                            ('kp_index', 'f4'),
                            ('r_index', 'f4'),
                            ('dst_index', 'f4'),
                            ('f_index', 'f4')
                        ])  
                    out_table.append(extended_data)
                    out_table.flush()
                
                finally:
                    in_h5.close()
                
        finally:
            out_h5.close()



if __name__ == "__main__":
    config_path = "config/reorganize_data_config.yaml"
    with open(config_path, 'r') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    
   
    subsampling_ratio = config["subsampling_ratio"]
    years_dict = config["years"]
    output_path = config["dslab_path"] + config["output_dir_name"] + "/"
    datapath = config["datapath"]
    solar_indices_mode = config["solar_indices_mode"]
    solar_indices_path = config["solar_indices_path"]
    assert solar_indices_mode in ["none", "daily", "hourly"], "solar_indices_mode must be one of 'none', 'daily' or 'hourly'."


    os.makedirs(output_path, exist_ok=True)
    shutil.copy(config_path, output_path + "reorganize_data_config.yaml")
    leap_years = [2020, 2024]
    
    # iterate through all given years and months
    for year, months_in_year in years_dict.items():
        day_count_per_month = [31,28,31,30,31,30,31,31,30,31,30,31]
        if year in leap_years:
            day_count_per_month[1] = 29
        month_indices = np.cumsum(day_count_per_month).tolist()
        

        datapaths_per_month = []

        if months_in_year != "all":
            assert type(month_indices) is list, "month_indices must be \"all\" or a list" 
            month_indices = [0] + month_indices
            for month in months_in_year:
                    datapaths_per_month.append(
                        [datapath + f"{year}/{str(doy + 1).zfill(3)}/ccl_{year}{str(doy + 1).zfill(3)}_30_5.h5" for doy in range(month_indices[month-1], month_indices[month])]
                    )
        else:        
            for last_index, index in zip([0] + month_indices[:-1], month_indices):
                datapaths_per_month.append(
                    [datapath + f"{year}/{str(doy + 1).zfill(3)}/ccl_{year}{str(doy + 1).zfill(3)}_30_5.h5" for doy in range(last_index, index)]
                )
        

        Parallel(n_jobs=-1)(delayed(reorganize_month)(year, month_idx + 1, datapaths, output_path, subsampling_ratio, solar_indices_mode, solar_indices_path) for month_idx, datapaths in enumerate(datapaths_per_month))

        # for month_idx, datapaths in enumerate(datapaths_per_month):

        #     month = month_idx + 1
        #     print(f"month: {month}")
        #     reorganize_month(year, month_idx + 1, datapaths, output_path, subsampling_ratio, solar_indices_mode, solar_indices_path)
            
        
    