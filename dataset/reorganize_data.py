"""A script to take the original dataset and generate a new subsampled dataset with train, val, test data separated in different files."""
import os
import tables
import numpy as np
from tqdm import tqdm
import shutil
import yaml
from joblib import Parallel, delayed

# srun --ntasks=1 --cpus-per-task=6 --mem-per-cpu=4096 -t 600 -o file.out -e file.err python dataset/reorganize_data.py &

class TableDescription(tables.IsDescription):
    """
    Schema of the table
    """
    station = tables.StringCol(itemsize=4, shape=(), dflt=np.bytes_(''), pos=0)
    sat = tables.StringCol(itemsize=3, shape=(), dflt=np.bytes_(''), pos=1)
    stec = tables.Float32Col(shape=(), dflt=np.float32(0.0), pos=2)
    vtec = tables.Float32Col(shape=(), dflt=np.float32(0.0), pos=3)
    satele = tables.Float32Col(shape=(), dflt=np.float32(0.0), pos=4)
    satazi = tables.Float32Col(shape=(), dflt=np.float32(0.0), pos=5)
    lon_ipp = tables.Float32Col(shape=(), dflt=np.float32(0.0), pos=6)
    lat_ipp = tables.Float32Col(shape=(), dflt=np.float32(0.0), pos=7)
    sm_lat_ipp = tables.Float32Col(shape=(), dflt=np.float32(0.0), pos=8)
    sm_lon_ipp = tables.Float32Col(shape=(), dflt=np.float32(0.0), pos=9)
    sod = tables.Float32Col(shape=(), dflt=np.float32(0.0), pos=10)
    lat_sta = tables.Float32Col(shape=(), dflt=np.float32(0.0), pos=11)
    lon_sta = tables.Float32Col(shape=(), dflt=np.float32(0.0), pos=12)
    sm_lat_sta = tables.Float32Col(shape=(), dflt=np.float32(0.0), pos=13)
    sm_lon_sta = tables.Float32Col(shape=(), dflt=np.float32(0.0), pos=14)
    gfphase = tables.Float32Col(shape=(), dflt=np.float32(0.0), pos=15)


def get_expected_rows(group: str, subsampling_ratio: float) -> float:
    """
    Returns the expected number of rows of the table when reorganizing data of a whole month, 
    based on the subsampling ratio and group (train, test or val).
    """
    rows_per_raw_day = 20e6
    row_fraction = 0.7 if group == "train" else (0.2 if group == "test" else 0.1)
    month_max_days = 31
    expectedrows = rows_per_raw_day*row_fraction*subsampling_ratio*month_max_days
    return expectedrows


def reorganize_month(
        year: int, 
        month: int, 
        datapaths: list[list[str]], 
        output_path: str, 
        subsampling_ratio: float, 
    ) -> None:
    """
    Subsamples the data of the corresponding month and year at the given datapaths and saves it at the indicated location.

    Args:
        year (int): The year which the desired data is from.
        month (int): The month we want to reorganize the data from.
        datapaths (list[list[str]]): A list of lists. Each sub-list contains all relevant datapaths for that month
        output_path (str): The path to the folder in which the new data should be stored.
        subsampling ratio (float): Percentage of the data that should be kept. A value of 0.2 means 2% of the data are kept.

    Returns:
        None
    """
    for group in ["train", "val", "test"]:
        # fetch list of stations that correspond to this data split
        with open(f"dataset/{group}.list", "r") as file:
            stations = [line.strip().encode('utf8') for line in file]
        
        month_output_path = output_path + str(year) + '-' + str(month) + "-" + group + ".h5"
        if os.path.exists(month_output_path): # if a folder for this month already exists, skip it
            print(f"Skipping {month_output_path}")
            continue
        
        try:
            print(f"Attempting {month_output_path}")
            # create empty table
            out_h5 = tables.open_file(month_output_path, mode='w')
            expectedrows = get_expected_rows(group, subsampling_ratio)
            out_table = out_h5.create_table('/', 'all_data', TableDescription, expectedrows=expectedrows)
            
            for datapath in tqdm(datapaths):
                if not os.path.exists(datapath):
                    # if a file doesn't exist, skip it
                    continue

                try:
                    # fetch data
                    in_h5 = tables.open_file(datapath, mode='r')
                    doy = datapath.split('/')[-2]
                    data = in_h5.get_node(f"/{year}/{doy}/all_data")

                    mask_stations = np.isin(data.col('station'), stations) 
                    # subsample based on  second of day. to ensure good coverage, add random perturbation to the sod we want to keep
                    if subsampling_ratio == 1:
                        mask_subsampling = np.ones_like(mask_stations)
                    else:
                        np.random.seed(10 + month + int(doy) + year)
                        seconds = [int(30 / subsampling_ratio)*i + 30*np.random.randint(0, 20) for i in range(int(2880*subsampling_ratio))]
                        mask_subsampling = np.isin(data.col('sod'), seconds)

                    # filter data based on subsampling and stations
                    filtered_data = data[mask_subsampling*mask_stations]

                    filtered_data['gfphase'] = doy

                    # drop the columns we don't want
                    filtered_data = filtered_data[list(TableDescription.__dict__['columns'].keys())]  
                    
                    # fix the schema
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
                    out_table.append(filtered_data) # add the new data to the table
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
 
    # create directory to save new data to
    os.makedirs(output_path, exist_ok=True)
    shutil.copy(config_path, output_path + "reorganize_data_config.yaml")
    leap_years = [2020, 2024]
    

    
    # iterate through all given years and months
    for year, months_in_year in years_dict.items():
        # create list of indices of the final day of each month
        day_count_per_month = [31,28,31,30,31,30,31,31,30,31,30,31]
        if year in leap_years:
            day_count_per_month[1] = 29
        month_indices = np.cumsum(day_count_per_month).tolist()
        
        # create nested list of the relevant datapaths for each month
        datapaths_per_month = []

        if months_in_year != "all":
            assert type(months_in_year) is list, "months_in_year must be \"all\" or a list" 

            # iterate through all desired months and make a list of the corresponding datapaths
            month_indices = [0] + month_indices
            for month in months_in_year:
                    datapaths_per_month.append(
                        [datapath + f"{year}/{str(doy + 1).zfill(3)}/ccl_{year}{str(doy + 1).zfill(3)}_30_5.h5" for doy in range(month_indices[month-1], month_indices[month])]
                    )
        else:
            # iterate through all months and make a list of the corresponding datapaths        
            for last_index, index in zip([0] + month_indices[:-1], month_indices):
                datapaths_per_month.append(
                    [datapath + f"{year}/{str(doy + 1).zfill(3)}/ccl_{year}{str(doy + 1).zfill(3)}_30_5.h5" for doy in range(last_index, index)]
                )
        
        # parallelizing allows individual months to be reorganized in parallel, instead of in sequence
        # Parallel(n_jobs=-1)(delayed(reorganize_month)(year, month_idx + 1, datapaths, output_path, subsampling_ratio) for month_idx, datapaths in enumerate(datapaths_per_month))

        # iterate through all months in series
        for month_idx, datapaths in enumerate(datapaths_per_month):
            month = month_idx + 1
            print(f"Month: {month}")
            reorganize_month(year, month_idx + 1, datapaths, output_path, subsampling_ratio)

        
    