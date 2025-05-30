"""Generates the example dataset used to illustrate how to run the training code."""
import h5py
import random 
random.seed(42)

if __name__ == "__main__":
    filepath = "/cluster/work/igp_psr/arrueegg/GNSS_STEC_DB/2024/183/ccl_2024183_30_5.h5"
    nodepath = "/2024/183/all_data" 
    outpath = 'example_code/2024/183/ccl_2024183_30_5.h5'
    file = h5py.File(filepath, 'r')
    data = file[nodepath][:]
    indices = random.sample(range(data.shape[0]), 20000)
    mock_data = data[indices]
    with h5py.File(outpath, 'w') as f:
        f.create_dataset(nodepath, data=mock_data)

    with h5py.File(outpath, 'r') as f:
        mock_data_loaded = f[nodepath][:]
        assert mock_data_loaded[0] == mock_data[0]

