"""Script used to plot rays in 3D space from stations to ionospheric pierce points."""
import numpy as np
import pandas as pd
import h5py
from matplotlib import pyplot as plt
from matplotlib import colormaps

datapath = "/cluster/work/igp_psr/arrueegg/GNSS_STEC_DB/2024/322/ccl_2024322_30_5.h5"
file = h5py.File(datapath, 'r')
data = file["2024"]["322"]["all_data"]

data = data[(data["sod"] <= 0)]

data = data[(data["lon_ipp"] > -100) *  (data["lon_ipp"] < 100) * (data["lat_ipp"] > -60) *  (data["lat_ipp"] < 60)]

cmap = colormaps["viridis"]
# NOTE: We plot in lon-lat coordinates instead of x and y. In order to plot rays, topology / shape of plot should be the same.
coords = np.array(data[['lon_sta', 'lon_ipp', 'lat_sta', 'lat_ipp', 'sod', 'station']].tolist())

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d", elev=75)
for coord in coords:
    coord = coord
    ax.plot(coord[:2].astype(float), coord[2:4].astype(float), [0, 1], color=cmap((hash(coord[5]) % 1000) / 1000), alpha=0.2)
fig.savefig('outputs/nerfs3D.png')