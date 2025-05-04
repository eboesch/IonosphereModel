from spacepy.coordinates import Coords
from spacepy.time import Ticktock
from datetime import datetime, timedelta

import pandas as pd
import numpy as np
import altair as alt
alt.data_transformers.disable_max_rows()

df = pd.read_csv("outputs/vtec_grid.csv")

lats = df["sta_lat"]
lons = df["sta_lon"]
def coord_transform(input_type, output_type, lats, lons, epochs):
    coords = np.array([[1 + 450 / 6371, lat, lon] for lat, lon in zip(lats, lons)], dtype=np.float64)
    geo_coords = Coords(coords, input_type, 'sph')
    geo_coords.ticks = Ticktock(epochs, 'UTC')
    return geo_coords.convert(output_type, 'sph')

date = datetime.strptime("2024-01-01", "%Y-%m-%d") + timedelta(days=82 - 1)
epochs = [date + timedelta(seconds=int(sod)) for sod in df["sod"]]
# Step 2: Transform to SM coordinates
sm_coords = coord_transform('SM', 'GEO', lats, lons, epochs)

out_coords = sm_coords.data

sub_df = df
sub_df["GEO_0"] = out_coords[:,0]
sub_df["GEO_1"] = out_coords[:,1]
sub_df["GEO_2"] = out_coords[:,2]
sub_df.to_csv("vtec_geo.csv",index=False)