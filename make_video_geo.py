from spacepy.coordinates import Coords
from spacepy.time import Ticktock
from datetime import datetime, timedelta
from vega_datasets import data
import pandas as pd
import numpy as np
import altair as alt
alt.data_transformers.disable_max_rows()

countries = alt.topo_feature(data.world_110m.url, 'countries')
back_ground = alt.Chart(countries).mark_geoshape(
    fill='lightgray',
    stroke='white',
    opacity=0.25
).project(
    "mercator"
)

df = pd.read_csv("outputs/vtec_grid_with_geo.csv")

for s in np.unique(df["sod"]):
    sub_df = df[df["sod"] == s]    
    spec_graph = alt.Chart(sub_df).mark_rect(strokeWidth=3).encode(
            x=alt.X("geo_lon:O",axis=alt.Axis(labels=False,ticks=False,grid=False,title="")),
            y=alt.Y("geo_lat:O",sort="descending",axis=alt.Axis(labels=False,ticks=False,grid=False,title="")),
            color=alt.Color('pred_stec:Q',scale=alt.Scale(scheme='plasma'),legend=None),
            fill=alt.Fill('pred_stec:Q',scale=alt.Scale(scheme='plasma'),legend=None),
            stroke=alt.Fill('pred_stec:Q',scale=alt.Scale(scheme='plasma'),legend=None),
        ).properties(height=360,width=360)+back_ground
    spec_graph.save("geo_images/grid_"+str(s)+".png")


import cv2
import os

image_folder = 'geo_images'
video_name = 'outputs/video_geo.avi'

images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
frame = cv2.imread(os.path.join(image_folder, images[0]))
height, width, layers = frame.shape

video = cv2.VideoWriter(video_name, 0, 30, (width,height))

for image in images:
    video.write(cv2.imread(os.path.join(image_folder, image)))

cv2.destroyAllWindows()
video.release()