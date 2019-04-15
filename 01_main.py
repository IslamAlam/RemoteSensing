# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 01:48:14 2019
Last modified on Wed Apr 06 2019
@author: Islam Mansour
Purpose: Reada shapefile with geopandas as a dataframe and reproject the
         shapefile.
"""
###############################################################################
# Import Packages
# ------------------------------
#
# To begin, import the needed packages. You will use a combination of several EarthPy
# modules including spatial, plot and mask.

import rasterio as rio
from rasterio.plot import show
from rasterio.mask import mask
from shapely.geometry import mapping
import numpy as np
import os
import matplotlib.pyplot as plt
import geopandas as gpd
import earthpy as et
import cartopy as cp
import rasterio as rio
from rasterio.warp import calculate_default_transform, reproject, Resampling

plt.ion()

os.chdir(os.path.join(et.io.HOME, '/home/islam/Desktop/multiband-classifier'))
# optional - turn off warnings
import warnings
warnings.filterwarnings('ignore')

#%%
RASTER_DATA_FILE = "./data/image/crop_p224r63_all_bands.tif"
TRAIN_DATA_PATH = "./data/train/"
TEST_DATA_PATH = "./data/test/"


#%%


# https://www.earthdatascience.org/courses/earth-analytics-python/lidar-raster-data/reproject-raster/
def reproject_layer(inpath, outpath, new_crs):
    '''
    * This function reprojects a raster GeoTIFF or vector (shp) to a new selected Coordinate Reference System (CRS) then
    save the reprojected GeoTIFF/shp to outpath

    PARAMETERS:
        inpath = path to orginal GeoTIFF to be reprojected/ shp; for example:
        outpath = path to save the newly reprojected GeoTIFF; for example: "data/reprojected.tif" or shp
        new_crs = required CRS projection ; for example: "epsg:4326"

    RETURNS:
        create a new reprojected GeoTIFF

    REQUIREMENTS:
        import rasterio as rio
        import os
        import earthpy as et
        import numpy as np
        from rasterio.warp import calculate_default_transform, reproject, Resampling
    '''
    dst_crs = new_crs  # CRS for web meractor

    if '.shp' in inpath:
        src = gpd.read_file(inpath)
        # Reproject the data
        data = src.to_crs(epsg=dst_crs.split('EPSG:', 1)[1])
        # Save to disk
        data.to_file(outpath)


    else:
        with rio.open(inpath) as src:
            transform, width, height = calculate_default_transform(
                src.crs, dst_crs, src.width, src.height, *src.bounds)
            kwargs = src.meta.copy()
            kwargs.update({
                'crs': dst_crs,
                'transform': transform,
                'width': width,
                'height': height
            })

            with rio.open(outpath, 'w', **kwargs) as dst:
                for i in range(1, src.count + 1):
                    reproject(
                        source=rio.band(src, i),
                        destination=rio.band(dst, i),
                        src_transform=src.transform,
                        src_crs=src.crs,
                        dst_transform=transform,
                        dst_crs=dst_crs,
                        resampling=Resampling.nearest)
#%%


# define crs
new_crs = 'EPSG:29192'


# import the GeoTIFF
TIFF_INPATH = RASTER_DATA_FILE
TIFF_OUTPATH = RASTER_DATA_FILE[:-4] + '_' + new_crs.split('EPSG:', 1)[1] + RASTER_DATA_FILE[-4:]

# import the data train data
TRAIN_DATA_INPATH = (TRAIN_DATA_PATH + "train_data.shp")
TRAIN_DATA_OUTPATH = TRAIN_DATA_INPATH[:-4] + '_' + new_crs.split('EPSG:', 1)[1] + TRAIN_DATA_INPATH[-4:]

# import the test data
TEST_DATA_INPATH = (TEST_DATA_PATH + "val_data.shp")
TEST_DATA_OUTPATH = TEST_DATA_INPATH[:-4] + '_' + new_crs.split('EPSG:', 1)[1] + TEST_DATA_INPATH[-4:]

#%% Re-project
reproject_layer(TIFF_INPATH, TIFF_OUTPATH, new_crs)
reproject_layer(TEST_DATA_INPATH, TEST_DATA_OUTPATH, new_crs)
reproject_layer(TRAIN_DATA_INPATH, TRAIN_DATA_OUTPATH, new_crs)

#%%
# Check to make sure function worked

tiff_project = rio.open(TIFF_OUTPATH)
train_data_project = gpd.read_file(TRAIN_DATA_OUTPATH)
test_data_project = gpd.read_file(TEST_DATA_OUTPATH)

print('GeoTIFF file crs', tiff_project.crs)
print("Train shape file crs", train_data_project.crs)
print("Val shape file crs", test_data_project.crs)

#%% https://www.earthdatascience.org/workshops/gis-open-source-python/crop-raster-data-in-python/

# open the lidar chm
with rio.open(TIFF_OUTPATH) as src:
    tiff_im = src.read(masked = True)[0]
    extent = rio.plot.plotting_extent(src)
    soap_profile = src.profile


fig, ax = plt.subplots(figsize=(10, 10))

show(tiff_im,
     cmap='terrain',
     ax=ax,
     extent=extent)

ax.set_title("Image",
             fontsize = 16);


# ax.imshow(tiff_project.read(1), vmin=0, vmax=0)
#%%

vector_layer = train_data_project
fig, ax = plt.subplots(figsize=(10, 10))

ax.imshow(tiff_im,
          cmap='terrain',
          extent=extent)
vector_layer.plot(ax=ax, alpha=.6, color='r')
