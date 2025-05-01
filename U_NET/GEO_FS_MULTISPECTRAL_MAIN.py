#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 29 15:55:16 2024

@author: nitaishah
"""

import os
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import tifffile as tiff
import rasterio
from rasterio.plot import show
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
import tensorflow as tf
from tensorflow import keras
from keras import layers, models
from keras.models import load_model
import skimage
from skimage.metrics import structural_similarity as ssim

def read_image_as_array(file_path):
    try:
        with rasterio.open(file_path) as src:
            # Read the RGB bands
            blue = src.read(1)  # Band 1
            green = src.read(2)  # Band 2
            red = src.read(3)  # Band 3
            # Stack into a single array (C, H, W)
            multispectral_array = np.stack([red, green, blue])
            return multispectral_array
    except Exception as e:
        print(f"Error reading image {file_path}: {e}")
        return None
    
array = read_image_as_array("/Users/nitaishah/Desktop/ECEN-PROJECT/EuroSAT_RGB/Industrial/Industrial_2230.jpg")
array.shape

def pad_image(image, tile_size):
    _, height, width = image.shape
    pad_height = (tile_size - height % tile_size) % tile_size
    pad_width = (tile_size - width % tile_size) % tile_size
    
    padded_image = np.pad(
        image,
        pad_width=((0, 0), (0, pad_height), (0, pad_width)),  # No padding on channels
        mode="constant",
        constant_values=0
    )
    return padded_image, pad_height, pad_width

tile_size = 64
padded_image, pad_h, pad_w = pad_image(array, tile_size)
print("Padded Image Shape:", padded_image.shape)
print("Padding Applied: Height =", pad_h, "Width =", pad_w)

def divide_into_tiles(image, tile_size):
    _, height, width = image.shape
    # Reshape and transpose to divide into tiles
    tiles = (
        image.reshape(3, height // tile_size, tile_size, width // tile_size, tile_size)
        .transpose(1, 3, 0, 2, 4)  # Rearrange to (tiles_row, tiles_col, channels, tile_h, tile_w)
    )
    return tiles

def divide_into_complete_tiles(image, tile_size):
    """
    Divide an image into complete tiles of a given size, ignoring padded regions.
    """
    _, height, width = image.shape
    
    # Calculate the number of complete tiles
    num_rows = height // tile_size
    num_cols = width // tile_size

    # Extract only the region that forms complete tiles
    cropped_image = image[:, :num_rows * tile_size, :num_cols * tile_size]
    
    # Divide into tiles
    tiles = (
        cropped_image.reshape(3, num_rows, tile_size, num_cols, tile_size)
        .transpose(1, 3, 0, 2, 4)  # Rearrange to (tiles_row, tiles_col, channels, tile_h, tile_w)
    )
    
    return tiles

tiles = divide_into_complete_tiles(padded_image, tile_size)
print("Tiles Shape:", tiles.shape)  # Expect shape: (rows, cols, channels, 64, 64)

def display_individual_tiles(tiles, rows, cols):
    rows, cols, _, _, _ = tiles.shape
    tile_count = 0
    for i in range(rows):
        for j in range(cols):
            tile = np.moveaxis(tiles[i, j], 0, -1)  # Convert (C, H, W) to (H, W, C)
            plt.figure(figsize=(4, 4))  # Set the size of each plot
            plt.imshow(tile)
            plt.title(f"Tile {tile_count} (Row {i}, Col {j})")
            plt.axis('off')  # Hide axis for better visualization
            plt.show()
            tile_count += 1
            
display_individual_tiles(tiles, 30, 30)

final_predicted_array = []

import os
models_folder = "/Users/nitaishah/Desktop/ECEN-PROJECT/U-NET/Weights/"
model_files = [f for f in os.listdir(models_folder) if f.endswith('.h5')]
model_files

models = []
for model_file in model_files:
    model_path = os.path.join(models_folder, model_file)  # Full path to the model file
    model = load_model(model_path)  # Load the model
    models.append(model)  # Append the model to the models list



def reconstruct_full_image(mask_tiles):
    rows = len(mask_tiles)
    cols = len(mask_tiles[0])
    tile_h, tile_w, _ = mask_tiles[0][0].shape
    
    full_image = np.zeros((rows * tile_h, cols * tile_w, 1))
    for i in range(rows):
        for j in range(cols):
            full_image[
                i * tile_h : (i + 1) * tile_h,
                j * tile_w : (j + 1) * tile_w,
                :
            ] = mask_tiles[i][j]
    
    return full_image


for model_file in model_files:
    model_path = os.path.join(models_folder, model_file)  # Full path to the model file
    model = load_model(model_path)  # Load the model
    print(f"Processing model: {model_file}")
    
    # Create the prediction array
    pred_array = []

    # Iterate over the tiles and make predictions
    for tile in tiles:
        tile = tile.transpose(0, 2, 3, 1)  # Adjust shape for model input (batch_size, height, width, channels)
        y_pred = model.predict(tile)  # Predict using the model
        pred_array.append(y_pred)  # Store predictions

    # Convert the list to a NumPy array
    pred_array = np.array(pred_array)

    # Reconstruct the full image from the predicted tiles
    full_image = reconstruct_full_image(pred_array)
    print(f"Full image shape for {model_file}: {full_image.shape}")

    # Append the reconstructed full image to the final predictions list
    final_predicted_array.append(full_image)


pred_array[0].shape
final_predicted_array
final_predicted_array = np.array(final_predicted_array)
array.shape
final_predicted_array.shape


rgb_height, rgb_width = array.shape[1], array.shape[2]
cropped_multispectral_array = final_predicted_array[:, :rgb_height, :rgb_width]


final_combined_array = np.zeros((13, array.shape[1], array.shape[2]))  # 13 bands, same height and width as the input

final_combined_array[0] = cropped_multispectral_array[5].squeeze()  # coastal_aerosol
final_combined_array[1] = array[0]                                # blue
final_combined_array[2] = array[1]                                # green
final_combined_array[3] = array[2]                                # red
final_combined_array[4] = cropped_multispectral_array[1].squeeze()  # vegetation_red_edge_1
final_combined_array[5] = cropped_multispectral_array[8].squeeze()  # vegetation_red_edge_2
final_combined_array[6] = cropped_multispectral_array[6].squeeze()  # vegetation_red_edge_3
final_combined_array[7] = cropped_multispectral_array[4].squeeze()  # nir
final_combined_array[8] = cropped_multispectral_array[9].squeeze()  # narrow_nir
final_combined_array[9] = cropped_multispectral_array[2].squeeze()  # water_vapor
final_combined_array[10] = cropped_multispectral_array[3].squeeze() # cirrus
final_combined_array[11] = cropped_multispectral_array[0].squeeze() # swir1
final_combined_array[12] = cropped_multispectral_array[7].squeeze() # swir2

final_combined_array.shape

import tifffile as tiff

output_tiff_path = "GEOFS_Multispectral_image_2.tif"


height, width = final_combined_array.shape[1], final_combined_array.shape[2]

# Open a new raster file to write the data
with rasterio.open(
    output_tiff_path, 
    'w', 
    driver='GTiff',  # GeoTIFF format
    count=13,  # Number of bands
    dtype='float32',  # You can use float32 or another type depending on the data
    width=width, 
    height=height,
    crs='+proj=latlong',  # Set coordinate reference system (adjust as necessary)
    transform=rasterio.transform.from_origin(0, height, 1, 1)  # Adjust transform as needed
) as dst:
    # Write each band to the TIFF file
    for i in range(13):
        dst.write(final_combined_array[i], i + 1)  # Band indices are 1-based in rasterio
        
with rasterio.open(output_tiff_path) as src:
    # Read all bands
    all_bands = src.read()  # Returns a 2D array for each band
    print(all_bands.shape)  # (13, height, width)