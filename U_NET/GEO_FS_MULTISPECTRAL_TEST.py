#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 30 10:33:12 2024

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


import tifffile as tiff

output_tiff_path = "/Users/nitaishah/Desktop/ECEN-PROJECT/demo/GEOFS_Multispectral_image.tif"




model_path = "/Users/nitaishah/Desktop/ECEN-PROJECT/demo/model_1.h5"
model = load_model(model_path)




def create_windows(tiff_path, tile_height, tile_width):
    """
    Splits a 3D array into overlapping or non-overlapping tiles of a given size.

    Parameters:
    - data (np.ndarray): Input array of shape (bands, height, width).
    - tile_height (int): Height of each tile.
    - tile_width (int): Width of each tile.

    Returns:
    - np.ndarray: Array of tiles with shape (rows, cols, bands, tile_height, tile_width).
    
    """
    with rasterio.open(tiff_path) as src:
        data = src.read()  # Read all bands into a numpy array (shape: bands, height, width)
    
    bands, height, width = data.shape

    # Ensure height and width are divisible by tile size
    pad_height = (tile_height - height % tile_height) % tile_height
    pad_width = (tile_width - width % tile_width) % tile_width

    # Pad the array to ensure all tiles are complete
    padded_data = np.pad(data, ((0, 0), (0, pad_height), (0, pad_width)), mode='constant', constant_values=0)

    # Calculate the number of tiles in each dimension
    rows = padded_data.shape[1] // tile_height
    cols = padded_data.shape[2] // tile_width

    # Split the array into tiles
    tiles = padded_data.reshape(
        bands,
        rows,
        tile_height,
        cols,
        tile_width
    ).transpose(1, 3, 0, 2, 4)  # Rearrange to (rows, cols, bands, tile_height, tile_width)

    return tiles

tile_height = 64
tile_width = 64

# Create windows
tiles = create_windows(output_tiff_path, tile_height, tile_width)
tiles

def predict_and_generate_masks(tiles, model, class_colors):
    rows, cols, _, _, _ = tiles.shape
    mask_tiles = []
    for i in range(rows):
        row_masks = []
        for j in range(cols):
            tile = tiles[i, j]  # Shape: (C, H, W)
            reshaped_tile = np.moveaxis(tile, 0, -1)  # Convert to (H, W, C) for prediction

            # Normalize the tile (assuming input range is 0-255)
            reshaped_tile = reshaped_tile / 255.0  # Scale to range [0, 1]
            
            reshaped_tile = np.expand_dims(reshaped_tile, axis=0)  # Add batch dimension
            
            # Predict the class for the tile
            predictions = model.predict(reshaped_tile)
            predicted_class = np.argmax(predictions, axis=1)[0]  # Get the class index
            
            # Create the mask using the class color
            color = class_colors[predicted_class]
            mask = np.zeros((tile.shape[1], tile.shape[2], 3))  # Initialize mask (H, W, 3)
            mask[:, :, 0] = color[0]  # Red channel
            mask[:, :, 1] = color[1]  # Green channel
            mask[:, :, 2] = color[2]  # Blue channel
            
            row_masks.append(mask)
        mask_tiles.append(row_masks)
    
    return mask_tiles

class_colors = {
    0: (0.4, 0.8, 0.4),          # Annual Crop - Green
    1: (0.0, 0.5, 0.0),          # Forest - Dark Green
    2: (0.6, 1.0, 0.6),          # Herbaceous Vegetation - Light Green
    3: (0.6, 0.6, 0.6),          # Highway - Gray
    4: (1.0, 0.5, 0.0),          # Industrial - Orange
    5: (0.8, 1.0, 0.4),          # Pasture - Yellow-Green
    6: (0.6, 0.3, 0.0),          # Permanent Crop - Brown
    7: (1.0, 0.0, 0.0),          # Residential - Red
    8: (0.0, 0.0, 1.0),          # River - Blue
    9: (0.0, 0.4, 1.0),          # Sea/Lake - Light Blue
}

# Step 4: Apply the model and generate masks
masks = predict_and_generate_masks(tiles, model, class_colors)

def reconstruct_full_image(mask_tiles):
    rows = len(mask_tiles)
    cols = len(mask_tiles[0])
    tile_h, tile_w, _ = mask_tiles[0][0].shape
    
    full_image = np.zeros((rows * tile_h, cols * tile_w, 3))
    for i in range(rows):
        for j in range(cols):
            full_image[
                i * tile_h : (i + 1) * tile_h,
                j * tile_w : (j + 1) * tile_w,
                :
            ] = mask_tiles[i][j]
    
    return full_image

full_image = reconstruct_full_image(masks)
plt.figure(figsize=(10,10))
plt.imshow(full_image)
plt.title("Full Mask Image")
plt.axis("off")
plt.show()

