#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 02:02:40 2024

@author: nitaishah
"""

import tensorflow as tf

tf.config.optimizer.set_experimental_options({"disable_meta_optimizer": True})

# Disable any GPU-related issues by forcing the model to use CPU or other settings
tf.config.set_visible_devices([], 'GPU')

import keras
from keras.models import load_model
import rasterio
import matplotlib.pyplot as plt
import numpy as np

# Check available devices
print("Available devices:", tf.config.list_physical_devices())


def read_image_as_array(file_path):
    with rasterio.open(file_path) as src:
    # Reading all 13 bands
       
        blue = src.read(1)                 # Band 2 (490 nm)
        green = src.read(2)                # Band 3 (560 nm)
        red = src.read(3)               # Band 4 (665 nm)
   
        # Optional: Stack into a single array of shape (13, H, W)
        multispectral_array = src.read(list(range(1, 4)))  # Reads all bands at once
        
        return multispectral_array
    
image_test = read_image_as_array("/Users/nitaishah/Desktop/ECEN-PROJECT/Segmentation_Masking/Input/Input_GeoFs.png")
plt.imshow(image_test.transpose(2,1,0))

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

tile_size = 32
padded_image, pad_h, pad_w = pad_image(image_test, tile_size)
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

model_path = "/Users/nitaishah/Desktop/ECEN-PROJECT/Segment_GeoFS/model_rgb_32*32_2.h5"
model = load_model(model_path)

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

# Step 1: Generate mask tiles
mask_tiles = predict_and_generate_masks(tiles, model, class_colors)
print("Complete Tiles Shape:", tiles.shape) 
# Step 2: Reconstruct the full mask image
full_mask_image = reconstruct_full_image(mask_tiles)

plt.figure(figsize=(10,10))
plt.imshow(full_mask_image)
plt.title("Full Mask Image")
plt.axis("off")
plt.show()
