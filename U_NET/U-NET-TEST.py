#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 03:44:11 2024

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

#tf.config.optimizer.set_experimental_options({"disable_meta_optimizer": True})

# Disable any GPU-related issues by forcing the model to use CPU or other settings
#tf.config.set_visible_devices([], 'GPU')

# Check available devices
print("Available devices:", tf.config.list_physical_devices())

# Path to the dataset
dataset_dir = '/Users/nitaishah/Desktop/ECEN-PROJECT/EuroSAT_MS/'  # Change this to your dataset path



class_labels = ["AnnualCrop", "Forest", "HerbaceousVegetation", "Highway", "Industrial",
                "Pasture", "PermanentCrop", "Residential", "River", "SeaLake"]

data = []
valid_extensions = {".tif", ".tiff"}

for label, class_name in enumerate(class_labels):
    class_dir = os.path.join(dataset_dir, class_name)
    
    # Loop through each file in the class directory
    for image_name in os.listdir(class_dir):
        # Check if the file has a valid extension
        if os.path.splitext(image_name)[1].lower() in valid_extensions:
            image_path = os.path.join(class_dir, image_name)
            
            # Add image path and label to the data list
            data.append({"image_path": image_path, "label": label, "class_name": class_name})


df = pd.DataFrame(data)
print(df.head())

df_sample = df.sample(n=5000, random_state=42).reset_index(drop=True)

def read_image_as_array(file_path):
    with rasterio.open(file_path) as src:
        blue = src.read(2)  # Band 2 (490 nm)
        green = src.read(3)  # Band 3 (560 nm)
        red = src.read(4)  # Band 4 (665 nm)
        ms_band = src.read(13)  # Band 8 (842 nm)
        
        # Stack RGB and NIR bands
        multispectral_array = np.stack((blue, green, red, ms_band), axis=0)
        
        return multispectral_array

def show_rgb(multispectral_array):
    # Extract and normalize RGB bands
    red = multispectral_array[2]
    green = multispectral_array[1]
    blue = multispectral_array[0]
    rgb = np.dstack((red, green, blue))
    rgb = rgb / (np.max(rgb) + 1e-8)
    plt.imshow(rgb)
    plt.axis('off')
    plt.show()
    
def get_X(multispectral_array):
    # Extract RGB bands and normalize
    red = multispectral_array[2]
    green = multispectral_array[1]
    blue = multispectral_array[0]
    rgb = np.dstack((red, green, blue))
    rgb = rgb / (np.max(rgb) + 1e-8)
    return rgb

def get_y(multispectral_array):
    # Extract NIR band and normalize
    ms_band = multispectral_array[3]
    ms_band = ms_band / (np.max(ms_band) + 1e-8)
    return ms_band

def get_dataset_as_array(dataset):
    X_array = []
    y_array = []
    for i in range(len(dataset)):
        array = read_image_as_array(dataset.iloc[i, 0])
        rgb = get_X(array)
        X_array.append(rgb)
        ms_band = get_y(array)
        y_array.append(ms_band)
    
    # Convert to NumPy arrays
    return np.array(X_array), np.array(y_array)

X_array, y_array = get_dataset_as_array(df_sample)
y_array = np.expand_dims(y_array, axis=-1)

model_path = "/Users/nitaishah/Desktop/ECEN-PROJECT/U-NET/Weights/U-NET-narrow-swir-2.h5"
model = load_model(model_path)

y_pred = model.predict(X_array)



def evaluate_predictions(y_test, y_pred):
    
    results = {}

    mae = np.mean(np.abs(y_pred - y_test))
    results["MAE"] = mae
    print("Mean Absolute Error:", mae)

    diff = np.abs(y_pred[2] - y_test[2])  # Example with the 10th sample
    plt.figure(figsize=(8, 6))
    plt.imshow(diff.squeeze(), cmap='hot')
    plt.title("Difference Map")
    plt.colorbar()
    plt.show()

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(y_test[2].squeeze(), cmap='gray')
    plt.title("Ground Truth")

    plt.subplot(1, 2, 2)
    plt.imshow(y_pred[2].squeeze(), cmap='gray')
    plt.title("Prediction")
    plt.show()

    # Calculate Histogram Intersection
    def histogram_intersection(h1, h2, bins=50):
        hist_1, _ = np.histogram(h1.ravel(), bins=bins, range=(0, 1))
        hist_2, _ = np.histogram(h2.ravel(), bins=bins, range=(0, 1))
        return np.minimum(hist_1, hist_2).sum() / hist_1.sum()

    intersection = histogram_intersection(y_pred, y_test)
    results["Histogram Intersection"] = intersection
    print("Histogram Intersection:", intersection)

    # Calculate Structural Similarity Index (SSIM)
    ssim_value = ssim(y_test.squeeze(), y_pred.squeeze(), data_range=y_test.max() - y_test.min())
    results["SSIM"] = ssim_value
    print("SSIM:", ssim_value)

    # Visualize the difference map for one sample (specific index)
    difference_map = np.abs(y_pred - y_test)
    plt.figure(figsize=(8, 6))
    plt.imshow(difference_map[1].squeeze(), cmap='hot')
    plt.colorbar()
    plt.title("Difference Map (Sample 1)")
    plt.show()

    # Flatten arrays for histogram comparison
    y_pred_flat = y_pred.flatten()
    y_test_flat = y_test.flatten()

    # Plot histograms of the predicted and actual IR bands
    plt.figure(figsize=(12, 6))
    plt.hist(y_pred_flat, bins=50, alpha=0.5, label="Predicted", color='blue')
    plt.hist(y_test_flat, bins=50, alpha=0.5, label="Actual", color='red')
    plt.legend()
    plt.title("Histogram Comparison: Predicted vs Actual")
    plt.xlabel("Intensity Value")
    plt.ylabel("Frequency")
    plt.show()

    return results

# Example Usage
# metrics = evaluate_predictions(y_test, y_pred)
# print(metrics)

metrics  = evaluate_predictions(y_array, y_pred)
metrics
