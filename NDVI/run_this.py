#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 15:09:08 2024

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
import keras
from keras import layers, models
import random

tf.config.optimizer.set_experimental_options({"disable_meta_optimizer": True})

# Disable any GPU-related issues by forcing the model to use CPU or other settings
tf.config.set_visible_devices([], 'GPU')

# Check available devices
print("Available devices:", tf.config.list_physical_devices())

dataset_dir = 'images_test/'  # Change this to your dataset path



class_labels = ["AnnualCrop", "Forest", "HerbaceousVegetation", "Highway", "Industrial",
                "Pasture", "PermanentCrop", "Residential", "River", "SeaLake"]


data = []

def read_image_as_array(file_path):
    with rasterio.open(file_path) as src:
    # Reading all 13 bands
        coastal_aerosol = src.read(1)      # Band 1 (443 nm)
        blue = src.read(2)                 # Band 2 (490 nm)
        green = src.read(3)                # Band 3 (560 nm)
        red = src.read(4)               # Band 4 (665 nm)
        vegetation_red_edge_1 = src.read(5)  # Band 5 (705 nm)
        vegetation_red_edge_2 = src.read(6)  # Band 6 (740 nm)
        vegetation_red_edge_3 = src.read(7)  # Band 7 (783 nm)
        nir = src.read(8)                  # Band 8 (842 nm)
        narrow_nir = src.read(9)           # Band 8A (865 nm)
        water_vapor = src.read(10)         # Band 9 (945 nm)
        cirrus = src.read(11)              # Band 10 (1375 nm)
        swir1 = src.read(12)               # Band 11 (1610 nm)
        swir2 = src.read(13)               # Band 12 (2190 nm)
    
        # Optional: Stack into a single array of shape (13, H, W)
        multispectral_array = src.read(list(range(1, 14)))  # Reads all bands at once
        
        return multispectral_array


def sample_images(dataset_dir, class_labels):
    sampled_data = []
    for label_idx, class_name in enumerate(class_labels):
        class_dir = os.path.join(dataset_dir, class_name)
        all_images = os.listdir(class_dir)
        sampled_images = random.sample(all_images, int(len(all_images)))
        for image_name in sampled_images:
            sampled_data.append({
                "image_path": os.path.join(class_dir, image_name),
                "label": label_idx,
                "class_name": class_name
            })
    return sampled_data

# Sample 20% of images
sampled_data = sample_images(dataset_dir, class_labels)
df_sampled = pd.DataFrame(sampled_data)

from keras.models import load_model
model_path = "model_1.h5"
model = load_model(model_path)

def predict_classes(model, dataset):
    predictions = []
    for i, row in dataset.iterrows():
        image_array = read_image_as_array(row['image_path'])  # Load image as array
        image_array = image_array / np.max(image_array)
        image_array = np.expand_dims(image_array, axis=0)
        image_array = image_array.transpose(0,2,3,1)# Add batch dimension
        predicted_class = np.argmax(model.predict(image_array), axis=1)[0]  # Predict class
        predictions.append(predicted_class)
    return predictions

# Predict classes for sampled data
df_sampled['predicted_class'] = predict_classes(model, df_sampled)

def calculate_ndvi(array):
    if array.shape[0] < 5:
        raise ValueError("Input array must have at least 5 bands for NDVI calculation.")
    
    nir = array[8].astype(float)
    red = array[3].astype(float)
    ndvi = (nir - red) / (nir + red)
    mean_ndvi = np.nanmean(ndvi)
    return mean_ndvi


df_sampled_ndvi = df_sampled.copy()


df_sampled_ndvi['NDVI'] = df_sampled_ndvi['image_path'].apply(lambda x: calculate_ndvi(read_image_as_array(x)))

df_sampled_ndvi['NDVI']

print(df_sampled_ndvi.columns)

ndvi_by_class = df_sampled_ndvi.groupby('class_name')['NDVI'].mean().reset_index()

plt.figure(figsize=(10, 6))
sns.barplot(x='class_name', y='NDVI', data=ndvi_by_class, palette='viridis')

# Step 3: Customize the plot (optional)
plt.title('Average NDVI by Class')
plt.xlabel('Class')
plt.ylabel('Average NDVI')
plt.xticks(rotation=45, ha='right')  # Rotate x labels for better readability
plt.tight_layout()

# Step 4: Show the plot
plt.show()

