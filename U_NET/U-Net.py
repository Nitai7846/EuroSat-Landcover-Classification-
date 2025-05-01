#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 19:34:53 2024

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

# Create DataFrame
df = pd.DataFrame(data)
print(df.head())


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
    
def show_nir(blue, green, ir):
    nir_composite = np.dstack((ir, green, blue)) 
    nir_composite = nir_composite/np.max(nir_composite)
    plt.imshow(nir_composite)
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



# Prepare the dataset
X_array, y_array = get_dataset_as_array(df)

# Print shapes for verification
print(f"X_array shape: {X_array.shape}")
print(f"y_array shape: {y_array.shape}")

X_train, X_temp, y_train, y_temp = train_test_split(X_array, y_array, test_size=0.2, random_state=42)

# Second split: Split the 20% into 50% validation, 50% test (i.e., 10% each of the total dataset)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Print shapes to verify
print(f"Training set X shape: {np.array(X_train).shape}, y shape: {np.array(y_train).shape}")
print(f"Validation set X shape: {np.array(X_val).shape}, y shape: {np.array(y_val).shape}")
print(f"Test set X shape: {np.array(X_test).shape}, y shape: {np.array(y_test).shape}")

y_train = np.expand_dims(y_train, axis=-1)
y_val = np.expand_dims(y_val, axis=-1)
y_test = np.expand_dims(y_test, axis=-1)

def build_unet(input_shape):
    inputs = layers.Input(input_shape)

    # Downsampling path
    c1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    c1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c1)
    p1 = layers.MaxPooling2D((2, 2))(c1)

    c2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(p1)
    c2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(c2)
    p2 = layers.MaxPooling2D((2, 2))(c2)

    c3 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(p2)
    c3 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(c3)
    p3 = layers.MaxPooling2D((2, 2))(c3)

    # Bottleneck
    c4 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(p3)
    c4 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(c4)

    # Upsampling path
    u5 = layers.Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(c4)
    u5 = layers.concatenate([u5, c3])
    c5 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(u5)
    c5 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(c5)

    u6 = layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = layers.concatenate([u6, c2])
    c6 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(u6)
    c6 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(c6)

    u7 = layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = layers.concatenate([u7, c1])
    c7 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(u7)
    c7 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c7)

    outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(c7)

    model = models.Model(inputs, outputs)
    return model

input_shape = X_train[0].shape  # Shape of the input images
model = build_unet(input_shape)
input_shape

# Compile the model
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    batch_size=256,
    epochs=10
)

X_test.shape

y_pred = model.predict(X_test)
y_pred.shape
y_test.shape

mae = np.mean(np.abs(y_pred - y_test))
print("Mean Absolute Error:", mae)


diff = np.abs(y_pred[10] - y_test[10])  # Example with the first image
plt.imshow(diff.squeeze(), cmap='hot')
plt.title("Difference Map")
plt.colorbar()
plt.show()

plt.subplot(1, 2, 1)
plt.imshow(y_test[10].squeeze(), cmap='gray')
plt.title("Ground Truth")

plt.subplot(1, 2, 2)
plt.imshow(y_pred[10].squeeze(), cmap='gray')
plt.title("Prediction")
plt.show()

from scipy.stats import entropy

def histogram_intersection(h1, h2, bins=50):
    hist_1, _ = np.histogram(h1.ravel(), bins=bins, range=(0, 1))
    hist_2, _ = np.histogram(h2.ravel(), bins=bins, range=(0, 1))
    return np.minimum(hist_1, hist_2).sum() / hist_1.sum()

intersection = histogram_intersection(y_pred, y_test)
print("Histogram Intersection:", intersection)

import skimage
from skimage.metrics import structural_similarity as ssim

ssim_value = ssim(y_test.squeeze(), y_pred.squeeze(), data_range=y_test.max() - y_test.min())
print("SSIM:", ssim_value)

y_pred_flat = y_pred.flatten()
y_test_flat = y_test.flatten()

# Calculate the absolute difference between predictions and true values
difference_map = np.abs(y_pred_flat - y_test_flat).reshape(y_pred.shape)

difference_map[5]

# Visualize the difference map for one sample (since it's 2700, you can select a specific index)
plt.imshow(difference_map[1, :, :, 0], cmap='hot')
plt.colorbar()
plt.title("Difference Map (Sample 0)")
plt.show()

y_pred_flat = y_pred.flatten()
y_test_flat = y_test.flatten()

y_pred[0].shape

# Plot histograms of the predicted and actual IR bands
plt.figure(figsize=(12, 6))
plt.hist(y_pred_flat, bins=50, alpha=0.5, label="Predicted IR", color='blue')
plt.hist(y_test_flat, bins=50, alpha=0.5, label="Actual IR", color='red')
plt.legend()
plt.title("Histogram Comparison: Predicted vs Actual IR Bands")
plt.xlabel("Intensity Value")
plt.ylabel("Frequency")
plt.show()


def construct_ms(file_path):
    multispectral_array = read_image_as_array(file_path)
    red = multispectral_array[2]
    green = multispectral_array[1]
    blue = multispectral_array[0]
    ms = multispectral_array[3]
    rgb = np.dstack((red, green, blue))
    rgb = rgb / (np.max(rgb) + 1e-8)
    rgb = np.expand_dims(rgb, axis = 0)
    print(rgb.shape)
    y_pred = model.predict(rgb).reshape(64,64)
    print(y_pred.shape)
    print(ms.shape)
    show_nir(blue, green, ms)
    print("executed")
    show_nir(blue, green, y_pred)
    return ms/np.max(ms), y_pred

ir, y_pred_indi = construct_ms(df['image_path'][5])

ir.shape
y_pred_indi.shape

intersection = histogram_intersection(ir, y_pred_indi)
print("Histogram Intersection:", intersection)

plt.imshow(ir)
plt.imshow(y_pred_indi)

ssim_value = ssim(ir.squeeze(), y_pred_indi.squeeze(), data_range=ir.max() - ir.min())
print("SSIM:", ssim_value)

ir.max()
ir.min()
y_pred_indi.max()

model.save("U-NET-narrow-swir-2.h5")
