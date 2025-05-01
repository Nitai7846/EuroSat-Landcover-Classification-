#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 01:04:26 2024

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

print(tf.__version__)

tf.config.optimizer.set_experimental_options({"disable_meta_optimizer": True})

# Disable any GPU-related issues by forcing the model to use CPU or other settings
tf.config.set_visible_devices([], 'GPU')

# Check available devices
print("Available devices:", tf.config.list_physical_devices())

dataset_dir = '/Users/nitaishah/Desktop/ECEN-PROJECT/EuroSAT_RGB/'  # Change this to your dataset path

# Class labels for the dataset
class_labels = ["AnnualCrop", "Forest", "HerbaceousVegetation", "Highway", "Industrial",
                "Pasture", "PermanentCrop", "Residential", "River", "SeaLake"]

# Function to read an image as a 3-band array (RGB)
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


def divide_image_into_tiles(image_array, tile_size=32):
    """
    Divide an image array into smaller tiles of the specified size.
    
    Args:
        image_array (np.ndarray): The input image array of shape (C, H, W).
        tile_size (int): The size of each tile (default: 16).
        
    Returns:
        list: A list of smaller tiles, each of shape (C, tile_size, tile_size).
    """
    _, height, width = image_array.shape
    tiles = []
    for i in range(0, height, tile_size):
        for j in range(0, width, tile_size):
            tile = image_array[:, i:i+tile_size, j:j+tile_size]
            if tile.shape[1] == tile_size and tile.shape[2] == tile_size:  # Ensure complete tiles
                tiles.append(tile)
    return tiles


def normalize_and_stack(dataset):
    """
    Normalize and stack image arrays into a single numpy array.
    
    Args:
        dataset (pd.DataFrame): Dataset with 'image_array' column containing image tiles.
        
    Returns:
        np.ndarray: Normalized and stacked array of all images.
    """
    image_arrays = []
    for img in dataset['image_array']:
        normalized_img = img / np.max(img)  # Normalize the image
        image_arrays.append(normalized_img)
    return np.stack(image_arrays)



# Function to read an image as a 3-band array (RGB)
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

data = []


for label, class_name in enumerate(class_labels):
    class_dir = os.path.join(dataset_dir, class_name)
    
    for image_name in os.listdir(class_dir):
        image_path = os.path.join(class_dir, image_name)
        
        # Read image and process tiles
        image_array = read_image_as_array(image_path)
        if image_array is not None and image_array.shape[1:] == (64, 64):  # Ensure 64x64 images
            tiles = divide_image_into_tiles(image_array)  # Divide into 16x16 tiles
            
            # Add each tile to the dataset
            for tile in tiles:
                data.append({"image_array": tile, "label": label, "class_name": class_name})

df = pd.DataFrame(data)
df

X = df[['image_array']]
y = df[['label']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42)

X_train_array = normalize_and_stack(X_train)
X_val_array = normalize_and_stack(X_val)
X_test_array = normalize_and_stack(X_test)

y_train_array = y_train.to_numpy()
y_val_array = y_val.to_numpy()
y_test_array = y_test.to_numpy()

print("Training data shape:", X_train_array.shape)
print("Validation data shape:", X_val_array.shape)
print("Test data shape:", X_test_array.shape)

# Visualize a sample tile
tile_sample = X_train_array[1000].transpose(1, 2, 0)  # Convert (C, H, W) to (H, W, C) for visualization
plt.imshow(tile_sample)
plt.title("Sample 16x16 Tile")
plt.show()

X_train_array = X_train_array.transpose(0, 2, 3, 1)  # Shape: (num_samples, 16, 16, 3)
X_val_array = X_val_array.transpose(0, 2, 3, 1)      # Shape: (num_samples, 16, 16, 3)

X_train_array.shape

def create_model(input_shape, num_classes):
    model = models.Sequential()
    
    # First convolutional block
    model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape))
    model.add(layers.MaxPooling2D((2, 2)))
    
    # Second convolutional block
    model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2, 2)))
    
    # Third convolutional block
    model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2, 2)))  # Keep this if dimensions allow; otherwise, remove it
    
    # Flatten and Fully Connected Layers
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(num_classes, activation='softmax'))  # Assuming multi-class classification
    
    return model

# Define the new input shape and number of classes
input_shape = (32, 32, 3)  # For 16x16 tiles
num_classes = len(np.unique(y_train))  # Adjust based on actual data format

# Create the modified model
model = create_model(input_shape, num_classes)

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',  # Use 'categorical_crossentropy' if you one-hot encode
              metrics=['accuracy'])

# Train the model
history = model.fit(X_train_array, y_train.to_numpy(),  # Convert y_train to numpy if it's a DataFrame
                    epochs=15,
                    batch_size=32,
                    validation_data=(X_val_array, y_val.to_numpy()))

def evaluate_model(y_truth, y_label):
    
    accuracy = accuracy_score(y_truth, y_label)
    
    
    precision = precision_score(y_truth, y_label, average='micro')
    recall = recall_score(y_truth, y_label, average='micro')
    f1 = f1_score(y_truth, y_label, average='micro')
    
    # Print the metrics
    print(f'Accuracy: {accuracy:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1 Score: {f1:.4f}')
    
    cm = confusion_matrix(y_truth.flatten(), y_label.flatten())
    
    unique_class_names = df['class_name'].unique()

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=unique_class_names,
        yticklabels=unique_class_names
    )
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix with Class Names')
    plt.show()   


import matplotlib.pyplot as plt

# Plot training & validation accuracy values
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend()
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()
plt.show()

X_test_array = X_test_array.transpose(0, 2, 3, 1)

test_loss, test_accuracy = model.evaluate(X_test_array, y_test_array)

print(f'Test Loss: {test_loss:.4f}')
print(f'Test Accuracy: {test_accuracy:.4f}')

predictions = model.predict(X_test_array)

predicted_classes = np.argmax(predictions, axis=1)

# If you want to see the first few predictions and their corresponding true labels:
for i in range(10):
    print(f"True label: {y_test_array[i]}, Predicted label: {predicted_classes[i]}")
    
model.save("/Users/nitaishah/Desktop/ECEN-PROJECT/Segment_GeoFS/model_rgb_32*32_2.h5")

evaluate_model(y_test_array, predicted_classes)
