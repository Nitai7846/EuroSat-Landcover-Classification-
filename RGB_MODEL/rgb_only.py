#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 02:13:01 2024

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
import Image_Functions
from Image_Functions import *

tf.config.optimizer.set_experimental_options({"disable_meta_optimizer": True})

# Disable any GPU-related issues by forcing the model to use CPU or other settings
tf.config.set_visible_devices([], 'GPU')

# Check available devices
print("Available devices:", tf.config.list_physical_devices())


# Path to the dataset
dataset_dir = '/Users/nitaishah/Desktop/ECEN-PROJECT/EuroSAT_RGB/'  # Change this to your dataset path



class_labels = ["AnnualCrop", "Forest", "HerbaceousVegetation", "Highway", "Industrial",
                "Pasture", "PermanentCrop", "Residential", "River", "SeaLake"]


data = []


for label, class_name in enumerate(class_labels):
    class_dir = os.path.join(dataset_dir, class_name)
    
    # Loop through each image in the class directory
    for image_name in os.listdir(class_dir):
        image_path = os.path.join(class_dir, image_name)
        
        # Add image path and label to the data list
        data.append({"image_path": image_path, "label": label, "class_name": class_name})


df = pd.DataFrame(data)
print(df.head())


def read_image_as_array(file_path):
    with rasterio.open(file_path) as src:
    # Reading all 13 bands
       
        blue = src.read(1)                 # Band 2 (490 nm)
        green = src.read(2)                # Band 3 (560 nm)
        red = src.read(3)               # Band 4 (665 nm)
        
        # Optional: Stack into a single array of shape (13, H, W)
        multispectral_array = src.read(list(range(1, 4)))  # Reads all bands at once
        
        return multispectral_array





X = df[['image_path']]
y = df[['label']]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42)  # 0.25 x 0.8 = 0.2

X.iloc[0][0]
trial = read_image_as_array(X.iloc[0][0])
trial.shape
trial_vis = trial.transpose(2,1,0)
trial_vis.shape
plt.imshow(trial_vis)

def get_dataset_as_array(dataset):
    final_array = []
    for i in range(0, dataset.shape[0]):
        array = read_image_as_array(dataset.iloc[i,0])
        array = array / np.max(array)
        final_array.append(array)
    
    final_array = np.stack(final_array)
    return final_array

X_train_array = get_dataset_as_array(X_train)
X_val_array = get_dataset_as_array(X_val)
X_test_array = get_dataset_as_array(X_test)

y_train_array = y_train.to_numpy()
y_val_array = y_val.to_numpy()
y_test_array = y_test.to_numpy()

import tensorflow as tf
from tensorflow.keras import layers, models

X_train_array.shape

X_train_array = X_train_array.transpose(0, 2, 3, 1)  # Now shape will be (16200, 64, 64, 13)

# Do the same for validation or test data if you have them
X_val_array = X_val_array.transpose(0, 2, 3, 1)  # Make sure to apply the same transformation

from tensorflow.keras import layers, models

def create_model(input_shape, num_classes):
    model = models.Sequential()
    
    # Example architecture
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(layers.MaxPooling2D((2, 2)))
    
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.Flatten())
    
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(num_classes, activation='softmax'))  # Assuming multi-class classification
    
    return model

# Define input shape and number of classes
input_shape = (64, 64, 3)  # Height, Width, Channels
num_classes = len(np.unique(y_train_array))  # Get the number of unique classes

model = create_model(input_shape, num_classes)

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',  # Use 'categorical_crossentropy' if you one-hot encode
              metrics=['accuracy'])

history = model.fit(X_train_array, y_train_array,  # Ensure y_train_array matches the format
                    epochs=10,
                    batch_size=32,
                    validation_data=(X_val_array, y_val_array))


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

# Convert probabilities to class indices
predicted_classes = np.argmax(predictions, axis=1)

# If you want to see the first few predictions and their corresponding true labels:
for i in range(10):
    print(f"True label: {y_test_array[i]}, Predicted label: {predicted_classes[i]}")

model.save("/Users/nitaishah/Desktop/ECEN-PROJECT/model_rgb.h5")


evaluate_model(y_test_array, predicted_classes)


