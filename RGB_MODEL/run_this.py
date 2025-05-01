#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 12:09:42 2024

@author: nitaishah
"""
from Image_Functions import *
from keras.models import load_model
import os
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

tf.config.optimizer.set_experimental_options({"disable_meta_optimizer": True})

# Disable any GPU-related issues by forcing the model to use CPU or other settings
tf.config.set_visible_devices([], 'GPU')

# Check available devices
print("Available devices:", tf.config.list_physical_devices())

def convert_image_to_array(image_path):
    array = read_image_as_array(image_path)
    array = array.transpose(1,2,0)
    array = array/np.max(array)
    array = np.expand_dims(array, axis=0) 
    return array

def display_results(array):
    red = array[2]
    green = array[1]
    blue = array[0]
    rgb = np.dstack((red, green, blue))
    rgb = rgb / np.max(rgb)
    plt.imshow(rgb)
    plt.axis('off')
    plt.title(f"Ground Truth: {ground_truth_class_name} | Predicted: {predicted_class_name}")
    plt.show()
    

model_path = "model_rgb.h5"
model = load_model(model_path)

test_images_folder = "images_test"

label_to_class_name = {0:"AnnualCrop", 1:"Forest", 2:"HerbaceousVegetation", 3:"Highway", 4:"Industrial",
                5:"Pasture", 6:"PermanentCrop", 7:"Residential", 8:"River", 9:"SeaLake"}

image_files = sorted(os.listdir(test_images_folder)) 
for idx, image_file in enumerate(image_files):
    image_path = os.path.join(test_images_folder, image_file)
    image_array = convert_image_to_array(image_path)

    predictions = model.predict(image_array)
    predicted_class = np.argmax(predictions)  

    ground_truth_class_name = label_to_class_name[idx]
    predicted_class_name = label_to_class_name[predicted_class]

    img_array = read_image_as_array(image_path)
    display_results(img_array)
  
        
        
        