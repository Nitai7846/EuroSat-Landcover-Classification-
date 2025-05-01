#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 13:48:58 2024

@author: nitaishah
"""

from transformers import CLIPProcessor, CLIPModel
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import random
from sklearn.metrics import confusion_matrix


# Load the CLIP model and processor from Hugging Face
model_name = "openai/clip-vit-base-patch32"
model = CLIPModel.from_pretrained(model_name)
processor = CLIPProcessor.from_pretrained(model_name)

print("Model and processor loaded successfully!")

from PIL import Image

dataset_dir = 'images_test'
class_labels = ["AnnualCrop", "Forest", "HerbaceousVegetation", "Highway", "Industrial",
                "Pasture", "PermanentCrop", "Residential", "River", "SeaLake"]

text_prompts = [f"An image of {label}" for label in class_labels]

def sample_images(dataset_dir, class_labels):
    sampled_data = []
    for label_idx, class_name in enumerate(class_labels):
        class_dir = os.path.join(dataset_dir, class_name)
        all_images = os.listdir(class_dir)
        for image_name in all_images:
            sampled_data.append({
                "image_path": os.path.join(class_dir, image_name),
                "label": label_idx,
                "class_name": class_name
            })
    return sampled_data

# Sample 20% of images
sampled_data = sample_images(dataset_dir, class_labels)

def evaluate_clip_model(sampled_data, text_prompts):
    image_predictions = []
    true_labels = []
    
    for item in sampled_data:
        # Load and preprocess the image
        image = Image.open(item['image_path']).convert("RGB")
        inputs = processor(text=text_prompts, images=image, return_tensors="pt", padding=True)
        
        # Run inference
        outputs = model(**inputs)
        logits_per_image = outputs.logits_per_image  # Image-text similarity scores
        predicted_idx = logits_per_image.argmax().item()
        
        # Store prediction and ground truth
        image_predictions.append(predicted_idx)
        true_labels.append(item['label'])
    
    return true_labels, image_predictions

true_labels, predictions = evaluate_clip_model(sampled_data, text_prompts)

print(true_labels)
print(predictions)


grouped_class_labels = ["Vegetation", "Built Areas", "Water Bodies"]

# Define text prompts for the grouped classes
grouped_text_prompts = [
    "An image of vegetation",   # Group 1
    "An image of built-up areas",  # Group 2
    "An image of water bodies"  # Group 3
]

# Create a mapping from original class indices to grouped class indices
group_mapping = {
    0: 0,  # Annual Crop -> Vegetation
    1: 0,  # Forest -> Vegetation
    2: 0,  # Herbaceous Vegetation -> Vegetation
    3: 1,  # Highway -> Built Areas
    4: 1,  # Industrial -> Built Areas
    5: 0,  # Pasture -> Vegetation
    6: 0,  # Permanent Crop -> Vegetation
    7: 1,  # Residential -> Built Areas
    8: 2,  # River -> Water Bodies
    9: 2   # SeaLake -> Water Bodies
}

def evaluate_clip_model_grouped(sampled_data, grouped_text_prompts, group_mapping):
    image_predictions = []
    true_labels = []
    
    for item in sampled_data:
        # Load and preprocess the image
        image = Image.open(item['image_path']).convert("RGB")
        inputs = processor(text=grouped_text_prompts, images=image, return_tensors="pt", padding=True)
        
        # Run inference
        outputs = model(**inputs)
        logits_per_image = outputs.logits_per_image  # Image-text similarity scores
        predicted_idx = logits_per_image.argmax().item()
        
        # Store grouped prediction and ground truth
        image_predictions.append(predicted_idx)
        true_labels.append(group_mapping[item['label']])  # Map to grouped labels
    
    return true_labels, image_predictions

# Run evaluation for grouped classes
true_labels_grouped, predictions_grouped = evaluate_clip_model_grouped(sampled_data, grouped_text_prompts, group_mapping)

print(true_labels_grouped)
print(predictions_grouped)
