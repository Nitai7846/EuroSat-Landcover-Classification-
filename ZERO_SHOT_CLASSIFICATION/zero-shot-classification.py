#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 17:13:53 2024

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

dataset_dir = '/Users/nitaishah/Desktop/ECEN-PROJECT/EuroSAT_RGB/'
class_labels = ["AnnualCrop", "Forest", "HerbaceousVegetation", "Highway", "Industrial",
                "Pasture", "PermanentCrop", "Residential", "River", "SeaLake"]

# Text prompts for each class
text_prompts = [f"An image of {label}" for label in class_labels]

# Function to sample 20% of images from each class
def sample_images(dataset_dir, class_labels, sample_fraction=0.2):
    sampled_data = []
    for label_idx, class_name in enumerate(class_labels):
        class_dir = os.path.join(dataset_dir, class_name)
        all_images = os.listdir(class_dir)
        sampled_images = random.sample(all_images, int(len(all_images) * sample_fraction))
        for image_name in sampled_images:
            sampled_data.append({
                "image_path": os.path.join(class_dir, image_name),
                "label": label_idx,
                "class_name": class_name
            })
    return sampled_data

# Sample 20% of images
sampled_data = sample_images(dataset_dir, class_labels)

# Function to preprocess and evaluate each image
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

# Run evaluation
true_labels, predictions = evaluate_clip_model(sampled_data, text_prompts)

# Evaluate the model's performance
accuracy = accuracy_score(true_labels, predictions)
precision = precision_score(true_labels, predictions, average='weighted')
recall = recall_score(true_labels, predictions, average='weighted')
f1 = f1_score(true_labels, predictions, average='weighted')

# Print evaluation metrics
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")

# Confusion Matrix
cm = sns.heatmap(
    confusion_matrix(true_labels, predictions),
    annot=True, fmt="d", cmap="Blues",
    xticklabels=class_labels, yticklabels=class_labels
)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()

#%%

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

# Function to sample 20% of images from each class
def sample_images(dataset_dir, class_labels, sample_fraction=0.2):
    sampled_data = []
    for label_idx, class_name in enumerate(class_labels):
        class_dir = os.path.join(dataset_dir, class_name)
        all_images = os.listdir(class_dir)
        sampled_images = random.sample(all_images, int(len(all_images) * sample_fraction))
        for image_name in sampled_images:
            sampled_data.append({
                "image_path": os.path.join(class_dir, image_name),
                "label": label_idx,
                "class_name": class_name
            })
    return sampled_data

# Sample 20% of images
sampled_data = sample_images(dataset_dir, class_labels)

# Function to preprocess and evaluate each image
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

# Evaluate the model's performance on the grouped classes
grouped_accuracy = accuracy_score(true_labels_grouped, predictions_grouped)
grouped_precision = precision_score(true_labels_grouped, predictions_grouped, average='weighted')
grouped_recall = recall_score(true_labels_grouped, predictions_grouped, average='weighted')
grouped_f1 = f1_score(true_labels_grouped, predictions_grouped, average='weighted')

# Print evaluation metrics for the grouped classes
print(f"Grouped Accuracy: {grouped_accuracy:.4f}")
print(f"Grouped Precision: {grouped_precision:.4f}")
print(f"Grouped Recall: {grouped_recall:.4f}")
print(f"Grouped F1 Score: {grouped_f1:.4f}")

# Confusion Matrix for Grouped Classes
cm_grouped = confusion_matrix(true_labels_grouped, predictions_grouped)
sns.heatmap(
    cm_grouped,
    annot=True, fmt="d", cmap="Blues",
    xticklabels=grouped_class_labels, yticklabels=grouped_class_labels
)
plt.title("Confusion Matrix for Grouped Classes")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()
