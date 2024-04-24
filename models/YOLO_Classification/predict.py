from ultralytics import YOLO
import torch
from torchvision import transforms
from PIL import Image
import shap
from shap import KernelExplainer, force_plot
import numpy as np
import cv2
import os

# Get the directory of the current script
script_dir = os.path.dirname(os.path.realpath(__file__))

# Construct the path to the best.pt file
model_path = os.path.join(script_dir, 'train4', 'weights', 'best.pt')

model = YOLO(model_path)

def predict_tile(tile_path):
    # Load the image
    tile = Image.open(tile_path)

    # Convert the image to RGB mode and resize it
    tile = tile.convert("RGB").resize((128, 128))

    # Convert the image to a tensor and add a batch dimension
    to_predict = transforms.ToTensor()(tile).unsqueeze(0)

    # Make a prediction
    results = model.predict(to_predict, verbose=False)

    # Get the first Result object in the list
    result = results[0]

    # Get the class names
    class_names = result.names

    # Get the probabilities
    probabilities = result.probs.data

    # Get the class with the highest probability
    max_prob_index = result.probs.top1
    max_prob_class_name = class_names[max_prob_index]

    # Return a list of the prediction array, the class name with highest probability and its index
    return [probabilities, max_prob_class_name, max_prob_index]

    






