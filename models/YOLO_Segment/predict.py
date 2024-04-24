from ultralytics import YOLO
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
import cv2

CLASSES = ["bicycle", "bridge", "bus", "car", "chimney", "crosswalk", "hydrant", "motorcycle", "other", "palm", "stairs", "traffic"]
"""{0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus',
      6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light', 10: 'fire hydrant', 11: 'stop sign',
        12: 'parking meter', 13: 'bench', 14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep',
          19: 'cow', 20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack',
            25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee', 30: 'skis',
              31: 'snowboard', 32: 'sports ball', 33: 'kite', 34: 'baseball bat', 35: 'baseball glove',
                36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39: 'bottle', 40: 'wine glass',
                  41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl', 46: 'banana', 47: 'apple',
                    48: 'sandwich', 49: 'orange', 50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza',
                      54: 'donut', 55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed',
                        60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote',
                          66: 'keyboard', 67: 'cell phone', 68: 'microwave', 69: 'oven', 70: 'toaster', 71: 'sink',
                            72: 'refrigerator', 73: 'book', 74: 'clock', 75: 'vase', 76: 'scissors',
                              77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'}
"""
MAPPING = [(1, 0), (5, 2), (2, 3), (10, 6), (3,7), (9, 11)] # mapping from COCO to captcha classes
# Load a pretrained YOLOv8n model
model = YOLO('yolov8n-seg.pt')



def predict(class_number, image_path):
  class_id_searched = None
  for map in MAPPING:
    if map[1] == class_number:
      class_id_searched = map[0]
      break
  
  img = Image.open(image_path)

  # Get the image's width and height
  image_width, image_height = img.size

  results = model.predict(image_path, save=False, imgsz=320, conf=0.5, visualize=False)

  # Define the grid
  grid_size = 4
  grid = np.zeros((grid_size, grid_size))

  # Initialize a variable to keep track of whether the searched class was found
  class_found = False

  # Check which grid cells the bounding boxes fall into
  for result in results:
    #print(result)
    for i, class_id in enumerate(result.boxes.cls):
      class_id = int(class_id.item())
      if class_id_searched == class_id:
        class_found = True

        # Get the class name
        class_name = result.names[class_id]

        #print(f"Predicted class: {class_name}")

        # Get the mask
        mask_segments = result.masks[i].xy

        # Create an empty mask
        mask = np.zeros((image_height, image_width))

        # Fill in the mask
        for segment in mask_segments:
            segment = np.array(segment, dtype=np.int32)
            cv2.fillPoly(mask, [segment], 255)

        cell_height = mask.shape[0] / grid_size
        cell_width = mask.shape[1] / grid_size
        # Iterate over each grid cell
        for i in range(grid_size):
            for j in range(grid_size):
                # Calculate the pixel coordinates of the corners of the grid cell
                top = int(i * cell_height)
                bottom = int((i + 1) * cell_height)
                left = int(j * cell_width)
                right = int((j + 1) * cell_width)

                # Extract the part of the mask that falls into the grid cell
                cell_mask = mask[top:bottom, left:right]

                # If any pixel in the cell is part of the mask, mark the grid cell as 1
                if np.any(cell_mask == 255):
                    grid[i, j] = 1

  return class_found, grid
    