import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from tensorflow.keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input, decode_predictions
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from skimage import io
from scipy.stats import entropy
from sklearn.metrics import mutual_info_score
from MI-Grad-CAM import smoothen_cam
from MI-Grad-CAM import generate_grad_mi_cam

########  FUNCTION TO GET THE GROUND TRUTH BOUNDING BOX #########
def parse_annotation(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    # Extract the bounding box coordinates
    for obj in root.findall('object'):
        xmin = int(obj.find('bndbox/xmin').text)
        ymin = int(obj.find('bndbox/ymin').text)
        xmax = int(obj.find('bndbox/xmax').text)
        ymax = int(obj.find('bndbox/ymax').text)

        return xmin, ymin, xmax, ymax

###### FUNCTION TO GET THE CAM BOUNDING BOX ########
def get_cam_bounding_box(cam, threshold=0.5):
    # Normalize CAM to range [0, 1]
    cam = (cam - cam.min()) / (cam.max() - cam.min())

    # Threshold the CAM to create a binary mask
    cam_binary = cam > threshold

    # Get the bounding box of the active region in CAM
    indices = np.argwhere(cam_binary)
    ymin, xmin = indices.min(axis=0)
    ymax, xmax = indices.max(axis=0)

    return xmin, ymin, xmax, ymax

#######  POINTING GAME  ############
def pointing_game(cam, ground_truth_mask, threshold=0.5):
    # Step 1: Normalize CAM
    cam_normalized = (cam - cam.min()) / (cam.max() - cam.min())

    # Step 2: Threshold CAM
    thresholded_cam = cam_normalized > threshold

    # Step 3: Find maximum energy point
    max_point = np.unravel_index(np.argmax(cam_normalized), cam_normalized.shape)

    # Step 4: Check if max point is inside ground truth bounding box
    hit = ground_truth_mask[max_point]

    return hit

#######  GROUND TRUTH MASK  ############
def get_ground_truth_mask(image_shape, bbox):
    """
    Args:
    - image_shape: Shape of the input image (height, width).
    - bbox: Bounding box in the format (x_min, y_min, x_max, y_max).

    Returns:
    - ground_truth_mask: Binary mask with 1 inside the bounding box and 0 elsewhere.
    """
    mask = np.zeros(image_shape[:2], dtype=np.uint8)  # Assuming single-channel mask

    # Extract bounding box coordinates
    x_min, y_min, x_max, y_max = bbox

    # Set the region inside the bounding box to 1
    mask[y_min:y_max, x_min:x_max] = 1

    return mask

model = VGG16(weights='imagenet', include_top=True, input_shape=(224, 224, 3))
folder_path = '/MI-Grad-CAM/Input Images'       #change accordingly
annotation_path = '/MI-Grad-CAM/Input Annotations'

EBPG = []
hit_total = []
model1 = VGG16(weights='imagenet', include_top=True, input_shape=(224, 224, 3))
for filename1, filename2 in zip(os.listdir(folder_path), os.listdir(annotation_path)):
  # Get input image
  image_path = os.path.join(folder_path, filename1)
  # Get Input annotations
  annotation = os.path.join(annotation_path, filename2)
  img = image.load_img(image_path, target_size=(224, 224))
  # Resize input image to that of the ground truth annotations
  img2 = image.load_img(image_path, target_size=(375, 500))
  x = image.img_to_array(img)
  x = np.expand_dims(x, axis=0)
  pred = model1.predict(x)
  pred_class = np.argmax(pred)
  # Get CAM
  cam = generate_grad_mi_cam(model, img, 'block5_conv3', pred_class)
  cam = cv2.resize(smooth_cam, (500, 375))
  # Get ground truth mask
  ground_truth_mask = get_ground_truth_mask(np.squeeze(np.array(img2)).shape, parse_annotation(annotation))
  # Get pointing game results
  hit = pointing_game(cam, ground_truth_mask)
  hit_total.append(hit)
  # Get energy based pointing game results
  result = ground_truth_mask * cam
  proportion = np.sum(result)/np.sum(cam)
  EBPG.append(proportion)


Avg_Energy_Based_Pointing_Game = sum(EBPG) / len(EBPG)
print(Avg_Energy_Based_Pointing_Game)

Pointing_Game_Hit_Rate = sum(hit_total) / len(hit_total)
print(Pointing_Game_Hit_Rate)
