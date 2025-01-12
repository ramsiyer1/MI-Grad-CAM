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

#####   FUNCTION FOR OBTAINING THE MASKED INPUT  #######
def apply_mask(image, cam_mask, threshold=0.5):
    # Normalize the CAM mask to the range [0, 1] (if not already)
    cam_mask = (cam_mask - cam_mask.min()) / (cam_mask.max() - cam_mask.min())

    # Expand cam_mask to have the same number of channels as the image
    cam_mask = np.expand_dims(cam_mask, axis=-1)
    cam_mask = np.repeat(cam_mask, image.shape[-1], axis=-1)  # repeat the mask for each channel

    # Create the mask where cam_mask values below the threshold are set to 0
    masked_image = np.where(cam_mask < threshold, 0, image)
    # Create the baseline image
    baseline_image = np.where(cam_mask > 0, 0, image)
    return masked_image, baseline_image

model = VGG16(weights='imagenet', include_top=True, input_shape=(224, 224, 3))
folder_path = '/MI-Grad-CAM/Input Images'       #change accordingly

average_drops = []
average_increases = []

for filename in os.listdir(folder_path):
    image_path = os.path.join(folder_path, filename)
    img = image.load_img(image_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    pred = model.predict(x)
    pred_class = np.argmax(pred)
    #original probabilities
    original_prob = pred[:,pred_class]
    #get mi_grad_cam
    cam = generate_grad_mi_cam(model, img, 'block5_conv3', pred_class)
    #get masked image and baseline image
    masked_image, baseline_image = apply_mask(np.array(img), cam)
    #get masked probabilities
    masked_pred = model.predict(np.expand_dims(masked_image, axis=0))
    masked_prob = masked_pred[:,pred_class]
    #get baseline probabilities
    baseline_pred = model.predict(np.expand_dims(baseline_image, axis=0))
    baseline_prob = baseline_pred[:,pred_class]
    # average drop and average increase function
    drop = (original_prob - masked_prob) / original_prob * 100
    if drop < 0:
        drop = 0
    average_drops.append(drop)
    increase = (masked_prob - baseline_prob) / masked_prob * 100
    if increase < 0:
        increase = 0
    average_increases.append(increase)
    print("drop:", drop)
    print("increase:", increase)
    
total_drop = sum(average_drops)
total_count = len(average_drops)
average_drop = total_drop / total_count
print("Average drop:", average_drop)

total_increase = sum(average_increases)
total_count_1 = len(average_increases)
average_increase = total_increase / total_count_1
print("Average increase:", average_increase)
