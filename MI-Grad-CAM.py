from tensorflow.keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input, decode_predictions
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from skimage import io
from scipy.stats import entropy
from sklearn.metrics import mutual_info_score
import cv2

model = VGG16(weights='imagenet', include_top=True, input_shape=(224, 224, 3))
img_path = "Grad-MI-CAM/Input Images/ostrich.jpg"    #change accordingly
img = image.load_img(img_path, target_size=(224, 224))

def smoothen_cam(cam, method, kernel_size=11, sigma=7):
  if method == 'gaussian':
    smoothed_cam = cv2.GaussianBlur(cam, (kernel_size, kernel_size), sigma)
  elif method == 'bilateral':
    smoothed_cam = cv2.bilateralFilter(cam.astype(np.float32), kernel_size, sigma, sigma)
  elif method == 'average':
    smoothed_cam = cv2.blur(cam, (kernel_size, kernel_size))
  else:
    raise ValueError(f"Unknown method '{method}'. Choose from 'gaussian', 'bilateral', or 'average'.")

  return smoothed_cam

def generate_grad_mi_cam(model, img, last_conv_layer_name, pred_index=None):
    grad_mi_model = tf.keras.Model(
        inputs=model.inputs,
        outputs=[model.get_layer(last_conv_layer_name).output, model.output]
    )
    img_array = np.expand_dims(image.img_to_array(img), axis=0)
    y = image.img_to_array(tf.image.rgb_to_grayscale(img))
    image_flattened = y.flatten()
    with tf.GradientTape() as tape:
        feature, predictions = grad_mi_model(img_array)
        class_channel = predictions[:, pred_index]
    input_shape = (14, 14, 512) #change according to the shape of the feature map generated from a particular layer.
    original_image_shape = (224, 224) #change according to input image shape.
    upsampling_factor = (original_image_shape[0] // input_shape[0], original_image_shape[1] // input_shape[1])
    upsample_layer = tf.keras.layers.UpSampling2D(size=upsampling_factor)
    upsampled_feature_map = upsample_layer(feature)
    upsampled_feature_map_1 = tf.image.resize(upsampled_feature_map, (224,224)) #featrue map resized to (224,224)

    grads = tape.gradient(class_channel, feature)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    pooled_grads_list = pooled_grads.numpy().tolist()
    mutual_information = []
    for i in range(upsampled_feature_map_1.shape[-1]):
      feature_map_flattened = np.array(upsampled_feature_map_1[:,:,:,i]).flatten()
      mutual_information.append(mutual_info_score(image_flattened, feature_map_flattened))
    mutual_information_array = np.array(mutual_information)
    mi_weights = mutual_information_array / np.sum(mutual_information_array)
    mi_weights_list = mi_weights.tolist()
    combined_weights = pooled_grads.numpy() * mi_weights
    combined_weights_list = combined_weights.tolist()
    final_output = np.zeros((1, 224, 224))
    for i in range(upsampled_feature_map_1.shape[-1]):
        final_output = final_output + (combined_weights_list[i] * np.array(upsampled_feature_map_1[:,:,:,i]))
    final_output_resized = np.squeeze(final_output)
    smoothed_cam = smoothen_cam(final_output_resized, 'gaussian', kernel_size=11, sigma=7)
    smoothed_cam = np.max(smoothed_cam, 0)  
    return smoothed_cam 
