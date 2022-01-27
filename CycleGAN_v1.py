import glob
import os
import time
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
from PIL import Image
from tensorflow.keras.preprocessing import image
from tensorflow_examples.models.pix2pix import pix2pix
import CG_model

# Set default values
height = 256
width = 256
channels = 3

# Path to the cartoon images (Real images)
# Dataset from kaggle: https://www.kaggle.com/kostastokis/simpsons-faces
cartoon_image_path = './Data/Cartoon/Cartoon_resized_256_256/*.png'
filelist_cartoon = glob.glob(cartoon_image_path)

# Path to the face images image (Base images)
base_image_path = './Data/OriginalFaces/Faces_resized_256_256/*.png'
filelist_base = glob.glob(base_image_path)

# Load real images (cartoon) and resize it, place it into numpy array
img_cartoon = np.array([np.array(Image.open(fname.replace('\\', '/'))) for fname in filelist_cartoon])

# Load base images (original faces) and resize it, place it into numpy array
img_base = np.array([np.array(Image.open(fname.replace('\\', '/'))) for fname in filelist_base])

# Normalizes data for real images (cartoon)
img_cartoon_norm = img_cartoon.reshape((img_cartoon.shape[0],) + (height, width, channels)).astype('float32') / 255.
np.random.shuffle(img_cartoon_norm)

# Normalizes data for real images (cartoon)
img_base_norm = img_base.reshape((img_base.shape[0],) + (height, width, channels)).astype('float32') / 255.
np.random.shuffle(img_base_norm)

# Split into train and test by 80 / 20
train_cartoon, test_cartoon = img_cartoon_norm[:int(0.8 * len(img_cartoon_norm)),:], img_cartoon_norm[int(0.8 * len(img_cartoon_norm)):,:]
train_base, test_base = img_base_norm[:int(0.8 * len(img_base_norm)),:], img_base_norm[int(0.8 * len(img_base_norm)):,:]

# Transform numpy arrays to tf.tensors
train_cartoon_tf = tf.data.Dataset.from_tensor_slices((train_cartoon))
train_base_tf = tf.data.Dataset.from_tensor_slices((train_base)) 
test_cartoon_tf = tf.data.Dataset.from_tensor_slices((train_cartoon))
test_base_tf = tf.data.Dataset.from_tensor_slices((train_base)) 

# Create batch datasets
train_cartoon_tf = train_cartoon_tf.batch(1)
train_base_tf = train_base_tf.batch(1)
test_cartoon_tf = test_cartoon_tf.batch(1)
test_base_tf = test_base_tf.batch(1)

# Create a EagerTensor of one image (to show progress later)
sample_cartoon = next(iter(train_cartoon_tf))
sample_base = next(iter(train_base_tf))

# TRAINING
CG_model.start_train(train_base_tf, train_cartoon_tf, sample_cartoon, sample_base)

# TESTING
# CG_model.start_test(test_base, test_cartoon)





