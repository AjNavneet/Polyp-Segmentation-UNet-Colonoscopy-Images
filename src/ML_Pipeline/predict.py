import cv2
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import imread

# Import the UNetPP network from a custom module
from ML_Pipeline.network import UNetPP
from argparse import ArgumentParser
from albumentations.augmentations import transforms
from albumentations import Resize
from albumentations.core.composition import Compose

# Define an image validation transform using Albumentations library
val_transform = Compose([
    Resize(256, 256),  # Resize images to 256x256
    transforms.Normalize(),  # Normalize the image
])

# Define a function to load and preprocess an image
def image_loader(image_name):
    # Read the image using matplotlib's imread
    img = imread(image_name)
    
    # Apply the defined validation transform
    img = val_transform(image=img)["image"]
    
    # Normalize the image to the range [0, 1]
    img = img.astype('float32') / 255
    
    # Transpose the image dimensions to match the expected format
    img = img.transpose(2, 0, 1)

    return img
