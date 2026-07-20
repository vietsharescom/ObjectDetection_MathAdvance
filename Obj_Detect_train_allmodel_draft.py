import os
import numpy as np
import cv2
import torch
from torchvision import transforms
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.models import image

data_path = 'D:\\Applied Math 2\\Group01_ObjectDetection\\Person Detection with head.v2i.yolov8\\data'

train_images_path = os.path.join(data_path, 'train/images')
test_images_path = os.path.join(data_path, 'test/images')
val_images_path = os.path.join(data_path, 'val/images')

