from django.shortcuts import render
from rest_framework import generics
from .models import Masking
from .serializers import MaskingSerializer

import os
import sys
import math
import re
import skimage.io
import numpy as np
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Root directory of the project
ROOT_DIR = os.path.abspath("C:\Mask_RCNN-master")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
from mrcnn import visualize
from mrcnn.visualize import display_images
import mrcnn.model as modellib
from mrcnn.model import log

import food_v4
import keras
from keras import backend
keras.backend.clear_session()
#%matplotlib inline 
#from IPython import get_ipython
#get_ipython().run_line_magic('matplotlib', 'inline')

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")
IMAGE_DIR = os.path.join(ROOT_DIR, "images")
# Path to Ballon trained weights
# You can download this file from the Releases page
# https://github.com/matterport/Mask_RCNN/releases
FOOD_WEIGHTS_PATH = "C:\Mask_RCNN-master\mask_rcnn_food_0030.h5"  # TODO: update this path

config = food_v4.FoodConfig()
FOOD_DIR = os.path.join(ROOT_DIR, "C:\Mask_RCNN-master\Food")

# Override the training configurations with a few
# changes for inferencing.
class InferenceConfig(food_v4.FoodConfig):
    # Run detection on one image at a time
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

config = InferenceConfig()
config.display()

# Device to load the neural network on.
# Useful if you're training a model on the same 
# machine, in which case use CPU and leave the
# GPU for training.
DEVICE = "/cpu:0"  # /cpu:0 or /gpu:0

# Inspect the model in training or inference modes
# values: 'inference' or 'training'
# TODO: code for 'training' test mode not ready yet
TEST_MODE = "inference"

def get_ax(rows=1, cols=1, size=16):
    """Return a Matplotlib Axes array to be used in
    all visualizations in the notebook. Provide a
    central point to control graph sizes.
    
    Adjust the size attribute to control how big to render images
    """
    _, ax = plt.subplots(rows, cols, figsize=(size*cols, size*rows))
    return ax

    # Load dataset
if config.NAME == 'shapes':
    dataset = shapes.ShapesDataset()
    #dataset.load_coco(FOOD_DIR, "train")
    dataset.load_shapes(500, config.IMAGE_SHAPE[0], config.IMAGE_SHAPE[1])
elif config.NAME == "food":
    dataset = food_v4.FoodDataset()
    dataset.load_food(FOOD_DIR, "train")

# Must call before using the dataset
dataset.prepare()

print("Image Count: {}".format(len(dataset.image_ids)))
print("Class Count: {}".format(dataset.num_classes))
for i, info in enumerate(dataset.class_info):
    print("{:3}. {:50}".format(i, info['name']))

    # Create model in inference mode
with tf.device(DEVICE):
    model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR,
                              config=config)

                              # Set path to balloon weights file

# Download file from the Releases page and set its path
# https://github.com/matterport/Mask_RCNN/releases
# weights_path = "/path/to/mask_rcnn_balloon.h5"

# Or, load the last model you trained
weights_path = "C:\Mask_RCNN-master\mask_rcnn_food_0030.h5"

# Load weights
print("Loading weights ", weights_path)
model.load_weights(FOOD_WEIGHTS_PATH, by_name=True)

class_names = ['BG', 'salad', 'bibimbab', 'sushi', 'dounut']

#Load a random image from the images folder
filename = os.path.join(IMAGE_DIR, 'KakaoTalk_20210225_203055649.jpg')
image= skimage.io.imread(filename)


# Run object detection
#results = model.detect([image], verbose=1)

# Display results
#ax = get_ax(1)
#r = results[0]
#visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], 
#                            dataset.class_names, r['scores'], ax=ax,
#                            title="Predictions")

# Get input and output to classifier and mask heads.
mrcnn = model.run_graph([image], [
    ("proposals", model.keras_model.get_layer("ROI").output),
    ("probs", model.keras_model.get_layer("mrcnn_class").output),
    ("deltas", model.keras_model.get_layer("mrcnn_bbox").output),
    ("masks", model.keras_model.get_layer("mrcnn_mask").output),
    ("detections", model.keras_model.get_layer("mrcnn_detection").output),
])
det_class_ids = mrcnn['detections'][0, :, 4].astype(np.int32)
det_count = np.where(det_class_ids == 0)[0][0]
det_class_ids = det_class_ids[:det_count]
detections = mrcnn['detections'][0, :det_count]

# i = 0
# B = False
# while i <= det_count:
#     if i == det_count:
#         class GIList(generics.ListCreateAPIView):
#             queryset = Masking.objects.filter(GI = 1000)
#             serializer_class = MaskingSerializer
#         break
#     for j in range(2, 13):
#         fff = Masking.objects.get(id = j)
#         if np.array(dataset.class_names)[det_class_ids][i] == fff.name and fff.GI > 55:
#             B = True
#             if fff.GI_A == '1':
#                 class GIList(generics.ListCreateAPIView):
#                     queryset = Masking.objects.filter(name__startswith=fff.name) | Masking.objects.filter(GI__lt = 55).filter(GI_A = 1).order_by('?')[:2]
#                     serializer_class = MaskingSerializer
#                 break
#             elif fff.GI_A == '2':
#                 class GIList(generics.ListCreateAPIView):
#                     queryset = Masking.objects.filter(name__startswith=fff.name) | Masking.objects.filter(GI__lt = 55).filter(GI_A = 2).order_by('?')[:2]
#                     serializer_class = MaskingSerializer
#                 break
#     if B == True:
#         break
#     i = i + 1


i = 0
k = 0
if det_count == 1:
    for j in range(2, 13):
        fff = Masking.objects.get(id = j)
        if np.array(dataset.class_names)[det_class_ids][0] == fff.name and fff.GI > 55:
            if fff.GI_A == '1':
                class GIList1(generics.ListCreateAPIView):
                    queryset = Masking.objects.filter(name__startswith=fff.name) | Masking.objects.filter(GI__lt = 55).filter(GI_A = 1).order_by('?')[:2]
                    serializer_class = MaskingSerializer
                break
            elif fff.GI_A == '2':
                class GIList1(generics.ListCreateAPIView):
                    queryset = Masking.objects.filter(name__startswith=fff.name) | Masking.objects.filter(GI__lt = 55).filter(GI_A = 2).order_by('?')[:2]
                    serializer_class = MaskingSerializer
                break
    if np.array(dataset.class_names)[det_class_ids][0] == fff.name and fff.GI < 55:
        class GIList1(generics.ListCreateAPIView):
            queryset = Masking.objects.filter(name__startswith=fff.name) | Masking.objects.filter(GI = 1000)
            serializer_class = MaskingSerializer
    k = k + 1

if det_count == 2:
    if i == 0:
        for j in range(2, 13):
            fff = Masking.objects.get(id = j)
            if np.array(dataset.class_names)[det_class_ids][i] == fff.name and fff.GI > 55:
                if fff.GI_A == '1':
                    class GIList1(generics.ListCreateAPIView):
                        queryset = Masking.objects.filter(name__startswith=fff.name) | Masking.objects.filter(GI__lt = 55).filter(GI_A = 1).order_by('?')[:2]
                        serializer_class = MaskingSerializer
                    break
                elif fff.GI_A == '2':
                    class GIList1(generics.ListCreateAPIView):
                        queryset = Masking.objects.filter(name__startswith=fff.name) | Masking.objects.filter(GI__lt = 55).filter(GI_A = 2).order_by('?')[:2]
                        serializer_class = MaskingSerializer
                    break
            if np.array(dataset.class_names)[det_class_ids][i] == fff.name and fff.GI < 55:
                class GIList1(generics.ListCreateAPIView):
                    queryset = Masking.objects.filter(name__startswith=fff.name) | Masking.objects.filter(GI = 1000)
                    serializer_class = MaskingSerializer
    i = i + 1
    k = k + 1
    if i == 1:
        for j in range(2, 13):
            fff = Masking.objects.get(id = j)
            if np.array(dataset.class_names)[det_class_ids][i] == fff.name and fff.GI > 55:
                if fff.GI_A == '1':
                    class GIList2(generics.ListCreateAPIView):
                        queryset = Masking.objects.filter(name__startswith=fff.name) | Masking.objects.filter(GI__lt = 55).filter(GI_A = 1).order_by('?')[:2]
                        serializer_class = MaskingSerializer
                    break
                elif fff.GI_A == '2':
                    class GIList2(generics.ListCreateAPIView):
                        queryset = Masking.objects.filter(name__startswith=fff.name) | Masking.objects.filter(GI__lt = 55).filter(GI_A = 2).order_by('?')[:2]
                        serializer_class = MaskingSerializer
                    break
            if np.array(dataset.class_names)[det_class_ids][i] == fff.name and fff.GI < 55:
                class GIList2(generics.ListCreateAPIView):
                    queryset = Masking.objects.filter(name__startswith=fff.name) | Masking.objects.filter(GI = 1000)
                    serializer_class = MaskingSerializer
    i = i + 1
    k = k + 1

if det_count == 3:
    if i == 0:
        for j in range(2, 13):
            fff = Masking.objects.get(id = j)
            if np.array(dataset.class_names)[det_class_ids][i] == fff.name and fff.GI > 55:
                if fff.GI_A == '1':
                    class GIList1(generics.ListCreateAPIView):
                        queryset = Masking.objects.filter(name__startswith=fff.name) | Masking.objects.filter(GI__lt = 55).filter(GI_A = 1).order_by('?')[:2]
                        serializer_class = MaskingSerializer
                    break
                elif fff.GI_A == '2':
                    class GIList1(generics.ListCreateAPIView):
                        queryset = Masking.objects.filter(name__startswith=fff.name) | Masking.objects.filter(GI__lt = 55).filter(GI_A = 2).order_by('?')[:2]
                        serializer_class = MaskingSerializer
                    break
            if np.array(dataset.class_names)[det_class_ids][i] == fff.name and fff.GI < 55:
                class GIList1(generics.ListCreateAPIView):
                    queryset = Masking.objects.filter(name__startswith=fff.name) | Masking.objects.filter(GI = 1000)
                    serializer_class = MaskingSerializer
    i = i + 1
    k = k + 1
    if i == 1:
        for j in range(2, 13):
            fff = Masking.objects.get(id = j)
            if np.array(dataset.class_names)[det_class_ids][i] == fff.name and fff.GI > 55:
                if fff.GI_A == '1':
                    class GIList2(generics.ListCreateAPIView):
                        queryset = Masking.objects.filter(name__startswith=fff.name) | Masking.objects.filter(GI__lt = 55).filter(GI_A = 1).order_by('?')[:2]
                        serializer_class = MaskingSerializer
                    break
                elif fff.GI_A == '2':
                    class GIList2(generics.ListCreateAPIView):
                        queryset = Masking.objects.filter(name__startswith=fff.name) | Masking.objects.filter(GI__lt = 55).filter(GI_A = 2).order_by('?')[:2]
                        serializer_class = MaskingSerializer
                    break
            if np.array(dataset.class_names)[det_class_ids][i] == fff.name and fff.GI < 55:
                class GIList2(generics.ListCreateAPIView):
                    queryset = Masking.objects.filter(name__startswith=fff.name) | Masking.objects.filter(GI = 1000)
                    serializer_class = MaskingSerializer
    i = i + 1
    k = k + 1
    if i == 2:
        for j in range(2, 13):
            fff = Masking.objects.get(id = j)
            if np.array(dataset.class_names)[det_class_ids][i] == fff.name and fff.GI > 55:
                if fff.GI_A == '1':
                    class GIList3(generics.ListCreateAPIView):
                        queryset = Masking.objects.filter(name__startswith=fff.name) | Masking.objects.filter(GI__lt = 55).filter(GI_A = 1).order_by('?')[:2]
                        serializer_class = MaskingSerializer
                    break
                elif fff.GI_A == '2':
                    class GIList3(generics.ListCreateAPIView):
                        queryset = Masking.objects.filter(name__startswith=fff.name) | Masking.objects.filter(GI__lt = 55).filter(GI_A = 2).order_by('?')[:2]
                        serializer_class = MaskingSerializer
                    break
            if np.array(dataset.class_names)[det_class_ids][i] == fff.name and fff.GI < 55:
                class GIList3(generics.ListCreateAPIView):
                    queryset = Masking.objects.filter(name__startswith=fff.name) | Masking.objects.filter(GI = 1000)
                    serializer_class = MaskingSerializer
    i = i + 1
    k = k + 1

if det_count == 4:
    if i == 0:
        for j in range(2, 13):
            fff = Masking.objects.get(id = j)
            if np.array(dataset.class_names)[det_class_ids][i] == fff.name and fff.GI > 55:
                if fff.GI_A == '1':
                    class GIList1(generics.ListCreateAPIView):
                        queryset = Masking.objects.filter(name__startswith=fff.name) | Masking.objects.filter(GI__lt = 55).filter(GI_A = 1).order_by('?')[:2]
                        serializer_class = MaskingSerializer
                    break
                elif fff.GI_A == '2':
                    class GIList1(generics.ListCreateAPIView):
                        queryset = Masking.objects.filter(name__startswith=fff.name) | Masking.objects.filter(GI__lt = 55).filter(GI_A = 2).order_by('?')[:2]
                        serializer_class = MaskingSerializer
                    break
            if np.array(dataset.class_names)[det_class_ids][i] == fff.name and fff.GI < 55:
                class GIList1(generics.ListCreateAPIView):
                    queryset = Masking.objects.filter(name__startswith=fff.name) | Masking.objects.filter(GI = 1000)
                    serializer_class = MaskingSerializer
    i = i + 1
    k = k + 1
    if i == 1:
        for j in range(2, 13):
            fff = Masking.objects.get(id = j)
            if np.array(dataset.class_names)[det_class_ids][i] == fff.name and fff.GI > 55:
                if fff.GI_A == '1':
                    class GIList2(generics.ListCreateAPIView):
                        queryset = Masking.objects.filter(name__startswith=fff.name) | Masking.objects.filter(GI__lt = 55).filter(GI_A = 1).order_by('?')[:2]
                        serializer_class = MaskingSerializer
                    break
                elif fff.GI_A == '2':
                    class GIList2(generics.ListCreateAPIView):
                        queryset = Masking.objects.filter(name__startswith=fff.name) | Masking.objects.filter(GI__lt = 55).filter(GI_A = 2).order_by('?')[:2]
                        serializer_class = MaskingSerializer
                    break
            if np.array(dataset.class_names)[det_class_ids][i] == fff.name and fff.GI < 55:
                class GIList2(generics.ListCreateAPIView):
                    queryset = Masking.objects.filter(name__startswith=fff.name) | Masking.objects.filter(GI = 1000)
                    serializer_class = MaskingSerializer
    i = i + 1
    k = k + 1
    if i == 2:
        for j in range(2, 13):
            fff = Masking.objects.get(id = j)
            if np.array(dataset.class_names)[det_class_ids][i] == fff.name and fff.GI > 55:
                if fff.GI_A == '1':
                    class GIList3(generics.ListCreateAPIView):
                        queryset = Masking.objects.filter(name__startswith=fff.name) | Masking.objects.filter(GI__lt = 55).filter(GI_A = 1).order_by('?')[:2]
                        serializer_class = MaskingSerializer
                    break
                elif fff.GI_A == '2':
                    class GIList3(generics.ListCreateAPIView):
                        queryset = Masking.objects.filter(name__startswith=fff.name) | Masking.objects.filter(GI__lt = 55).filter(GI_A = 2).order_by('?')[:2]
                        serializer_class = MaskingSerializer
                    break
            if np.array(dataset.class_names)[det_class_ids][i] == fff.name and fff.GI < 55:
                class GIList3(generics.ListCreateAPIView):
                    queryset = Masking.objects.filter(name__startswith=fff.name) | Masking.objects.filter(GI = 1000)
                    serializer_class = MaskingSerializer
    i = i + 1
    k = k + 1
    if i == 3:
        for j in range(2, 13):
            fff = Masking.objects.get(id = j)
            if np.array(dataset.class_names)[det_class_ids][i] == fff.name and fff.GI > 55:
                if fff.GI_A == '1':
                    class GIList4(generics.ListCreateAPIView):
                        queryset = Masking.objects.filter(name__startswith=fff.name) | Masking.objects.filter(GI__lt = 55).filter(GI_A = 1).order_by('?')[:2]
                        serializer_class = MaskingSerializer
                    break
                elif fff.GI_A == '2':
                    class GIList4(generics.ListCreateAPIView):
                        queryset = Masking.objects.filter(name__startswith=fff.name) | Masking.objects.filter(GI__lt = 55).filter(GI_A = 2).order_by('?')[:2]
                        serializer_class = MaskingSerializer
                    break
            if np.array(dataset.class_names)[det_class_ids][i] == fff.name and fff.GI < 55:
                class GIList4(generics.ListCreateAPIView):
                    queryset = Masking.objects.filter(name__startswith=fff.name) | Masking.objects.filter(GI = 1000)
                    serializer_class = MaskingSerializer
    i = i + 1
    k = k + 1

def asdf():
    return k

kkk = asdf()









if det_count == 1:
    class MaskingList(generics.ListCreateAPIView):
        queryset = Masking.objects.filter(name__startswith=np.array(dataset.class_names)[det_class_ids][0])
        serializer_class = MaskingSerializer
elif det_count == 2:
    class MaskingList(generics.ListCreateAPIView):
        queryset = Masking.objects.filter(name__startswith=np.array(dataset.class_names)[det_class_ids][0]) | Masking.objects.filter(name__startswith=np.array(dataset.class_names)[det_class_ids][1])
        serializer_class = MaskingSerializer
elif det_count == 3:
    class MaskingList(generics.ListCreateAPIView):
        queryset = Masking.objects.filter(name__startswith=np.array(dataset.class_names)[det_class_ids][0]) | Masking.objects.filter(name__startswith=np.array(dataset.class_names)[det_class_ids][1]) | Masking.objects.filter(name__startswith=np.array(dataset.class_names)[det_class_ids][2])
        serializer_class = MaskingSerializer
elif det_count == 4:
    class MaskingList(generics.ListCreateAPIView):
        queryset = Masking.objects.filter(name__startswith=np.array(dataset.class_names)[det_class_ids][0]) | Masking.objects.filter(name__startswith=np.array(dataset.class_names)[det_class_ids][1]) | Masking.objects.filter(name__startswith=np.array(dataset.class_names)[det_class_ids][2]) | Masking.objects.filter(name__startswith=np.array(dataset.class_names)[det_class_ids][3])
        serializer_class = MaskingSerializer
elif det_count == 5:
    class MaskingList(generics.ListCreateAPIView):
        queryset = Masking.objects.filter(name__startswith=np.array(dataset.class_names)[det_class_ids][0]) | Masking.objects.filter(name__startswith=np.array(dataset.class_names)[det_class_ids][1]) | Masking.objects.filter(name__startswith=np.array(dataset.class_names)[det_class_ids][2]) | Masking.objects.filter(name__startswith=np.array(dataset.class_names)[det_class_ids][3]) | Masking.objects.filter(name__startswith=np.array(dataset.class_names)[det_class_ids][4])
        serializer_class = MaskingSerializer
elif det_count == 6:
    class MaskingList(generics.ListCreateAPIView):
        queryset = Masking.objects.filter(name__startswith=np.array(dataset.class_names)[det_class_ids][0]) | Masking.objects.filter(name__startswith=np.array(dataset.class_names)[det_class_ids][1]) | Masking.objects.filter(name__startswith=np.array(dataset.class_names)[det_class_ids][2]) | Masking.objects.filter(name__startswith=np.array(dataset.class_names)[det_class_ids][3]) | Masking.objects.filter(name__startswith=np.array(dataset.class_names)[det_class_ids][4]) | Masking.objects.filter(name__startswith=np.array(dataset.class_names)[det_class_ids][5])
        serializer_class = MaskingSerializer
elif det_count == 7:
    class MaskingList(generics.ListCreateAPIView):
        queryset = Masking.objects.filter(name__startswith=np.array(dataset.class_names)[det_class_ids][0]) | Masking.objects.filter(name__startswith=np.array(dataset.class_names)[det_class_ids][1]) | Masking.objects.filter(name__startswith=np.array(dataset.class_names)[det_class_ids][2]) | Masking.objects.filter(name__startswith=np.array(dataset.class_names)[det_class_ids][3]) | Masking.objects.filter(name__startswith=np.array(dataset.class_names)[det_class_ids][4]) | Masking.objects.filter(name__startswith=np.array(dataset.class_names)[det_class_ids][5]) | Masking.objects.filter(name__startswith=np.array(dataset.class_names)[det_class_ids][6])
        serializer_class = MaskingSerializer
elif det_count == 8:
    class MaskingList(generics.ListCreateAPIView):
        queryset = Masking.objects.filter(name__startswith=np.array(dataset.class_names)[det_class_ids][0]) | Masking.objects.filter(name__startswith=np.array(dataset.class_names)[det_class_ids][1]) | Masking.objects.filter(name__startswith=np.array(dataset.class_names)[det_class_ids][2]) | Masking.objects.filter(name__startswith=np.array(dataset.class_names)[det_class_ids][3]) | Masking.objects.filter(name__startswith=np.array(dataset.class_names)[det_class_ids][4]) | Masking.objects.filter(name__startswith=np.array(dataset.class_names)[det_class_ids][5]) | Masking.objects.filter(name__startswith=np.array(dataset.class_names)[det_class_ids][6]) | Masking.objects.filter(name__startswith=np.array(dataset.class_names)[det_class_ids][7])
        serializer_class = MaskingSerializer
elif det_count == 9:
    class MaskingList(generics.ListCreateAPIView):
        queryset = Masking.objects.filter(name__startswith=np.array(dataset.class_names)[det_class_ids][0]) | Masking.objects.filter(name__startswith=np.array(dataset.class_names)[det_class_ids][1]) | Masking.objects.filter(name__startswith=np.array(dataset.class_names)[det_class_ids][2]) | Masking.objects.filter(name__startswith=np.array(dataset.class_names)[det_class_ids][3]) | Masking.objects.filter(name__startswith=np.array(dataset.class_names)[det_class_ids][4]) | Masking.objects.filter(name__startswith=np.array(dataset.class_names)[det_class_ids][5]) | Masking.objects.filter(name__startswith=np.array(dataset.class_names)[det_class_ids][6]) | Masking.objects.filter(name__startswith=np.array(dataset.class_names)[det_class_ids][7]) | Masking.objects.filter(name__startswith=np.array(dataset.class_names)[det_class_ids][8])
        serializer_class = MaskingSerializer
elif det_count == 10:
    class MaskingList(generics.ListCreateAPIView):
        queryset = Masking.objects.filter(name__startswith=np.array(dataset.class_names)[det_class_ids][0]) | Masking.objects.filter(name__startswith=np.array(dataset.class_names)[det_class_ids][1]) | Masking.objects.filter(name__startswith=np.array(dataset.class_names)[det_class_ids][2]) | Masking.objects.filter(name__startswith=np.array(dataset.class_names)[det_class_ids][3]) | Masking.objects.filter(name__startswith=np.array(dataset.class_names)[det_class_ids][4]) | Masking.objects.filter(name__startswith=np.array(dataset.class_names)[det_class_ids][5]) | Masking.objects.filter(name__startswith=np.array(dataset.class_names)[det_class_ids][6]) | Masking.objects.filter(name__startswith=np.array(dataset.class_names)[det_class_ids][7]) | Masking.objects.filter(name__startswith=np.array(dataset.class_names)[det_class_ids][8]) | Masking.objects.filter(name__startswith=np.array(dataset.class_names)[det_class_ids][9])
        serializer_class = MaskingSerializer

class MaskingDetail(generics.RetrieveUpdateDestroyAPIView):
    queryset = Masking.objects.all()
    serializer_class = MaskingSerializer
