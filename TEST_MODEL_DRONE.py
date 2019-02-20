import matplotlib.image as mpimg
from scipy.ndimage import filters
import urllib
import numpy as np
from numpy import random
import cv2
from imutils import paths
from math import sqrt
import os
from glob import glob
from keras.preprocessing.image import img_to_array
import shutil
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import img_to_array
from keras.utils import to_categorical

from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras import backend as K
import os

import tensorflow as tf

from caffe_classes import class_names

size_of_grid = 9
from PIL import Image

class LeNet:
    @staticmethod
    def build(width, height, depth, classes):
        # initialize the model
        model = Sequential()
        inputShape = (height, width, depth)

        # if we are using "channels first", update the input shape
        if K.image_data_format() == "channels_first":
            inputShape = (depth, height, width)

            # first set of CONV => RELU => POOL layers
        model.add(Conv2D(20, (5, 5), padding="same",
                             input_shape=inputShape))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        # second set of CONV => RELU => POOL layers
        model.add(Conv2D(50, (5, 5), padding="same"))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        # first (and only) set of FC => RELU layers
        model.add(Flatten())
        model.add(Dense(500))
        model.add(Activation("relu"))

        # softmax classifier
        model.add(Dense(classes))
        model.add(Activation("softmax"))

        # return the constructed network architecture
        return model



def is_square(apositiveint):
  x = apositiveint // 2
  seen = set([x])
  while x * x != apositiveint:
    x = (x + (apositiveint // x)) // 2
    if x in seen: return False
    seen.add(x)
  return True

def crop2(infile,imgheight,imgwidth,imgheight2, imgwidth2):
    #im = Image.open(infile)
    #imgwidth, imgheight = im.size
    b= []
    for i in range(imgheight//imgheight2):
        for j in range(imgwidth//imgwidth2):
            crop_img = infile[i*imgheight2:(i+1)*imgheight2, j*imgwidth2:(j+1)*imgwidth2]
            b.append(crop_img)
    return b

def crop(im, k):
    k2 = 9

    boolean_var = is_square(k)
    #im = Image.open(input)
    # 290 , 640
    imgheight, imgwidth, chan = im.shape

    imgheight = int(imgheight - (imgheight % sqrt(k)))
    imgwidth = int(imgwidth - (imgwidth % sqrt(k)))

    imgheight2 = int(int(imgheight - (imgheight % sqrt(k))) / 3)
    imgwidth2 = int(int(imgwidth - (imgwidth % sqrt(k))) / 3)



    rim = cv2.resize(im, (imgwidth, imgheight))

    M = rim.shape[0] // 2
    N = rim.shape[1] // 2

    #rimg = Image.fromarray(rim, 'L')

    #96, 213 dimensions of tile

    tiles = crop2(rim, imgheight, imgwidth,imgheight2, imgwidth2)
    return tiles

#213 X 96

    #imgwidth, imgheight = im.size



    # if boolean_var == True:
    #     #M = int(imgwidth / sqrt(k))
    #     #N = int(imgheight / sqrt(k))
    #     tiles = [rim[x:x + M, y:y + N] for x in range(0, rim.shape[0], M) for y in range(0, rim.shape[1], N)]
    #     return tiles
    # else:
    #     print("Cannot Split, please return a perfect square in crop function")
    #
    # return []

def grid_correction(tiles ):
    pass

path_model = r'C:\Users\Sean\Drive_by_Ryan_Gosling\The_Og\drone_proj\model\model_min.hdf5'
path_plot = r'C:\Users\Sean\Drive_by_Ryan_Gosling\The_Og\drone_proj\plots\plot_min'
print("[INFO] serializing network...")
from keras.models import load_model

model = LeNet.build(width=213, height=96, depth=1, classes=2)
model = load_model(path_model)

print("MODEL LOADED")

pos_path = r'C:\Users\Sean\Drive_by_Ryan_Gosling\The_Og\drone_project\drone_practice_files\positive'

neg_path = r'C:\Users\Sean\Drive_by_Ryan_Gosling\The_Og\drone_project\drone_practice_files\negative'




# grab the image paths and randomly shuffle them
imagePaths = sorted(list(paths.list_images(pos_path)))
random.seed(42)
random.shuffle(imagePaths)

pos_data=[]
pos_labels = []
neg_data=[]
neg_labels = []
# loop over the input images
for imagePath in imagePaths:
    # load the image, pre-process it, and store it in the data list

    # extract the class label from the image path and update the
    # labels list
    label = imagePath.split(os.path.sep)[-2]
    label = 1 if label == "positive" else 0
    pos_labels.append(label)

i = 0
pos_labels2 = []
for imagePath in imagePaths:

    print(imagePath)
    image = cv2.imread(imagePath)
    pos_data.extend(crop(image, size_of_grid))
    b = [pos_labels[i]] * size_of_grid
    pos_labels2.extend(b)
    i = i + 1

imagePaths = sorted(list(paths.list_images(neg_path)))
random.seed(42)
random.shuffle(imagePaths)

for imagePath in imagePaths:
    # load the image, pre-process it, and store it in the data list

    # extract the class label from the image path and update the
    # labels list
    label = imagePath.split(os.path.sep)[-2]
    label = 1 if label == "positive" else 0
    neg_labels.append(label)

# Now that the images are split and gathered into arrays
# 1. we must get a for loop where we take the positive images and then split them
# 2. Start cropping the images, and then assign them to a label extrapolated in length to the k value of grid

i = 0
neg_labels2 = []
for imagePath in imagePaths:

    print(imagePath)
    image = cv2.imread(imagePath)
    neg_data.extend(crop(image, size_of_grid))
    b = [neg_labels[i]] * size_of_grid
    neg_labels2.extend(b)
    i = i + 1

pos_data2 = pos_data
pos_labels3 = pos_labels2

np.save('neg_test_data_np_min.npy', pos_data2)
#pos_data2 = np.load('data_np.npy')

#np.save('data_np.npy', pos_labels3)
#pos_labels31 = np.load('labels_np_min.npy')

pos_data.extend(neg_data)

pos_labels2.extend(neg_labels2)
np.save('neg_test_labels_np_min.npy', pos_labels2)


data = pos_data
labels = pos_labels2

pos_labels3 = pos_labels2

print("CORRECTING DATA ")
data = np.array(data, dtype="float") / 255.0

#np.save('test_data_floated_np_min.npy', data)

total = 170
average = 0

for i in np.random.choice(np.arange(0, len(pos_labels3)), size=(170,)):
    is_crash = model.predict(data[np.newaxis, i])
    average = average + is_crash

average2 = average / total

k =0