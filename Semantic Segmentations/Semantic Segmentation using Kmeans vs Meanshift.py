import tensorflow as tf
import warnings
from distutils.version import LooseVersion
import csv
import time
from sklearn.cluster import MeanShift, estimate_bandwidth
import re
import random
import numpy as np
import os
import cv2
import scipy.misc
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans

def Read_Images(Images_path):
    All_Images = []
    for img in os.listdir(Images_path):
        path = os.path.join(Images_path + img)
        ReadImage = cv2.imread(path)
        RGB_Image = cv2.cvtColor(ReadImage, cv2.COLOR_BGR2RGB)
        ResizedImage = cv2.resize(RGB_Image, (200, 200))
        ReshapedImage = np.reshape(ResizedImage, (-1, 3))
        All_Images.append(ReshapedImage)
    random.shuffle(All_Images)
    return All_Images

#Quantile is used to control the bandwidth size, this is usually chosen upon trial and error
def MeanShift_Func(Images):
    i = 0
    output_dir = './OutputImages/'
    for img in Images:
        bandwidth = estimate_bandwidth(img, quantile=0.1, n_samples=500)
        ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
        ms.fit(img)
        Labels = ms.labels_
        cluster_centers = ms.cluster_centers_
        #SegImage = np.reshape(Labels, [200, 200])
        SegImage = cluster_centers[np.reshape(Labels, (200, 200))]
        # SegImage = cv2.cvtColor(SegImage, cv2.COLOR_BGRA2RGB)
        #cv2.imwrite(os.path.join(output_dir + str(i)), SegImage)
        #i = i + 1
        plt.figure(figsize=(20, 20))
        plt.grid()
        plt.imshow(SegImage.astype(np.uint8))
        plt.show()


def KMeans_Func(Images):
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    k = 4
    for img in Images:
        pixel_values = np.float32(img)
        _, labels, (centers) = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        centers = np.uint8(centers)
        labels = labels.flatten()
        segmented_image = centers[np.reshape(labels, (200, 200))]
        plt.figure(figsize=(20, 20))
        plt.grid()
        plt.imshow(segmented_image.astype(np.uint8))
        plt.show()


if __name__ == '__main__':
    Images_path = 'Images/'
    MyImages = Read_Images(Images_path)
    #MeanShift_Func(MyImages)
    KMeans_Func(MyImages)