import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse
import time
from pyramid_image import *
from template_matcher import *
import cv2.aruco as Aruco

def_dict = Aruco.DICT_7X7_50


def accurate_aruco_matcher(image):
'''Takes shot and gives (x,y,theta).'''
# Get predefined dictionary.
dictionary = Aruco.getPredefinedDictionary(def_dict)

# Detect aruco marker - Aruco library.


# Genearate template for marker with id of detected.


# Affine translations of the marker.


# Contours of templates.


# Gradient of image.


# Matching of grad and contours.


# Find correlation eye.


# Open video file.
cap = cv2.VideoCapture(args["video"])
# Open target image.
target_img = cv2.imread(args["target"])
