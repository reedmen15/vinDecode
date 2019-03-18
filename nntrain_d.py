#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import cv2
import numpy as np
import sys
import json
import matplotlib.image as mpimg
from matplotlib import pyplot as plt
import warnings
warnings.filterwarnings('ignore')

CODEVIN_DIR = os.path.abspath('.')
sys.path.append(CODEVIN_DIR)

# specify the path to Mask_RCNN if you placed it outside Nomeroff-net project
MASK_RCNN_DIR = os.path.join(CODEVIN_DIR, 'Mask_RCNN')
MASK_RCNN_LOG_DIR = os.path.join(CODEVIN_DIR, "logs")
NUMBER_MODEL_PATH = os.path.join(CODEVIN_DIR, "models/mask_rcnn_vin_cod_0003.h5")

DATASET_NAME = "train"
VERSION = "2019_03"
MASK_RCNN_FROZEN_PATH = os.path.join(CODEVIN_DIR, "models/", 'vin_{}_{}.pb'.format(DATASET_NAME, VERSION))
# Import license plate recognition tools.
from codevin import  Detector
from codevin.Base import convert_keras_to_freeze_pb

CONFIG = {
    "GPU_COUNT": 1,
    "IMAGES_PER_GPU": 1,
    "WEIGHTS": "models/mask_rcnn_vin_cod_0003.h5",
    "EPOCHS": 20,
    "CLASS_NAMES": ["BG", "vin_cod"], 
    "NAME": "vin_cod",
    "DATASET_DIR": "datasets/mrcnn",
    "LAYERS": "all",
    "NUM_CLASSES": 2
}

# Initialize npdetector with default configuration file.
nnet = Detector(MASK_RCNN_DIR, MASK_RCNN_LOG_DIR, CONFIG)

nnet.train()
#nnet.loadModel(MASK_RCNN_MODEL_PATH)
