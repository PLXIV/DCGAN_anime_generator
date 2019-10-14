#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 11 17:42:11 2019

@author: plxiv
"""

import cv2
import sys
import os
import time
from PIL import Image
from pathlib import Path

if __name__ == '__main__':    
    dataset_dir = '../../../dataset/'
    dataset_normalized = '../../dataset_normalized_2/'
    images = os.listdir(dataset_dir)
    for i  in images:
        b = Image.open(dataset_dir + i)
        storing = b.resize((64,64),1)
        storing.save(dataset_normalized + i)