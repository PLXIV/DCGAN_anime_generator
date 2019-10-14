#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 15:21:01 2019

@author: plxiv
"""

import cv2
import sys
import os
import time
from PIL import Image
from pathlib import Path

def detect(folder, filename, store_folder, cascade_file = "../external_tools/lbpcascade_animeface.xml"):
    if not os.path.isfile(cascade_file):
        raise RuntimeError("%s: not found" % cascade_file)

    cascade = cv2.CascadeClassifier(cascade_file)
    image = cv2.imread(folder + filename, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    faces = cascade.detectMultiScale(gray,
                                     # detector options
                                     scaleFactor = 1.1,
                                     minNeighbors = 5,
                                     minSize = (24, 24))
    for i, (x, y, w, h) in enumerate(faces):
        new_image = image[y:y+h, x:x+w]
        if is_grey_scale(new_image) > 5:
            cv2.imwrite(store_folder + str(i) + '_' + filename, new_image )

def is_grey_scale(img):
    img = Image.fromarray(img)
    w,h = img.size
    df = 0
    for i in range(w):
        for j in range(h):
            r,g,b = img.getpixel((i,j))
            rg = abs(r - g);
            rb = abs(r - b);
            gb = abs(g - b);
            df += rg + rb + gb;
    return df / (h * w) 
    
    w,h,_ = img.shape
    for i in range(w):
        for j in range(h):
            r,g,b = img.getpixel((i,j))
            if r != g != b: return False
    return True
    

def main():
    danbooru_path = Path('/media/plxiv/AEF0C4B1F0C480D7/danbooru2018/512px/')
    dataset_dir = '../../dataset/'
    folder = '/media/plxiv/AEF0C4B1F0C480D7/danbooru2018/512px/' 
    for subfolder in range(200,999):
        a = '0' + str(subfolder)
        print(a)
        for filename in os.listdir(danbooru_path / a):
            f = folder + a + '/'
            detect(f, filename, dataset_dir)


    filename = '0_5001.jpg'
if __name__ == '__main__':    
    main()