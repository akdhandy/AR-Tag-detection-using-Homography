# -*- coding: utf-8 -*-
"""
Created on Sat Feb 22 17:07:49 2020

@author: Arun
"""

import cv2 as cv
import numpy as np
#from matplotlib import pyplot as plt

print("Choose from the selected options for Tag videos")
print("press 0 for Tag0")
print("press 1 for Tag1")
print("press 2 for Tag2")
print("press 3 for Multiple_tags")
print("")
a = int(input("Make your selection: "))
if a == 0:
    vcap = cv.VideoCapture('Tag0.mp4')
elif a == 1:
    vcap = cv.VideoCapture('Tag1.mp4')
elif a == 2:
    vcap = cv.VideoCapture('Tag2.mp4')
elif a == 3:
    vcap = cv.VideoCapture('multipleTags.mp4')
else:
    print("Wrong Selection, exiting code")
    exit(0)
    
def MatrixTag(img):
    dimension = img.shape  
    h_img = dimension[0]
    print(h_img)
    w_img = dimension[1]
    print(w_img)
    bit_h = int(h_img/8)
    print(bit_h)
    bit_w = int(w_img/8)
    print(bit_w)
    a=0 
    ar_tag = np.empty((8,8))
    for i in range(0,h_img,bit_h):
        b=0
        for j in range(0,w_img,bit_w):
            count_black = 0
            count_white = 0   
            for x in range(0,bit_h-1):
                for y in range(0,bit_w-1):
                    if(img[i+x][j+y].all()==0):
                        count_black += 1
                    else:
                        count_white += 1
                        
            if(count_white >= count_black):
                ar_tag[a][b]=1
            else:
                ar_tag[a][b]=0
            b=b+1
        a=a+1
    return ar_tag

img = cv.imread('ref_marker_grid.png')
print(MatrixTag(img))
