# -*- coding: utf-8 -*-
"""
Created on Sat Feb 22 11:00:14 2020

@author: Praveen
"""
import numpy as np
from scipy import signal, ndimage
import cv2

cap = cv2.VideoCapture('tag0.mp4')
fourcc = cv2.VideoWriter_fourcc(*'MP4V')
out = cv2.VideoWriter('ArtagDetection.mp4', fourcc, 20, (640, 480))
while(cap.isOpened()):
    ret, arTag = cap.read()
    arTag = cv2.resize(arTag, (640, 480))
    # arTag = cv2.imread('ref_marker.png')
    arTag_grayscale = cv2.cvtColor(arTag, cv2.COLOR_BGR2GRAY)
    k = 0.05
    sobelx = np.array([[-1, 0, 1],[-2, 0, 2],[-1, 0, 1]])
    sobely = np.array([[-1, -2, -1],[0, 0, 0],[1, 2, 1]])
    Ix = signal.convolve2d(arTag_grayscale, sobelx, mode = 'same')
    Iy = signal.convolve2d(arTag_grayscale, sobely, mode = 'same')
    Ixx = ndimage.gaussian_filter(Ix*Ix, sigma = 1)
    Ixy = ndimage.gaussian_filter(Ix*Iy, sigma = 1)
    Iyy = ndimage.gaussian_filter(Iy*Iy, sigma = 1)
    determinantA = Ixx*Iyy - Ixy*Ixy
    traceA = Ixx + Iyy
    harrisResponse = determinantA - k*traceA*traceA
    corners = np.copy(arTag)
    edges = np.copy(arTag)
    for i, j in enumerate(harrisResponse):
        for k, r in enumerate(j):
            if r > 0:
                corners[i, k] = [255,0,0]
            elif r < 0:
                edges[i, k] = [0,255,0]
    # cv2.imshow('arTag', arTag)
    # cv2.imshow('corners', corners)
    cv2.imshow('edges', edges)
    out.write(edges)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
cap.release()
out.release()
cv2.destroyAllWindows()
