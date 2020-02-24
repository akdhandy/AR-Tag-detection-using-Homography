# -*- coding: utf-8 -*-
"""
Created on Sun Feb 23 10:59:05 2020

@author: Praveen
"""

import cv2
# import numpy as np

cap = cv2.VideoCapture('tag0.mp4')
fourcc = cv2.VideoWriter_fourcc(*'MP4V')
out = cv2.VideoWriter('ArtagDetection2.mp4', fourcc, 30, (1920, 1080))
while(cap.isOpened()):
    ret, arTag = cap.read()
    arTag_grayscale = cv2.cvtColor(arTag, cv2.COLOR_BGR2GRAY)
    # filename = 'AR1.png'
    # img = cv2.imread(filename)
    # img = cv2.resize(img, (480,480))
    # gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    blur = cv2.medianBlur(arTag_grayscale, 25)
    (threshold, binary) = cv2.threshold(blur, 127, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    close = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)
    dilate = cv2.dilate(close, kernel, iterations=2)
    contours, hierarchy = cv2.findContours(dilate, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = []
    # list_for_corners = []
    for c, h in zip(contours, hierarchy[0]):
        c = cv2.convexHull(c)
        epsilon = 0.0001 * cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, epsilon, True)
        cnts.append((approx, h))
        corners = []
    for i in range(len(cnts)):
        if (len(cnts[i][0]) > 4) and ((cnts[i][1][3]!= -1) and (cnts[i][1][2]!= -1)):
            corners = cnts[i]
            
    binary = cv2.drawContours(arTag, corners, 0, (0,255,0),3)
    cv2.imshow('binary', binary)
    out.write(binary)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
cap.release()
out.release()
cv2.destroyAllWindows()

'''
for i in contours:
    corner_four=[] 		#To check 4 corners are there
    if cv2.contourArea(i) > 1000:
        epsilon = 0.1*cv2.arcLength(i,True)
        approx = cv2.approxPolyDP(i, epsilon, True)
        for j in approx:
            count = 0 
            topleft, topright, bottomleft, bottomright = 0, 0, 0, 0
            if j[0][0] < (np.shape(binary)[1]-20) and j[0][1] < (np.shape(binary)[0]-20):
                x=j[0][1]
                y=j[0][0]
                if binary[x-10][y-10]==255:
                    count=count+1
                    topleft=1
                if binary[x+10][y+10]==255:
                    count=count+1
                    bottomright=1
                if binary[x-10][y+10]==255:
                    count=count+1
                    topright=1
                if binary[x+10][y-10]==255:
                    count=count+1
                    bottomleft=1
                if count == 3:
                    if topleft == 1 and topright == 1 and bottomleft == 1:
                        corner='TOPLEFT'
                    elif topleft == 1 and topright == 1 and bottomright == 1:
                        corner='TOPRIGHT'
                    elif topleft == 1 and bottomright == 1 and bottomleft == 1:
                        corner='BOTTOMLEFT'
                    elif bottomright== 1 and topright == 1 and bottomleft == 1:
                        corner='BOTTOMRIGHT'
                    cv2.drawContours(img, approx, -1, (0, 0, 255), 3)
                    corner_four.append([y,x,corner])
                        

            if len(corner_four)==4:
                list_for_corners.append(corner_four)
    if list_for_corners != []:		#if listforcorners is not empty then go forward
        for i in range(0,len(list_for_corners)):
            corner_position = [0,0,0,0]	#to put x and y corner values in a list
            for value in list_for_corners[i]:
                if value[-1] == 'TOPLEFT':
                    corner_position[0] = value[0:2]
                elif value[-1] == 'TOPRIGHT':
                    corner_position[1] = value[0:2]
                elif value[-1] == 'BOTTOMLEFT':
                    corner_position[2] = value[0:2]
                elif value[-1] == 'BOTTOMRIGHT':
                    corner_position[3] = value[0:2]
'''
                    
# binary = cv2.drawContours(img, [cnts], -1, (255, 0, 0), 3)
# cv2.imshow('binary', binary)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
