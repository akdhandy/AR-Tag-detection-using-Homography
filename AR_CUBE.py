# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 20:04:42 2020

@author: Arun
"""

import cv2
import numpy as np
import math 
# import imutils

cap = cv2.VideoCapture('Tag1.mp4')
lenaImage = cv2.imread('Lena.png')
lenaImage = cv2.resize(lenaImage, (200, 200))
# fourcc = cv2.VideoWriter_fourcc(*'MP4V')
# out = cv2.VideoWriter('ArtagDetection3.mp4', fourcc, 30, (1920, 1080))
while(cap.isOpened()):
    ret, arTag = cap.read()
    size = arTag.shape
    if ret == False:
        break
    arTag_grayscale = cv2.cvtColor(arTag, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(arTag_grayscale, (5,5),0)
    (threshold, binary) = cv2.threshold(blur, 127, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    close = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)
    dilate = cv2.dilate(close, kernel, iterations=2)
    contours, hierarchy = cv2.findContours(dilate, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    unwantedContours = []
    for contour, h in enumerate(hierarchy[0]):
        if h[2] == -1 or h[3] == -1:
            unwantedContours.append(contour)
            
    cnts = [c2 for c1, c2 in enumerate(contours) if c1 not in unwantedContours]
    cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:2]
    
    tagContours = []
    for c in cnts:
        c = cv2.convexHull(c)
        epsilon = 0.01*cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, epsilon, True)
        if len(approx) == 4:
            tagContours.append(approx)

    corners = []
    for tc in tagContours:
        coords = []
        for p in tc:
            coords.append([p[0][0],p[0][1]])
        corners.append(coords)
    contourImage = cv2.drawContours(arTag, tagContours, -1, (0,0,255), 4)
    for tag1, tag2 in enumerate(corners):
        corner_points_x = []
        corner_points_y = []
        ordered_corner_points = np.zeros((4,2))
        corner_points = np.array(tag2)
        corner_points_sorted_x = corner_points[np.argsort(corner_points[:, 0]), :]
        left = corner_points_sorted_x[:2, :]
        right = corner_points_sorted_x[2:, :]
        left = left[np.argsort(left[:, 1]), :]
        (ordered_corner_points[0], ordered_corner_points[3]) = left
        d1 = math.sqrt((right[0][0] - ordered_corner_points[0][0])**2 + (right[0][1] - ordered_corner_points[0][1])**2)
        d2 = math.sqrt((right[1][0] - ordered_corner_points[0][0])**2 + (right[1][1] - ordered_corner_points[0][1])**2)
        if d1 > d2:
            ordered_corner_points[2] = right[0]
            ordered_corner_points[1] = right[1]
        else:
            ordered_corner_points[1] = right[0]
            ordered_corner_points[2] = right[1]

        for point in ordered_corner_points:
            corner_points_x.append(point[0])
            corner_points_y.append(point[1])
            
        reference_corners_x = [0,199,199,0]
        reference_corners_y = [0,0,199,199]
        
        A  = np.array([
                   [ corner_points_x[0], corner_points_y[0], 1 , 0  , 0 , 0 , -reference_corners_x[0]*corner_points_x[0], -reference_corners_x[0]*corner_points_y[0], -reference_corners_x[0]],
                   [ 0 , 0 , 0 , corner_points_x[0], corner_points_y[0], 1 , -reference_corners_y[0]*corner_points_x[0], -reference_corners_y[0]*corner_points_y[0], -reference_corners_y[0]],
                   [ corner_points_x[1], corner_points_y[1], 1 , 0  , 0 , 0 , -reference_corners_x[1]*corner_points_x[1], -reference_corners_x[1]*corner_points_y[1], -reference_corners_x[1]],
                   [ 0 , 0 , 0 , corner_points_x[1], corner_points_y[1], 1 , -reference_corners_y[1]*corner_points_x[1], -reference_corners_y[1]*corner_points_y[1], -reference_corners_y[1]],
                   [ corner_points_x[2], corner_points_y[2], 1 , 0  , 0 , 0 , -reference_corners_x[2]*corner_points_x[2], -reference_corners_x[2]*corner_points_y[2], -reference_corners_x[2]],
                   [ 0 , 0 , 0 , corner_points_x[2], corner_points_y[2], 1 , -reference_corners_y[2]*corner_points_x[2], -reference_corners_y[2]*corner_points_y[2], -reference_corners_y[2]],
                   [ corner_points_x[3], corner_points_y[3], 1 , 0  , 0 , 0 , -reference_corners_x[3]*corner_points_x[3], -reference_corners_x[3]*corner_points_y[3], -reference_corners_x[3]],
                   [ 0 , 0 , 0 , corner_points_x[3], corner_points_y[3], 1 , -reference_corners_y[3]*corner_points_x[3], -reference_corners_y[3]*corner_points_y[3], -reference_corners_y[3]],
                   ], dtype=np.float64)
 
        U,S,V = np.linalg.svd(A)
        H = V[:][8]/V[8][8]
        # H = V[:][8]
        H_matrix = np.reshape(H, (3,3))
        H_inverse = np.linalg.inv(H_matrix)
        coords = np.indices((200, 200)).reshape(2, -1)
        coords=np.vstack((coords, np.ones(coords.shape[1]))) 
        x2, y2 = coords[0], coords[1]# Apply inverse transform and round it (nearest neighbour interpolation)
        warp_coords = ( H_inverse@coords)
        warp_coords=warp_coords/warp_coords[2]
        x1, y1 = warp_coords[0, :], warp_coords[1, :]# Get pixels within image boundaries
        indices = np.where((x1 >= 0) & (x1 < size[1]) &
                      (y1 >= 0) & (y1 < size[0]))
        xpix1, ypix1 = x2[indices], y2[indices]
        xpix1=xpix1.astype(np.int)
        ypix1=ypix1.astype(np.int)
        xpix2, ypix2 = x1[indices], y1[indices]# Map Correspondence
        xpix2=xpix2.astype(np.int)
        ypix2=ypix2.astype(np.int)
        perspectiveImage = np.zeros((200,200))
        perspectiveImage[ypix1, xpix1] = binary[ypix2,xpix2]

        
        stride = perspectiveImage.shape[0]//8
        x = 0
        y = 0
        tagGrid = np.zeros((8,8))
        for i in range(8):
            for j in range(8):
                cell = perspectiveImage[y:y+stride, x:x+stride]
                cv2.rectangle(perspectiveImage,(x,y),(x+stride,y+stride), (255,0,0), 1)
                if cell.mean() > 255//2:
                    tagGrid[i][j] = 1
                x = x + stride
            x = 0
            y = y + stride
        if(tagGrid[2][2] == 0 and tagGrid[2][5] == 0 and tagGrid[5][2] == 0 and tagGrid[5][5] == 1):
            orientation = 0
            Id = tagGrid[3][3]*1 + tagGrid[4][3]*8 + tagGrid[4][4]*4 + tagGrid[3][4]*2
            # rotatedLena = imutils.rotate_bound(lenaImage, orientation)
            reference_corners_x = [0,lenaImage.shape[0]-1,lenaImage.shape[0]-1,0]
            qreference_corners_y = [0,0,lenaImage.shape[0]-1,lenaImage.shape[0]-1]
        elif(tagGrid[2][2] == 1 and tagGrid[2][5] == 0 and tagGrid[5][2] == 0 and tagGrid[5][5] == 0):
            orientation = 180
            Id = tagGrid[3][3]*4 + tagGrid[4][3]*2 + tagGrid[4][4] + tagGrid[3][4]*8
            # rotatedLena = imutils.rotate_bound(lenaImage, orientation)
            reference_corners_x = [lenaImage.shape[0]-1,0,0,lenaImage.shape[0]-1]
            reference_corners_y = [lenaImage.shape[0]-1,lenaImage.shape[0]-1,0,0]
        elif(tagGrid[2][2] == 0 and tagGrid[2][5] == 1 and tagGrid[5][2] == 0 and tagGrid[5][5] == 0):
            orientation = 90
            Id = tagGrid[3][3]*2 + tagGrid[3][4]*4 + tagGrid[4][4]*8 + tagGrid[4][3]*1
            # rotatedLena = imutils.rotate_bound(lenaImage, orientation)
            reference_corners_x = [lenaImage.shape[0]-1,lenaImage.shape[0]-1,0,0]
            reference_corners_y = [0,lenaImage.shape[0]-1,lenaImage.shape[0]-1,0]
        elif(tagGrid[2][2] == 0 and tagGrid[2][5] == 0 and tagGrid[5][2] == 1 and tagGrid[5][5] == 0):
            orientation = -90
            Id = tagGrid[3][3]*8 + tagGrid[3][4] + tagGrid[4][4]*2 + tagGrid[4][3]*4
            # rotatedLena = imutils.rotate_bound(lenaImage, orientation)
            reference_corners_x = [0,0,lenaImage.shape[0]-1,lenaImage.shape[0]-1]
            reference_corners_y = [lenaImage.shape[0]-1,0,0,lenaImage.shape[0]-1]
        else:
            orientation = None
        
        A  = np.array([
                   [ corner_points_x[0], corner_points_y[0], 1 , 0  , 0 , 0 , -reference_corners_x[0]*corner_points_x[0], -reference_corners_x[0]*corner_points_y[0], -reference_corners_x[0]],
                   [ 0 , 0 , 0 , corner_points_x[0], corner_points_y[0], 1 , -reference_corners_y[0]*corner_points_x[0], -reference_corners_y[0]*corner_points_y[0], -reference_corners_y[0]],
                   [ corner_points_x[1], corner_points_y[1], 1 , 0  , 0 , 0 , -reference_corners_x[1]*corner_points_x[1], -reference_corners_x[1]*corner_points_y[1], -reference_corners_x[1]],
                   [ 0 , 0 , 0 , corner_points_x[1], corner_points_y[1], 1 , -reference_corners_y[1]*corner_points_x[1], -reference_corners_y[1]*corner_points_y[1], -reference_corners_y[1]],
                   [ corner_points_x[2], corner_points_y[2], 1 , 0  , 0 , 0 , -reference_corners_x[2]*corner_points_x[2], -reference_corners_x[2]*corner_points_y[2], -reference_corners_x[2]],
                   [ 0 , 0 , 0 , corner_points_x[2], corner_points_y[2], 1 , -reference_corners_y[2]*corner_points_x[2], -reference_corners_y[2]*corner_points_y[2], -reference_corners_y[2]],
                   [ corner_points_x[3], corner_points_y[3], 1 , 0  , 0 , 0 , -reference_corners_x[3]*corner_points_x[3], -reference_corners_x[3]*corner_points_y[3], -reference_corners_x[3]],
                   [ 0 , 0 , 0 , corner_points_x[3], corner_points_y[3], 1 , -reference_corners_y[3]*corner_points_x[3], -reference_corners_y[3]*corner_points_y[3], -reference_corners_y[3]],
                   ], dtype=np.float64)
        '''
        A  = np.array([
                   [ reference_corners_x[0], reference_corners_y[0], 1 , 0  , 0 , 0 , -corner_points_x[0]*reference_corners_x[0], -corner_points_x[0]*reference_corners_y[0], -corner_points_x[0]],
                   [ 0 , 0 , 0 , reference_corners_x[0], reference_corners_y[0], 1 , -corner_points_y[0]*reference_corners_x[0], -corner_points_y[0]*reference_corners_y[0], -corner_points_y[0]],
                   [ reference_corners_x[1], reference_corners_y[1], 1 , 0  , 0 , 0 , -corner_points_x[1]*reference_corners_x[1], -corner_points_x[1]*reference_corners_y[1], -corner_points_x[1]],
                   [ 0 , 0 , 0 , reference_corners_x[1], reference_corners_y[1], 1 , -corner_points_y[1]*reference_corners_x[1], -corner_points_y[1]*reference_corners_y[1], -corner_points_y[1]],
                   [ reference_corners_x[2], reference_corners_y[2], 1 , 0  , 0 , 0 , -corner_points_x[2]*reference_corners_x[2], -corner_points_x[2]*reference_corners_y[2], -corner_points_x[2]],
                   [ 0 , 0 , 0 , reference_corners_x[2], reference_corners_y[2], 1 , -corner_points_y[2]*reference_corners_x[2], -corner_points_y[2]*reference_corners_y[2], -corner_points_y[2]],
                   [ reference_corners_x[3], reference_corners_y[3], 1 , 0  , 0 , 0 , -corner_points_x[3]*reference_corners_x[3], -corner_points_x[3]*reference_corners_y[3], -corner_points_x[3]],
                   [ 0 , 0 , 0 , reference_corners_x[3], reference_corners_y[3], 1 , -corner_points_y[3]*reference_corners_x[3], -corner_points_y[3]*reference_corners_y[3], -corner_points_y[3]],
                   ], dtype=np.float64)
        '''
        U,S,V = np.linalg.svd(A)
        H = V[:][8]/V[8][8]
        # H = V[:][8]
        H_matrix = np.reshape(H, (3,3))
        H_inverse = np.linalg.inv(H_matrix)
        warp_coords = ( H_inverse@coords)
        warp_coords=warp_coords/warp_coords[2]
        x1, y1 = warp_coords[0, :], warp_coords[1, :]# Get pixels within image boundaries
        indices = np.where((x1 >= 0) & (x1 < size[1]) &
                      (y1 >= 0) & (y1 < size[0]))
        xpix1, ypix1 = x2[indices], y2[indices]
        xpix1=xpix1.astype(np.int)
        ypix1=ypix1.astype(np.int)
        xpix2, ypix2 = x1[indices], y1[indices]# Map Correspondence
        xpix2=xpix2.astype(np.int)
        ypix2=ypix2.astype(np.int)
        arTag[ypix2, xpix2] = lenaImage[ypix1,xpix1]
        
        # cv2.imshow('rotated lena', rotatedLena)
        cv2.imshow('perspective projection', perspectiveImage)
        cv2.imshow('superimpose', arTag)
    # cv2.imshow('tag contours', contourImage)
    # cv2.imshow('superimpose', arTag)
        #Placing a Virtual Cube over the tag
        #Camera Intrinsic Parameters
        K = np.array([[1406.08415449821,0,0],[2.20679787308599, 1417.99930662800,0],[1014.13643417416, 566.347754321696,1]])
        h1 = H_inverse[:,0]
        h2 = H_inverse[:,1]
        K = np.transpose(K)
        K_inv = np.linalg.inv(K)
        lamda = 1/((np.linalg.norm(np.dot(K_inv,h1))+np.linalg.norm(np.dot(K_inv,h2)))/2)
        Btilde = np.dot(K_inv,H)
        if np.linalg.det(Btilde)>0:
            B = Btilde
        else:
            B = -Btilde
        b1 = B[:,0]
        b2 = B[:,1]
        b3 = B[:,2]
        r1 = lamda*b1
        r2 = lamda*b2
        r3 = np.cross(r1,r2)
        t = lamda*b3
        projectionMatrix = np.dot(K, (np.stack((r1,r2,r3,t), axis=1)))
          
            
        x1,y1,z1 = np.matmul(projectionMatrix,[0,0,0,1])
        x2,y2,z2 = np.matmul(projectionMatrix,[0,199,0,1])
        x3,y3,z3 = np.matmul(projectionMatrix,[199,0,0,1])
        x4,y4,z4 = np.matmul(projectionMatrix,[199,199,0,1])
        x5,y5,z5 = np.matmul(projectionMatrix,[0,0,-199,1])
        x6,y6,z6 = np.matmul(projectionMatrix,[0,199,-199,1])
        x7,y7,z7 = np.matmul(projectionMatrix,[199,0,-199,1])
        x8,y8,z8 = np.matmul(projectionMatrix,[199,199,-199,1])
        
        cv2.circle(arTag,(int(x1/z1),int(y1/z1)), 5, (0,255,255), -1)
        cv2.circle(arTag,(int(x2/z2),int(y2/z2)), 5, (0,255,255), -1)
        cv2.circle(arTag,(int(x3/z3),int(y3/z3)), 5, (0,255,255), -1)
        cv2.circle(arTag,(int(x4/z4),int(y4/z4)), 5, (0,255,255), -1)
        cv2.circle(arTag,(int(x5/z5),int(y5/z5)), 5, (0,255,255), -1)
        cv2.circle(arTag,(int(x6/z6),int(y6/z6)), 5, (0,255,255), -1)
        cv2.circle(arTag,(int(x7/z7),int(y7/z7)), 5, (0,255,255), -1)
        cv2.circle(arTag,(int(x8/z8),int(y8/z8)), 5, (0,255,255), -1)
        cv2.line(arTag,(int(x1/z1),int(y1/z1)),(int(x5/z5),int(y5/z5)), (255,255,0), 4)
        cv2.line(arTag,(int(x2/z2),int(y2/z2)),(int(x6/z6),int(y6/z6)), (0,255,255), 4)
        cv2.line(arTag,(int(x3/z3),int(y3/z3)),(int(x7/z7),int(y7/z7)), (0,150,255), 4)
        cv2.line(arTag,(int(x4/z4),int(y4/z4)),(int(x8/z8),int(y8/z8)), (255,255,255), 4)
        cv2.line(arTag,(int(x1/z1),int(y1/z1)),(int(x2/z2),int(y2/z2)), (0,155,155), 4)
        cv2.line(arTag,(int(x1/z1),int(y1/z1)),(int(x3/z3),int(y3/z3)), (0,205,100), 4)
        cv2.line(arTag,(int(x2/z2),int(y2/z2)),(int(x4/z4),int(y4/z4)), (200,50,0), 4)
        cv2.line(arTag,(int(x3/z3),int(y3/z3)),(int(x4/z4),int(y4/z4)), (0,255,255), 4)
        cv2.line(arTag,(int(x5/z5),int(y5/z5)),(int(x6/z6),int(y6/z6)), (100,0,255), 4)
        cv2.line(arTag,(int(x5/z5),int(y5/z5)),(int(x7/z7),int(y7/z7)), (0,0,255), 4)
        cv2.line(arTag,(int(x6/z6),int(y6/z6)),(int(x8/z8),int(y8/z8)), (255,0,0), 4)
        cv2.line(arTag,(int(x7/z7),int(y7/z7)),(int(x8/z8),int(y8/z8)), (0,0,100), 4)
                              
        

#Displaying the image output in a window named "Display"
    cv2.imshow("DISPLAY", arTag)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()