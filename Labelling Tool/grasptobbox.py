# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 11:48:11 2020

@author: Elijah
"""
import os
import glob
import math
import cv2
import numpy as np
from PIL import Image

filepath = './Images/test/flash_light.png'
filename = './Labels/test/flash_light.txt'


def grasp_to_bbox(x, y, theta, h, w, img):
    # converts the g representations grasps to bounding boxes
    #theta = np.arctan(tan)
    
    theta = -math.radians(theta)
    v1x = x - w/2*np.cos(theta) - h/2*np.sin(theta)
    v1y = y + w/2*np.sin(theta) - h/2*np.cos(theta) #(x,y) coordinates
    v2x = x + w/2*np.cos(theta) - h/2*np.sin(theta)
    v2y = y - w/2*np.sin(theta) - h/2*np.cos(theta)
    v3x = x + w/2*np.cos(theta) + h/2*np.sin(theta)
    v3y = y - w/2*np.sin(theta) + h/2*np.cos(theta)
    v4x = x - w/2*np.cos(theta) + h/2*np.sin(theta)
    v4y = y + w/2*np.sin(theta) + h/2*np.cos(theta)
    #this conversion is correct
    p1 = (int(v1x), int(v1y))
    p2 = (int(v2x), int(v2y))
    p3 = (int(v3x), int(v3y))
    p4 = (int(v4x), int(v4y))
    
    
    #Point test
    #p1 = [int(253),int(320)]
    #cv2.circle(img,1, 10, (0,255,0), -1)
    
    cv2.line(img, p1, p2, (0, 0, 255))
    cv2.line(img, p2, p3, (0, 255, 0))
    cv2.line(img, p3, p4, (0, 0, 255))
    cv2.line(img, p4, p1, (0, 255, 0))
    return [v1x, v1y, v2x, v2y, v3x, v3y, v4x, v4y] 

with open(filename, "r") as pcd_file:
    lines = [line.strip().split(" ") for line in pcd_file.readlines()]

number_of_bbox = 0
index = 0

#order of outputs from LabelTool xc,yc,w,h,theta
#order of inputs from CNN xc, yc, tan, h, w

img_show = cv2.imread(filepath)

for line in lines:
        if index > 0:
            print("\nBox number", index, "\n")
            box = grasp_to_bbox(float(line[0]),float(line[1]),float(line[4]),float(line[3]), float(line[2]), img_show)
            cv2.imshow('bbox', img_show)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        index += 1   

