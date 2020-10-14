# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 11:48:11 2020
Converts the grasp representation x,y,theta,w,h to rectangle vertices for the vertices coordinates


@author: Elijah
"""
import os
import glob
import math
import cv2
import numpy as np


filepath = './Images/BSD/'
filelabel = './Labels/BSD/'
label_out = './Labels/'
num_of_data = 3
shape = 'shape'
filenames = []
for i in range(0,num_of_data):
    name = filelabel + shape + str(i) + '.txt'
    print(name)
    filenames.append(name)


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
    

    #p1 = [int(253),int(320)]
    #cv2.circle(img,1, 10, (0,255,0), -1)
    #cv2.circle(img, p1, 10, (0,255,0), 2)
    cv2.line(img, p1, p2, (0, 0, 255))
    #Red lines are for the gripper opening width
    cv2.line(img, p2, p3, (255, 0, 0))
    cv2.line(img, p3, p4, (0, 0, 255))
    cv2.line(img, p4, p1, (255, 0, 0))
    return [v1x, v1y, v2x, v2y, v3x, v3y, v4x, v4y] 



for filename in filenames:
    image_path = filepath + filename[13:-4] + '.png'
    print(image_path)
    
    with open(filename, "r") as pcd_file:
        lines = [line.strip().split(" ") for line in pcd_file.readlines()]
    
    number_of_bbox = 0
    index = 0
    
    #order of outputs from LabelTool xc,yc,w,h,theta
    #order of inputs from CNN xc, yc, tan, h, w
    
    img_show = cv2.imread(image_path)
    label = label_out + filename[13:]
    file = open(label, 'w')
    for line in lines:
            if index > 0:
                print("\nBox number", index, "\n")
                box = grasp_to_bbox(float(line[0]),float(line[1]),float(line[4]),float(line[3]), float(line[2]), img_show)
                cv2.imshow('bbox', img_show)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

                file.writelines([("%.2f"%box[0])," ",("%.2f"%box[1]),"\n"])
                file.writelines([("%.2f"%box[2])," ",("%.2f"%box[3]),"\n"])
                file.writelines([("%.2f"%box[4])," ",("%.2f"%box[5]),"\n"])
                file.writelines([("%.2f"%box[6])," ",("%.2f"%box[7]),"\n"])
            index += 1   
    file.close()       
