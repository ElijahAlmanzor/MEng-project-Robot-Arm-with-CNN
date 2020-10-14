# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 21:35:11 2020
Script to essentially augment the images
as well as their labels

@author: Elijah
"""
from PIL import Image
import random
import cv2
import numpy as np
import math
import glob
import os




def grasp_to_bbox(x, y, theta, h, w, img):
    #This function should be called last tbh
    
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

    return [v1x, v1y, v2x, v2y, v3x, v3y, v4x, v4y]

def bboxes_to_grasps(box):
    #convert the loaded boxes back into grasps, so it can then be edited
    x = (box[0] + (box[4] - box[0])/2) 
    y = (box[1] + (box[5] - box[1])/2) 
    try:
        tan = (box[3] -box[1]) / (box[2] -box[0])
    except:
        print("Some error")
        return True
    theta = np.degrees(np.arctan(tan))
    w = np.sqrt(np.power((box[2] - box[0]), 2) + np.power((box[3] - box[1]), 2)) #euclidean distance to get h and w
    h = np.sqrt(np.power((box[6] - box[0]), 2) + np.power((box[7] - box[1]), 2))
    return [x,y,theta,h,w]
    #since want to make the gripper opening width to be the one with respect to the angle of rotation, then swittch h and w
    #This function works on arbitrary point order
    

def disp(box, img):
    n_box = len(box)
    for i in range(n_box):
        cv2.line(img, (int(box[i][0]),(int(box[i][1]))), (int(box[i][2]),(int(box[i][3]))), (0, 0, 255))
        cv2.line(img, (int(box[i][2]),(int(box[i][3]))), (int(box[i][4]),(int(box[i][5]))), (255, 0, 0))
        cv2.line(img, (int(box[i][4]),(int(box[i][5]))), (int(box[i][6]),(int(box[i][7]))), (0, 0, 255))
        cv2.line(img, (int(box[i][6]),(int(box[i][7]))), (int(box[i][0]),(int(box[i][1]))), (255, 0, 0)) 
        i += 4



def augment(img, imgd, lines, data_aug):
    global index
    width, height = img.size   # Get dimensions
    new_width = 640 #e.g. 640x480 for the test set
    new_height = 480
    left = (width - new_width)/2 # 
    top = (height - new_height)/2 #
    right = (width + new_width)/2
    bottom = (height + new_height)/2
    xc = width/2
    yc = height/2 
    
    
    bbox = []
    box = []
    
    
    for cur, line in enumerate(lines): #loads the box coordinates
        try:
            box.append(int(float(line[0])))
            box.append(int(float(line[1])))
        except:
            print("some string error?")
            print(cur)
        if len(box) == 8:
            bbox.append(box)
            box = []
    
    grasps = []
    for box in bbox: #converts the box coordinates to grasp representations
        grasps.append(bboxes_to_grasps(box))
    
    for i in range(3):
        #For each produced image, want it to augment image once, but augment all of the individual grasps
        delta_tau = random.randrange(-50,50,1) #generates the rotation of the image
        im = img.rotate(-delta_tau)
        imd = imgd.rotate(-delta_tau)        
        pos_or_neg = [-1.75,1]
        ypos_or_neg = [-1,2]
        rx = random.sample(pos_or_neg, 1)[0]
        ry = random.sample(ypos_or_neg, 1)[0]
        
        dx = random.randrange(0,80,1) * rx 
        dy = random.randrange(0,80,1) * ry
        a = 1
        b = 0
        c = dx #left/right (i.e. 5/-5) #via number of pixels
        d = 0
        e = 1
        f = dy #up/down (i.e. 5/-5)        
        im = im.transform(img.size, Image.AFFINE, (a, b, c, d, e, f))
        imd = imd.transform(img.size, Image.AFFINE, (a, b, c, d, e, f))        
        im = im.crop((left, top, right, bottom))
        imd = imd.crop((left, top, right, bottom))
        
        image_name = "image"
        depth_name = "_depth"
        text_box = "cpos.txt"
        file = open(data_aug+image_name+str(index)+text_box, "w")
        im.save("augment.png") #don't change so there's only of the augmented thing
        imd.save("augmentd.png")
        image = cv2.imread("augment.png")
        imaged = cv2.imread("augmentd.png")
        image[np.where((image==[0,0,0]).all(axis=2))] = [255,255,255]
        mean = cv2.mean(imaged)
        imaged[np.where((imaged==[0,0,0]).all(axis=2))] = [mean[0],mean[1],mean[2]]
        #mean = cv2.mean(image)
        #meand = cv2.mean(imaged)
        #border_image = cv2.copyMakeBorder(image, top=int(top),bottom=int(top),left=int(left),right=int(left),borderType=cv2.BORDER_CONSTANT,value=[int(mean[0]), int(mean[1]), int(mean[2])])  
        #border_imaged = cv2.copyMakeBorder(imaged, top=int(top),bottom=int(top),left=int(left),right=int(left),borderType=cv2.BORDER_CONSTANT,value=[int(meand[0]), int(meand[1]), int(meand[2])])
        cv2.imwrite(data_aug+image_name+str(index)+".png", image)
        cv2.imwrite(data_aug+image_name+depth_name+str(index)+".png", imaged)
        
        
        #file = open("./Labels/img"+str(index)+"aug"+str(i)+".txt", "w")
        #im.save("./Images/img"+str(index)+"aug"+str(i)+".png")
        #imd.save("./Images/imgd"+str(index)+"aug"+str(i)+".png")
        #image = cv2.imread("./Images/img"+str(index)+"aug"+str(i)+".png")        
        bbox = []
        for grasp in grasps:
            try:
                x = float(grasp[0])
                y = float(grasp[1])
                theta = float(grasp[2])
                h = float(grasp[3])
                w = float(grasp[4])
            except:
                print("another error")
            tau = -1*math.degrees(np.arctan2(-1*(y-yc),(x-xc)))
            r = math.sqrt((yc-y)**2+(xc-x)**2)
            theta_prime = (theta + delta_tau) #the new angle for the box
            delta_x = r*math.cos(math.radians(tau+delta_tau))
            delta_y = r*math.sin(math.radians(tau+delta_tau))
            cur_x = xc + delta_x
            cur_y = yc + delta_y
            x_temp = cur_x - dx
            y_temp = cur_y - dy
            x_prime = x_temp# - left
            y_prime = y_temp# - top
            if grasp_to_bbox(x_prime, y_prime, theta_prime, h, w, image) != True:
                box = grasp_to_bbox(x_prime, y_prime, theta_prime, h, w, image)  
                bbox.append(box)
          
            file.writelines([("%.2f"%box[0])," ",("%.2f"%box[1]),"\n"])
            file.writelines([("%.2f"%box[2])," ",("%.2f"%box[3]),"\n"])
            file.writelines([("%.2f"%box[4])," ",("%.2f"%box[5]),"\n"])
            file.writelines([("%.2f"%box[6])," ",("%.2f"%box[7]),"\n"])
        file.close()
        disp(bbox, image)
        #cv2.imshow(data_aug+image_name+str(index)+"no"+str(i)+".png", image)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()     
        index += 1

dataset = './cornell_grasping_dataset/'
data_aug = './cgd_test/'
folders = range(1,11)
folders = ['0'+str(i) if i<10 else '10' for i in folders]
filenames = []

for i in folders:
    for name in glob.glob(os.path.join(dataset, i, 'pcd'+i+'*wb.png')): 
        filenames.append(name) #searches for all the images that end with *r.png

np.random.shuffle(filenames)

index = 0
for filename in filenames:
    depth_image = filename[:-6]+'_depth.png' #depth image without the white bit
    box_text = filename[:-6]+'cpos.txt'
    print(filename, depth_image, box_text)
    img = Image.open(filename)
    imgd = Image.open(depth_image)
    with open(box_text) as pcd_file:
        lines = [line.strip().split(" ") for line in pcd_file.readlines()]
        augment(img, imgd, lines, data_aug)
        #print(index)
#try to use a global variable to do the counting up instead so names are image+number
        #and image_depth+number and image+number+cpos


