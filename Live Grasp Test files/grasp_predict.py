#!/usr/bin/env python

import tensorflow as tf
from grasp_inf import inference #that bit that makes the predictions
import numpy as np
import cv2

#filepath = 'pcd0416r.png'
#filepath_depth = 'pcd0416_depth_image.png'

def draw_bbox(img, bbox):
    p1 = (int(float(bbox[0][0]) / 0.35), int(float(bbox[0][1]) / 0.47)) #0.35 and 0.47 just scales the images back to 640 x 480
    p2 = (int(float(bbox[1][0]) / 0.35), int(float(bbox[1][1]) / 0.47))
    p3 = (int(float(bbox[2][0]) / 0.35), int(float(bbox[2][1]) / 0.47))
    p4 = (int(float(bbox[3][0]) / 0.35), int(float(bbox[3][1]) / 0.47))
    #print(bbox)
    #print("\n")
    #print(p1,p2,p3,p4)
    cv2.line(img, p1, p2, (0, 0, 255))
    cv2.line(img, p2, p3, (255, 0, 0))
    cv2.line(img, p3, p4, (0, 0, 255))
    cv2.line(img, p4, p1, (255, 0, 0))
    #Overlays the grasping over the image
    
def grasp_to_bbox(x, y, tan, h, w):
    # converts the g representations grasps to bounding boxes
    theta = tf.atan(tan)
    edge1 = (x -w/2*tf.cos(theta) +h/2*tf.sin(theta), y -w/2*tf.sin(theta) -h/2*tf.cos(theta)) #(x,y) coordinates
    edge2 = (x +w/2*tf.cos(theta) +h/2*tf.sin(theta), y +w/2*tf.sin(theta) -h/2*tf.cos(theta))
    edge3 = (x +w/2*tf.cos(theta) -h/2*tf.sin(theta), y +w/2*tf.sin(theta) +h/2*tf.cos(theta))
    edge4 = (x -w/2*tf.cos(theta) -h/2*tf.sin(theta), y -w/2*tf.sin(theta) +h/2*tf.cos(theta))
    #print(edge1)
    return [edge1, edge2, edge3, edge4] #should be caled vertices?


def transfer_to_robot_inv_kine(x,y,tan,h,w, sess):
    xc = sess.run(x[0])/0.35
    yc = sess.run(y[0])/0.47
    theta = np.degrees(sess.run(tf.atan(tan*0.35/0.47)[0]))#scaling isn't 1:1 so needed to scale the angle back to 640x480
    height = sess.run(h[0])
    width = sess.run(w[0])
    print(width, height)
    w = np.sqrt((width*np.cos(theta)/0.35)**2+(width*np.sin(theta)/0.47)**2)
    h = np.sqrt((height*np.sin(theta)/0.35)**2+(height*np.cos(theta)/0.47)**2)
    #this function converts the x,y,thet,w,h from the 224x224 dimensions of the CNN back to 640x480
    print("X coordinate: ",xc,"\nY coordinate: ",yc,"\nAngle: ",theta,"\nGripper opening width: ",w)#,"\nHeight: ",h)
    return [xc, yc, theta, w,h]





def normalize_depth(image_depth):
    normalized_depth = tf.div(
                    tf.subtract(image_depth, 
                            tf.reduce_min(image_depth)), 
                    tf.subtract(tf.reduce_max(image_depth), 
                                tf.reduce_min(image_depth)))
    return normalized_depth

def predict_grasp(filepath, filepath_depth, tp):
    tf.reset_default_graph()
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()) #just initialises variables like before
    sess = tf.Session()
    sess.run(init_op)
    
    img_raw_data = tf.gfile.FastGFile(filepath, 'rb').read() #reads the image
    img_raw_depth = tf.gfile.FastGFile(filepath_depth, 'rb').read()
    img_show = cv2.imread(filepath) #this is just to help display it then?
    
    #This bit decodes and augments the normal image
    img_data = tf.image.decode_png(img_raw_data)
    img_data = tf.image.convert_image_dtype(img_data, dtype=tf.float32)
    img_reshape = tf.image.resize_images(img_data, [224, 224])
    img_reshape = tf.reshape(img_reshape, shape=[1, 224, 224, 3])
    
    #This bit is for decoding the depth data
    img_depth = tf.image.decode_png(img_raw_depth)
    img_depth = tf.image.convert_image_dtype(img_depth, dtype=tf.float32)
    img_depth_reshape = tf.image.resize_images(img_depth, [224, 224])
    img_depth_reshape = tf.reshape(img_depth_reshape, shape=[1, 224, 224, 3])
    img_depth_reshape = tf.image.rgb_to_grayscale(img_depth_reshape) 
    
    rgbd_image = tf.concat([img_reshape, img_depth_reshape], 3)
    #print(rgbd_image.get_shape())
    #rgbd_image = tf.reshape(rgbd_image, shape=[1, 224, 224, 4])
    rgbd_image = normalize_depth(rgbd_image) #should in theory be RGBD with pixel values between 0-1
    
    # Convert from 0-1 to -1 to 1
    rgbd_image = tf.subtract(rgbd_image, 0.5)

    rgbd_image = tf.multiply(rgbd_image, 2.0)
    
    
    #normalise the pixels of the images as well so that it's between -1 and 1? because surely if its been trained on that, then itll work better when its like that
    #Up to this point this essentially, just scales the images
    
    
    
    #Just operations
    x_hat, y_hat, tan_hat, h_hat, w_hat = tf.unstack(inference(rgbd_image), axis=1) #uses the inference function/module
    bbox_hat = grasp_to_bbox(x_hat, y_hat, tan_hat, h_hat, w_hat) #some function that converts to a bounding box
    dg={}
    lg = ['w1', 'b1', 'w2', 'b2', 'w3', 'b3', 'w4', 'b4', 'w5', 'b5', 'w_fc1', 'b_fc1', 'w_fc2', 'b_fc2', 'w_output', 'b_output']
    for i in lg:
        dg[i] = [v for v in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES) if v.name == i+':0'][0]

    saver_g = tf.train.Saver(dg)
    saver_g.restore(sess, '/home/narly/zed_ur_ws_yf/src/ur_rg2/ur_control/scripts/my_models/rgbdstep-6500') #uses this model to make the grasping predictions
    
    #test = sess.run(dg[1])
    #print(test)
    bbox_model = sess.run(bbox_hat) 
    draw_bbox(img_show, bbox_model) #so just overlays the bounding box over the original image (all in one frame)
    #print(sess.run(x_hat.eval())) #this should be the bit that is given to the robot inverse kine script.
    grasp_representation = transfer_to_robot_inv_kine(x_hat, y_hat, tan_hat, h_hat, w_hat, sess)
    save_im = '/home/narly/zed_ur_ws_yf/src/ur_rg2/ur_control/scripts/live_test_image/' + str(tp) +'imagegrasp.png'
    cv2.imwrite(save_im, img_show)
    #cv2.imshow('bbox', img_show)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()   
    return grasp_representation  

if __name__ == '__main__':
     predict_grasp()
