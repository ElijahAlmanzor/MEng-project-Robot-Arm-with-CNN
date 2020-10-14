#!/usr/bin/env python2



"""OpenCV feature detectors with ros CompressedImage Topics in python.

This example subscribes to a ros topic containing sensor_msgs 
CompressedImage. It converts the CompressedImage into a numpy.ndarray, 
then detects and marks features in that image. It finally displays 
and publishes the new image - again as CompressedImage topic.
"""
__author__ =  'Mojtaba Ahmadieh Khanesar <ezzma5 at exmail.nottingham.ac.uk>'
__version__=  '0.1'
__license__ = 'BSD'
# Python libs
import sys, time

# numpy and scipy
import numpy as np
from scipy.ndimage import filters
import message_filters

# OpenCV


# Ros libraries
import roslib
import rospy


global flag1
flag1=0

global flag2
flag2=0

global tp
tp = 0

from math import pi
import numpy as np
from numpy import linalg
from geometry_msgs.msg import *
import time
import os
import pypcd

import math

from std_msgs.msg import String, Float32

import cv2
from cv_bridge import CvBridge, CvBridgeError
#CvBridge = cv_bridge.CvBridge()


from sensor_msgs.msg import PointCloud2
from sensor_msgs import point_cloud2
# Ros Messages
from sensor_msgs.msg import Image
#

global xx
xx=-1


global rrjj
rrjj=20

def depth_image(data):

    global xx
    global flag1
    try:
	#bridge = CvBridge()
	cv_image = CvBridge().imgmsg_to_cv2(data, "32FC1")

	if flag1==1:
	      
              depth_array = np.array(cv_image, dtype = np.dtype('f8'))
	      dnorm = cv2.normalize(depth_array, depth_array, 0, 255, cv2.NORM_MINMAX)
	      num = xx-1
              #/home/narly/zed_ur_ws/BSD/
	      imaged=('/home/narly/zed_ur_ws_yf/src/ur_rg2/ur_control/scripts/live_test_image/image_depth%d.png'%(num))# % (tp))#
	      small = cv2.resize(dnorm, (640, 480))
	      #for some dumb reason depth image 1 is just black but the rest are not
#wait until image0 and depth0 have been taken!
	      cv2.imwrite(imaged, small)
    except CvBridgeError as e:
	print(e)
    flag1 = 0

    


def image(frame1):
    global xx
    
    bridge = CvBridge()
    cv_image = bridge.imgmsg_to_cv2(frame1, "bgr8")
    global flag2
    if flag2==1:

        image=('/home/narly/zed_ur_ws_yf/src/ur_rg2/ur_control/scripts/live_test_image/image%d.png'%(xx) )
        im=('/home/narly/zed_ur_ws_yf/src/ur_rg2/ur_control/scripts/live_test_image/%dimage.png' % (tp))
	small = cv2.resize(cv_image, (640, 480))
	#cv2.imshow("", small)
	#cv2.waitKey()
        #cv2.destroyAllWindows()
        cv2.imwrite(image, small)
        cv2.imwrite(im, small)
        flag2=0
	#return





    
    





def record_motion():
    global flag
    global flag2
    print 'record motion'
    global xx
    global rrjj
    global flag
    global rr_back
    rr_back=60
    for kkll in range(0,2):
        print xx, kkll





        flag1=1
        flag2=1
        rospy.Subscriber('/zed/zed_node/depth/depth_registered', Image,depth_image)
        rospy.Subscriber('/zed/zed_node/left/image_rect_color', Image, image)
        rospy.sleep(5)
    xx=xx+1
    





    






def take_photo(test_position):

        global client
        global pos_x
        global pos_y
        global pos_z
        global recording
        global recorded
        global button
        global flag1
        global flag2
        global xx
        xx = -1
	global tp
	tp = test_position
        rospy.init_node("test_move", anonymous=True, disable_signals=True)
	count = -1
        while(count < 0):
            inp1 = raw_input("take picture? y/n: ")[0]
            if inp1 != 'n':
		record_motion()
		flag1=1
			#xx = -1
		flag2=1
			#xx = -1
		
		count += 1
	    else:
		break
		
        print("Finished taking images")
    
if __name__ == '__main__': take_photo()


