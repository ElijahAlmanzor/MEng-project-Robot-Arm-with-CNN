#!/usr/bin/env python
import cv2
import numpy as np


def augment():
        
	file_path = '/home/narly/zed_ur_ws_yf/src/ur_rg2/ur_control/scripts/live_test_image/image-1.png'
	file_path_depth = '/home/narly/zed_ur_ws_yf/src/ur_rg2/ur_control/scripts/live_test_image/image_depth-1.png'

	#file_seg = '/home/narly/zed_ur_ws_yf/src/ur_rg2/ur_control/scripts/live_test_image/' +str(test_position) + 'image-1.png'
	#file_seg_depth = '/home/narly/zed_ur_ws_yf/src/ur_rg2/ur_control/scripts/live_test_image/' +str(test_position) + 'image_depth-1.png'

	image_bgr = cv2.imread(file_path)

	#cv2.imshow("", image_bgr)
	#cv2.waitKey(0)
	#cv2.destroyAllWindows()
	x = 240
	width = 350
	y = 30
	height = 350

	image_rgb = image_bgr[y:y+height, x:x+width]
	#cv2.imshow("", image_rgb)
	#cv2.waitKey(0)
	#cv2.destroyAllWindows()
	

	rectangle = (10, 10, width-20, height-20)
	mask = np.zeros(image_rgb.shape[:2], np.uint8)

	bgdModel = np.zeros((1, 65), np.float64)
	fgdModel = np.zeros((1, 65), np.float64)


	cv2.grabCut(image_rgb, # Our image
		mask, # The Mask
		rectangle, # Our rectangle
		bgdModel, # Temporary array for background
		fgdModel, # Temporary array for background
		5, # Number of itesrations
		cv2.GC_INIT_WITH_RECT) # Initiative using our rectangle

	# Create mask where sure and likely backgrounds set to 0, otherwise 1
	mask_2 = np.where((mask==2) | (mask==0), 0, 1).astype('uint8')

	# Multiply image with new mask to subtract background
	image_rgb_nobg = image_rgb * mask_2[:, :, np.newaxis]
	print(image_rgb_nobg.shape)
	image_rgb_nobg[np.where((image_rgb_nobg==[0,0,0]).all(axis=2))] = [255,255,255] #INCLUDE THIS LINE IN THE CGD_AUGM for depth images with mean values instead 255 white
	image_rgb_nobg = cv2.copyMakeBorder(
	image_rgb_nobg,
	top=y,
	bottom=480-(y+height),
	left=x,
	right=640-(x+width),
	borderType=cv2.BORDER_CONSTANT,
	value=[255, 255, 255])
	cv2.imwrite(file_path, image_rgb_nobg)
	#Above is to do with segmenting the RGB image



	image_d = cv2.imread(file_path_depth)
	meand = cv2.mean(image_d)
	imd_cropped = image_d[y:y+height, x:x+width]
  

	bgdModel = np.zeros((1, 65), np.float64)
	fgdModel = np.zeros((1, 65), np.float64)


	cv2.grabCut(imd_cropped, # Our image
		mask, # The Mask
		rectangle, # Our rectangle
		bgdModel, # Temporary array for background
		fgdModel, # Temporary array for background
		5, # Number of itesrations
		cv2.GC_INIT_WITH_RECT) # Initiative using our rectangle

	# Create mask where sure and likely backgrounds set to 0, otherwise 1
	#mask_2 = np.where((mask==2) | (mask==0), 0, 1).astype('uint8')

	# Multiply image with new mask to subtract background
	imd_no_bg = ~imd_cropped * mask_2[:, :, np.newaxis]

	imd_no_bg[np.where((imd_no_bg==[0,0,0]).all(axis=2))] = [meand[0],meand[1],meand[2]]
	imd_no_bg = cv2.copyMakeBorder(
	imd_no_bg,
	top=y,
	bottom=480-(y+height),
	left=x,
	right=640-(x+width),
	borderType=cv2.BORDER_CONSTANT,
	value=[meand[0],meand[1], meand[2]])
	#cv2.imshow('test', image_rgb_nobg)    
	#cv2.waitKey(0)
	#cv2.destroyAllWindows()
	#cv2.imshow('depth', imd_no_bg)    
	#cv2.waitKey(0)
	#cv2.destroyAllWindows()
	#cv2.imwrite(file_path, image_rgb_nobg)
	#Above is to do with segmenting the RGB image
 	

        #image_d = ~image_d
        #image_d = image_d[0:480, 150:640]
        #mean = cv2.mean(image_d)
        #image_d = cv2.copyMakeBorder(
	#image_d,
	#top=0,
	#bottom=0,
	#left=150,
	#right=0,
	#borderType=cv2.BORDER_CONSTANT,
	#value=[mean[0], mean[1], mean[2]])
        cv2.imwrite(file_path_depth, imd_no_bg)

if __name__ == '__main__':
	augment()
