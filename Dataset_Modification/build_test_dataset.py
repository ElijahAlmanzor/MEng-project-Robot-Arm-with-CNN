#!/usr/local/bin/python
'''Converts Cornell Grasping Dataset data into TFRecords data format using Example protos.
The raw data set resides in png and txt files located in the following structure:
    
    dataset/03/pcd0302r.png
    dataset/03/pcd0302cpos.txt
Core code written by Nikolla
Modified to be RGBD, and to load the augmented version of the dataset by Elijah Almanzor

'''


import tensorflow as tf
#import base64
import os
import glob
import numpy as np
#import cv2 
#from PIL import Image
dataset = './cgd_test/'

class ImageCoder(object):
    def __init__(self):
        self._sess = tf.Session()
        self._decode_png_data = tf.placeholder(dtype=tf.string)
        self._decode_png = tf.image.decode_png(self._decode_png_data, channels=3,dtype = tf.uint8) #3 channels because RGB
    def decode_png(self, image_data):
        return self._sess.run(self._decode_png,
                              feed_dict={self._decode_png_data: image_data})
        



def _process_image(filename, coder):
    # Decode the image
    
    #with open(filename, 'rb') as f:
    #   image_data = f.read()
    image_data = tf.gfile.FastGFile(filename, 'rb').read() #Really, just the same thing as above
    image = coder.decode_png(image_data)
    assert len(image.shape) == 3 #assert that 
    height = image.shape[0]
    width = image.shape[1]
    assert image.shape[2] == 3 #just asserts that there are 3 channels (RGB)    
    return image_data, height, width

#Function for processing the depth image
def _process_depth_image(filename):
    image_data = tf.gfile.FastGFile(filename, 'rb').read()
    return image_data
    



def _process_bboxes(name):
    '''Create a list with the coordinates of the grasping rectangles. Every 
    element is either x or y of a vertex.'''
    with open(name, 'r') as f:
        bboxes = list(map(
              lambda coordinate: float(coordinate), f.read().strip().split()))
    return bboxes

def _int64_feature(v):
    if not isinstance(v, list):
        v = [v]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=v))

def _floats_feature(v):
    if not isinstance(v, list):
        v = [v]
    return tf.train.Feature(float_list=tf.train.FloatList(value=v))

def _bytes_feature(v):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[v]))

def _convert_to_example(filename, bboxes, image_buffer, image_depth_buffer, height, width):
    # Build an Example proto for an example
    example = tf.train.Example(features=tf.train.Features(feature={
          'image/filename': _bytes_feature(filename.encode()),
          'image/encoded': _bytes_feature(image_buffer), #the actual image is converted to string then into Bytes
          'image/encoded_depth': _bytes_feature(image_depth_buffer), #there must be a way to only use one channel for training? (so that there's only 4 inputs)
          'image/height': _int64_feature(height),
          'image/width': _int64_feature(width), #ints are int64
          'bboxes': _floats_feature(bboxes)}))
    return example
    
def main():
    
    test_file = os.path.join(dataset,'CGD_test_dataset')
    #validation_file = os.path.join('./dataset', 'validation-cgd-nobg') #joins the string dataset with train-cgd and validation cgd respectively for the outputs
    print(test_file)
    #print(validation_file)
    writer_test = tf.python_io.TFRecordWriter(test_file)
    #writer_validation = tf.python_io.TFRecordWriter(validation_file) #this saves it
   
    # Creating a list with all the image paths
    #folders = range(1,11)
    #folders = ['0'+str(i) if i<10 else '10' for i in folders]
    filenames = []
    '''
    for i in folders:
        for name in glob.glob(os.path.join(dataset, i, 'pcd'+i+'*r.png')): 
            filenames.append(name) #searches for all the images that end with *r.png
    '''
    #dataset = './dataset/CGD_augm
    for name in glob.glob(dataset+'image_depth*.png'):
        filenames.append(name)
   
        
    
   
    # Shuffle the list of image paths
    np.random.shuffle(filenames)
    
    count = 0


    coder = ImageCoder()
    
    
    for filename_depth in filenames:
        filename = filename_depth.replace('_depth','') #The name for the normal image
        bbox = filename[:-4]+'cpos.txt'
        #print(filename)
        #print(filename_depth)
        #print(bbox)
        
        bboxes = _process_bboxes(bbox) 
        
        #Loads then decodes the actual RGB image
        image_buffer, height, width = _process_image(filename, coder)
        
        #Loads then decodes the Greyscale (but RGB cause PNG) image
        image_depth_buffer = _process_depth_image(filename_depth)
        
        #Converts the features into an 'example' so it can be serialised
        example = _convert_to_example(filename, bboxes, image_buffer, image_depth_buffer, height, width)

        writer_test.write(example.SerializeToString())
        print(count)
        count +=1
        
        
    data_num = open("number_of_test.txt", "w")
    data_num.write("%d\n"%count)
    data_num.close()
    print('Done converting %d images in TFRecords with %d test images ' % (count, count))
    #Easiest way would to print it out onto a binary file, have it read by grasp_det at the start then use that to figure out how many images in the TFRecord
    writer_test.close()

    #closes the writing up of the dataset

if __name__ == '__main__':
    main()
#Remember the point-clouds - converted into greyscale - in png - is actually in 3 channels but all channels are the same
    #need to take that into account.