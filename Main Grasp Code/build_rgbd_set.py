#!/usr/local/bin/python
'''Converts Cornell Grasping Dataset data into TFRecords data format using Example protos.
The raw data set resides in png and txt files located in the following structure:
    
    dataset/03/pcd0302r.png
    dataset/03/pcd0302cpos.txt
Core code written by Nikolla
Modified to be RGBD by Elijah

The RGBD-D Objects are organised into 51 categories
'''


import tensorflow as tf
#import base64
import os
import glob
import numpy as np
import random
from PIL import Image
#import cv2 
#from PIL import Image



class ImageCoder(object):
    def __init__(self):
        self._sess = tf.Session()
        self._decode_png_data = tf.placeholder(dtype=tf.string)
        self._decode_png = tf.image.decode_png(self._decode_png_data, channels=3,dtype = tf.uint8) #3 channels because RGB
    def decode_png(self, image_data):
        return self._sess.run(self._decode_png,
                              feed_dict={self._decode_png_data: image_data})
        
    #Need to find a way to open up RGB pointclouds


def _process_image(filename, coder):
    # Decode the image
    image_data = tf.gfile.FastGFile(filename, 'rb').read()
    image = coder.decode_png(image_data)
    assert len(image.shape) == 3 #assert that 
    height = image.shape[0]
    width = image.shape[1]
    assert image.shape[2] == 3 #just asserts that there are 3 channels (RGB)    
    return image_data, height, width

#Function for processing the depth image
def _process_depth_image(filename, coder):
    image_data = tf.gfile.FastGFile(filename, 'rb').read()
    image = coder.decode_png(image_data)
    assert image.shape[2] == 3
    return image_data
    
def good_image(filename, sess):
    image_data = tf.gfile.FastGFile(filename, 'rb').read()
    try:
        #tf.image.decode_png(image_data)
        sess.run(tf.image.decode_png(image_data))
        return True
    except:
        return False
    
    
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

def _convert_to_example(filename, image_buffer, image_depth_buffer, height, width, label, text):
    # Build an Example proto for an example
    example = tf.train.Example(features=tf.train.Features(feature={
          'image/filename': _bytes_feature(filename.encode()),
          'image/encoded': _bytes_feature(image_buffer), #the actual image is converted to string then into Bytes
          'image/encoded_depth': _bytes_feature(image_depth_buffer),
          'image/height': _int64_feature(height),
          'image/width': _int64_feature(width), #ints are int64
          'image/class/label': _int64_feature(label), #NEED TO ADD THE ARGUMENTS INTO THE FUNCTIONS #label is integer
          'image/class/text': _bytes_feature(text.encode())}))
    return example
    
def test_img(examples_serialized):
    height = 224
    width = 224
    feature_map={
        'image/encoded': tf.FixedLenFeature([], dtype=tf.string, default_value=''),
        'image/encoded_depth': tf.FixedLenFeature([], dtype=tf.string,default_value=''),
        'image/class/label': tf.FixedLenFeature([1], dtype=tf.int64,
                                                default_value=-1),
        'image/class/text': tf.FixedLenFeature([], dtype=tf.string,
                                               default_value='')
        } #Image class text?
    features=tf.parse_single_example(examples_serialized, feature_map)
    try: 
        label = tf.cast(features['image/class/label'], dtype=tf.int32)
        image = features['image/encoded']
        image_depth = features['image/encoded_depth']
        image = tf.image.decode_png(image, channels=3)
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)
        image = tf.image.resize_images(image, [height,width])
        image_depth = tf.image.decode_png(image_depth, channels=3)
        image_depth = tf.image.convert_image_dtype(image_depth, dtype=tf.float32)
        image_depth = tf.image.resize_images(image_depth, [height,width])
        image_depth = tf.image.rgb_to_grayscale(image_depth)
        rgbd_image = tf.concat([image, image_depth], 2)
        return True
    except:
        print('bad image')
        return False


def main():
    #make this close to the _find_image_files function
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    data_dir = './dataset/data_dir'
    labels_file = './lab.txt'
    train_file = os.path.join(data_dir, 'train-rgbd')
    validation_file = os.path.join(data_dir, 'validation-rgbd') #joins the string dataset with train-cgd and validation cgd respectively for the outputs
    writer_train = tf.python_io.TFRecordWriter(train_file)
    writer_validation = tf.python_io.TFRecordWriter(validation_file) #this saves it
   
    unique_labels = [l.strip() for l in tf.gfile.FastGFile(labels_file, 'r').readlines()]
    print(unique_labels) #literally just loads all of the labels
    
    labels = []
    filenames = []
    texts = []
    label_index = 1
    
    
    for text in unique_labels:
        image_file_path = '%s/%s/*depth.png' % (data_dir, text) #better loading depth as its more specified #also the depth images are shit - hoping the other photos are better?
        matching_files = tf.gfile.Glob(image_file_path)
        #print(matching_files)
        labels.extend([label_index] * len(matching_files)) #literally just a list full of the numerical labels
        texts.extend([text] * len(matching_files))
        filenames.extend(matching_files)
        #if not label_index % 100:
        #    print("Finished finding files in %d of %d classes." %label_index, len(labels))
        label_index += 1
        
    #so once it has found all the training data]
    shuffled_index = list(range(len(filenames)))
    random.seed(123456)
    random.shuffle(shuffled_index)
    #print(shuffled_index)
    '''Need to do something about loading all the images from their respective folders'''
    #do something about checking for pngencoding?
    count = 0
    valid_img = 0
    train_img = 0

    coder = ImageCoder()
    
    for i in shuffled_index:
        filename_depth = filenames[i] #Currently just the depth data
        filename = filename_depth[:-10]+'.png'
        label = labels[i]
        text = texts[i]
        #if good_image(filename, sess) == True:      
        image_buffer, height, width = _process_image(filename, coder)
        #else:
        #    print('bad image')
        #    continue
        
        #if good_image(filename_depth, sess) == True:
        image_buffer_depth = _process_depth_image(filename_depth, coder)
        #else:
        #    print('bad image')
        #    continue
        
        example = _convert_to_example(filename, image_buffer, image_buffer_depth, height, width, label, text)
        test = example.SerializeToString()
        
        if test_img(test) ==  False:
            print(filename, "is a bad image")
            continue
        
        if count % 5 == 0:
            writer_validation.write(example.SerializeToString()) #write the example
            valid_img +=1
        else:
            writer_train.write(example.SerializeToString())
            train_img +=1
        print(filename)
        count +=1
        
        
    print('Done converting %d images into TFRecords with %d train images and %d validation images'%(count, train_img, valid_img))
    data_count = open("Output.txt", "w")
    data_count.write("%d\n%d"%(train_img, valid_img))
    data_count.close()
    writer_train.close()
    writer_validation.close()
    #now to train the classification on this data

if __name__ == '__main__':
    main()
