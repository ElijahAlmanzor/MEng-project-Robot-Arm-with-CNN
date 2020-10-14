# -*- coding: utf-8 -*-
import argparse
import tensorflow as tf
import numpy as np
from shapely.geometry import Polygon
from grasp_inf import inference_cont #Inference - in other words contains the model of the neural network
import time




#The arguments/some predefined constants so that the rest of the code has 

def bboxes_to_grasps(bboxes):
    # converting and scaling bounding boxes into grasps, g = {x, y, tan, h, w} 
    box = tf.unstack(bboxes, axis=1) #essentially just a 1D list
    x = (box[0] + (box[4] - box[0])/2) * 0.35 #0.35 and 0.47 are just scaling factors
    y = (box[1] + (box[5] - box[1])/2) * 0.47
    tan = (box[3] -box[1]) / (box[2] -box[0]) *0.47/0.35
    w = tf.sqrt(tf.pow((box[2] - box[0])*0.35, 2) + tf.pow((box[3] - box[1])*0.47, 2)) #euclidean distance to get h and w
    h = tf.sqrt(tf.pow((box[6] - box[0])*0.35, 2) + tf.pow((box[7] - box[1])*0.47, 2))
    '''
    #Switched w and h as want to make width to be the one with respect to the angle of rotation
    #The actual dimensions of the images are 640x480
    #The input to the CNN are 224x224
    #hence in the x direction 224/640 it is scaled down to 0.35
    #y direction it is 224/480 = scaled down to 0.47
    '''
    return x, y, tan, h, w

def grasp_to_bbox(x, y, tan, h, w):
    # converts the g representations grasps to bounding boxes
    theta = tf.atan(tan)
    edge1 = (x -w/2*tf.cos(theta) +h/2*tf.sin(theta), y -w/2*tf.sin(theta) -h/2*tf.cos(theta)) 
    edge2 = (x +w/2*tf.cos(theta) +h/2*tf.sin(theta), y +w/2*tf.sin(theta) -h/2*tf.cos(theta))
    edge3 = (x +w/2*tf.cos(theta) -h/2*tf.sin(theta), y +w/2*tf.sin(theta) +h/2*tf.cos(theta))
    edge4 = (x -w/2*tf.cos(theta) -h/2*tf.sin(theta), y -w/2*tf.sin(theta) +h/2*tf.cos(theta))
    return [edge1, edge2, edge3, edge4] 

def parse_example_proto(examples_serialized):
    feature_map={
        'image/encoded': tf.FixedLenFeature([], dtype=tf.string,
                                            default_value=''),
        'image/encoded_depth': tf.FixedLenFeature([], dtype=tf.string,default_value=''),
        'bboxes': tf.VarLenFeature(dtype=tf.float32)
        } 
    features=tf.parse_single_example(examples_serialized, feature_map)  
    bboxes = features['bboxes'] 
    return features['image/encoded'],features['image/encoded_depth'], bboxes
 
def normalize_depth(image_depth):
    normalized_depth = tf.div(
                    tf.subtract(image_depth, 
                            tf.reduce_min(image_depth)), 
                    tf.subtract(tf.reduce_max(image_depth), 
                                tf.reduce_min(image_depth)))
    return normalized_depth

def image_preprocessing(image_buffer, image_buffer_depth, train, thread_id=0):
    
    height = 224
    width = 224
    
    image = tf.image.decode_png(image_buffer, channels=3) 
    image = tf.image.convert_image_dtype(image, dtype=tf.float32) 
    image = tf.image.resize_images(image, [height,width]) 
    
    image_depth = tf.image.decode_png(image_buffer_depth, channels=3)
    image_depth = tf.image.convert_image_dtype(image_depth, dtype=tf.float32)
    image_depth = tf.image.resize_images(image_depth, [height,width])
    image_depth = tf.image.rgb_to_grayscale(image_depth)

    normalized_depth_image = normalize_depth(image_depth)
    image = tf.concat([image, normalized_depth_image], 2)
    
    #this rescales from 0-1 to -1to1
    image = tf.subtract(image, 0.5)
    image = tf.multiply(image, 2.0)
    return image


def batch_inputs(data_files, train, num_epochs, batch_size,
                 num_preprocess_threads, num_readers):

    filename_queue = tf.train.string_input_producer(data_files,
                                                        num_epochs,
                                                        shuffle=False,
                                                        capacity=1)
    
    '''examples_per_shard = 1024
    min_queue_examples = examples_per_shard * FLAGS.input_queue_memory_factor
    #before the shuffling and deqeueing can continue
    examples_queue = tf.FIFOQueue( #for the validation, tis not random because want it to unload everything in order!
        capacity=examples_per_shard + 3 * batch_size,
        dtypes=[tf.string])'''

    reader = tf.TFRecordReader()
    _, examples_serialized = reader.read(filename_queue)
    #This all just threading stuff
    
    images_and_bboxes=[]
    
    
    for thread_id in range(num_preprocess_threads):
        image_buffer, image_depth_buffer, bbox = parse_example_proto(examples_serialized) #really just loads the encoded image + picks a random ground truth rectangle
        image = image_preprocessing(image_buffer, image_depth_buffer, train, thread_id) #pre-processes the images (distorts them automatically!)
        images_and_bboxes.append([image, bbox])
    
    images, bboxes = tf.train.batch_join(
        images_and_bboxes,
        batch_size=batch_size,
        capacity=2*num_preprocess_threads*batch_size)
    
    height = 224
    width = 224
    depth = 4
    
    images = tf.cast(images, tf.float32)
    images = tf.reshape(images, shape=[batch_size, height, width, depth])

    return images, bboxes 


def valid_inputs(data_files, num_epochs=1, train=False, batch_size=1):
    #1 epoch as only evaluates once, 1 batch size - e.g transferred one by one
    with tf.device('/cpu:0'):
        print(train)
        images, bboxes = batch_inputs(
            data_files, train, num_epochs, batch_size,
            num_preprocess_threads=12,
            num_readers=1)
    
    return images, bboxes




def online_validation(number, model_path, valid_no):
    #temp_graph = tf.Graph()
    #tf.reset_default_graph()
    print('Validation inputs - no image processing')
    data_files_ = 'dataset/cornell_grasping_dataset/validation-cgd' 
    images, bboxes = valid_inputs([data_files_]) 


    #x, y, tan, h, w = bboxes_to_grasps(bboxes) 
    x_hat, y_hat, tan_hat, h_hat, w_hat = tf.unstack(inference_cont(images, 'validation'), axis=1) 
  
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

    sess2 = tf.Session()
    sess2.run(init_op)
    
    coord2 = tf.train.Coordinator() #coordinator just helps make sure all the threads created stop together!
    threads = tf.train.start_queue_runners(sess=sess2, coord=coord2) #just starts the queue runners to be created
    
    
    
    
    
    #dg={}
    #lg = ['w1', 'b1', 'w2', 'b2', 'w3', 'b3', 'w4', 'b4', 'w5', 'b5', 'w_fc1', 'b_fc1', 'w_fc2', 'b_fc2', 'w_output', 'b_output']
    #for i in lg:
    #    dg[i] = [v for v in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES) if v.name == i+':0'][0]
        
    saver_g = tf.train.Saver()#dg) 
    #'my_models/rgbd_model/whole100epochs2760steps/rgbdstep-2250'
    saver_g.restore(sess2, model_path)################################################## Need an input here
    error = np.zeros(valid_no) ################################################### Need another input here

    try:
        step = 0
        while not coord2.should_stop():
            start_batch = time.time()
            bbox_hat = grasp_to_bbox(x_hat, y_hat, tan_hat, h_hat, w_hat) #output from inference_cont
            bbox_value, bbox_model, tan_model = sess2.run([bboxes, bbox_hat, tan_hat]) #bbox_value is list value of the box
            #print('here')
            #print(list(bbox_value[1]))
            bbox_value = list(bbox_value[1])
            boxes = list(np.reshape(bbox_value, (-1,8 ))) #now a list of all the possible bounding boxes
            for box in boxes:
                bbox_value = np.reshape(box, -1) #just a list
                bbox_value = [(bbox_value[0]*0.35,bbox_value[1]*0.47),(bbox_value[2]*0.35,bbox_value[3]*0.47),(bbox_value[4]*0.35,bbox_value[5]*0.47),(bbox_value[6]*0.35,bbox_value[7]*0.47)] 
                tan_value = (box[3] - box[1]) / (box[2] -box[0]) *0.47/0.35
                #Takes in the x,y coordinates of the vertices, and creates rectangles from vertices
                p1 = Polygon(bbox_value)
                p2 = Polygon(bbox_model)
                
                #Jaccard Index/ if area is greater than 25% then it counds
                jaccard = p1.intersection(p2).area / (p1.area +p2.area -p1.intersection(p2).area) 
                
                #Also if the angle is within 30 degrees of the randomly picked rectangle then:
                angle_diff = np.abs(np.arctan(tan_model)*180/np.pi -np.arctan(tan_value)*180/np.pi)
                
                
                duration = time.time() -start_batch
                if angle_diff < 30. and jaccard >= 0.25:
                    #Add to the count of the 'correct' box 
                    #count += 1
                    error[step] = 1
                    print('image: %d | duration = %.2f | count = %d | jaccard = %.2f | angle_difference = %.2f' %(step, duration, np.count_nonzero(error == 1), jaccard, angle_diff))
                    break
                    
                    
  
            step +=1
        
    except tf.errors.OutOfRangeError: 
        #print("Number of validation data: ", step)
        error = (valid_no-np.count_nonzero(error == 1))/valid_no
        print("\nError of %.2f%%" % (error))
    finally:
        coord2.request_stop() #stops the threading/queueing
    
    
    coord2.join(threads) #rejoins the threads
    
    sess2.close()
    f = open("./online_val/out"+str(number)+'.txt', 'w')
    f.write("%.2f"%error)
    f.close()
    return error


def main(number, model_path, valid_no):
    online_validation(number, model_path, valid_no)

if __name__ == '__main__':
    #This bit just says if run this code, this will activate run from start (providing its run from this script)
    #Code arguments basically
  
    parser = argparse.ArgumentParser()
    parser.add_argument('number', type=int)
    parser.add_argument('model_path', type=str)
    parser.add_argument('valid_no', type=int)
    args = parser.parse_args()
    main(number=args.number, model_path=args.model_path, valid_no=args.valid_no)

