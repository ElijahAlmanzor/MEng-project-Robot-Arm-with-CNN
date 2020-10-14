#!/usr/local/bin/python
''' 
Training a network on cornell grasping dataset for detecting grasping positions. This is the bit that trains everything
Modified version of Nikolla's code to accept depth data as well - concatenates the disparity image with the rgb channels before being fed to CNN
'''
#Test what it's like to train (without pre-training) on pure RGBD images alone
import sys
import argparse
import os.path
import tensorflow as tf
import numpy as np
from shapely.geometry import Polygon
import grasp_img_proc #File that processes the images
from grasp_inf import inference_cont #Inference - in other words contains the model of the neural network
import time
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import math
import os
#Important module that it uses grasp_img_proc, grasp_inf

#path to dataset
TRAIN_FILE = 'dataset/cornell_grasping_dataset/train-cgd' 
VALIDATE_FILE = 'dataset/cornell_grasping_dataset/validation-cgd' 

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

x_plot = []
norm_loss = []
valid_error = []
plt.style.use('classic')
fig = plt.figure()
ax1 = fig.add_subplot(1,1,1)
def animate(epoch,error, loss):
    x_plot.append(epoch)
    norm_loss.append(loss)
    valid_error.append(error)
    plt.xlabel('Steps')
    plt.ylabel('Loss normalised to 0-1')
    plt.cla()
    plt.plot(x_plot, norm_loss)
    plt.plot(x_plot, valid_error)
    

def confidence_interval(error, valid_no):
    n = valid_no #885 images validation images
    es = error #for a 95 interval
    Zn_95 = 1.96
    '''For Redmons single grasp - image wise split'''
    lower_interval = es - Zn_95*math.sqrt(es*(1-es)/n)
    upper_interval = es + Zn_95*math.sqrt(es*(1-es)/n)
    return lower_interval, upper_interval


def run_training():
    tf.reset_default_graph()


    data_files_ = TRAIN_FILE #datafiles is the actual tfrdata
    images, bboxes = grasp_img_proc.distorted_inputs([data_files_], FLAGS.num_epochs, batch_size=FLAGS.batch_size) 

    # These are the labels (uses the grasp_img_proc module with the disorted images), also everything is processed as bboxes as that the labels are in xy coords
    x, y, tan, h, w = bboxes_to_grasps(bboxes) 
    
    
    # These are the outputs of the model - used for the error minimisation (uses the grasp_inf model) - outputs the grasp representation directly
    x_hat, y_hat, tan_hat, h_hat, w_hat = tf.unstack(inference_cont(images, FLAGS.train_or_validation), axis=1) 
    
    
    
    # tangent of 85 degree is 11 
    tan_hat_confined = tf.minimum(50., tf.maximum(-50., tan_hat))
    tan_confined = tf.minimum(50., tf.maximum(-50., tan))
    
    # Loss function
    gamma = tf.constant(10.)
    # A custom cost function for the box regression - essentially just a custom MSE
    loss = tf.reduce_sum(tf.pow(x_hat -x, 2) +tf.pow(y_hat -y, 2) + gamma*tf.pow(tan_hat_confined - tan_confined, 2) +tf.pow(h_hat -h, 2) +tf.pow(w_hat -w, 2))
    
    # Instead of stochastic gradient descent
    train_op = tf.train.AdamOptimizer(epsilon=0.1).minimize(loss)
    
    # Initiliases the variables to have the values they have been parametised by
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()) 
    
    #Create a session
    sess = tf.Session()
    sess.run(init_op)
    
    #Allows for thread qeueing
    coord = tf.train.Coordinator() #coordinator just helps make sure all the threads created stop together!
    threads = tf.train.start_queue_runners(sess=sess, coord=coord) #just starts the queue runners to be created
    
    #Some stuff related to plotting

    
    #save/restore model
    
    #d are the weights that were used to pre-train on Imagenet! - hence why it doesnt contain w_output and b_output (those would have been classification outputs)
    d={} 
    l = ['w1', 'b1', 'w2', 'b2', 'w3', 'b3', 'w4', 'b4', 'w5', 'b5', 'w_fc1', 'b_fc1', 'w_fc2', 'b_fc2']
    # Iterates through the list l, if its in the GraphKeys, store it in the tuple d
    for i in l: 
        d[i] = [v for v in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES) if v.name == i+':0'][0]

    dg={}
    lg = ['w1', 'b1', 'w2', 'b2', 'w3', 'b3', 'w4', 'b4', 'w5', 'b5', 'w_fc1', 'b_fc1', 'w_fc2', 'b_fc2', 'w_output', 'b_output']
    for i in lg:
        dg[i] = [v for v in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES) if v.name == i+':0'][0]


    # This essentially just saves the current models
    #saver = tf.train.Saver(d)
    saver_g = tf.train.Saver(max_to_keep = 1000) 
    
    
    #my_models/test_crash/test_crash_continuation
    if FLAGS.continue_from_trained_model == 'yes':
        #saver.restore(sess, "/root/grasp/grasp-detection/models/imagenet/m2/m2.ckpt")
        saver_g.restore(sess, 'my_models/rgbd_model/whole100epochs2760steps/rgbdstep-2250')
        '''Restores a previously trained model'''
        
    #Just a quick code to determine the ecurrent epoch
    #steps_per_epoch = int(708/FLAGS.batch_size)
    normaliser_value = 0
    num_val = 175
    try:
        
        step = 0
        #epoch = 0
        start_time = time.time()
        while not coord.should_stop():
            start_batch = time.time()
            
            
            
            #so everytime the loss, x, x_hat, tan etc are called, their operations as well as their operation flows are also called implicitly - graph flow
            _, loss_value, x_value, x_model, tan_value, tan_model, h_value, h_model, w_value, w_model = sess.run([train_op, loss, x, x_hat, tan, tan_hat, h, h_hat, w, w_hat])
            duration = time.time() - start_batch
                
            if step > 0:
                if step == 20:
                    normaliser_value = loss_value
                if step % 20 == 0:
                    path = "my_models/test_save/online_val-"+str(step)
                    saver_g.save(sess, "my_models/test_save/online_val", global_step = step)                    
                    print("Step %d | loss = %s\n | x = %s\n | x_hat = %s\n | tan = %s\n | tan_hat = %s\n | h = %s\n | h_hat = %s\n | w = %s\n | w_hat = %s\n | (%.3f sec/batch\n)"%(step, loss_value, x_value[:3], x_model[:3], tan_value[:3], tan_model[:3], h_value[:3], h_model[:3], w_value[:3], w_model[:3], duration))                    
                    loss_normalised = loss_value/normaliser_value
                    
                    os.system('python grasp_validation.py {} {} {}'.format(step, path, num_val))
                    f = open("./online_val/out"+str(step)+'.txt', 'r')
                    error = float(f.read())     
                    
                    FuncAnimation(fig, animate(step,error,loss_normalised), 1)
                    plt.tight_layout()
                    plt.show() 
            
                    
                    
                    
            step +=1
        
    except tf.errors.OutOfRangeError: #some error
        saver_g.save(sess, "my_models/test_save/last_save") #Best to save it again at the end time
        print('Done training for %d epochs, %d steps, %.1f min.' % (FLAGS.num_epochs, step, (time.time()-start_time)/60))

    finally:
        coord.request_stop() #stops the threading/queueing
        
        

    coord.join(threads) #rejoins the threads

    sess.close()

     
'''Maybe add a function that evaluates the accuracy of the trained data as well as their confidence levels'''
#Trying to make it so loops the validation several times - so chance to compare with most of the correct rectangles

def main(_):
    if FLAGS.train_or_validation == 'train': 
        run_training()

    
if __name__ == '__main__':
    #This bit just says if run this code, this will activate run from start (providing its run from this script)
    #Code arguments basically
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=0.001,
        help='Initial learning rate.'
    )
    parser.add_argument(
        '--data_dir',
        type=str,
        default='dataset/cornell_grasping_dataset/train-cgd',
        help='Directory with training data.'
    )
    parser.add_argument(
        '--num_epochs',
        type=int,
        default=100,
        help='Number of epochs to run trainer.' 
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=64,
        help='Batch size.'
    ) #Number of iterations needed = number of instance/batch size
    parser.add_argument(
        '--log_dir',
        type=str,
        default='my_models',
        help='Tensorboard log_dir.'
    )
    '''
    parser.add_argument(
        '--model_path',
        type=str,
        default='my_models/test_crash/test_crash_continuation',
        help='Variables for the model.'
    
    )''' #During training, the model will be saved here
    parser.add_argument(
        '--train_or_validation',
        type=str,
        default='train',
        help='Train or evaluate the dataset'
    )
    parser.add_argument(
        '--continue_from_trained_model',
        type=str,
        default='no',
        help='Yes to continue training a trained model, no to start from scratch, or for validation'
        #Quick note,if it does crash, must train from scratch (can reload the weights)
        #Due to how TFRecords work
    )
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed) #then runs main give all the inputs
