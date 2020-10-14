#!/usr/local/bin/python
''' 
Training a network on cornell grasping dataset for detecting grasping positions. This is the bit that trains everything
Modified version of Nikolla's code to accept depth data as well - concatenates the disparity image with the rgb channels before being fed to CNN
'''
#Test what it's like to train (without pre-training) on pure RGBD images alone
import sys
import argparse
import os.path
import glob
import tensorflow as tf
import numpy as np
from shapely.geometry import Polygon
import grasp_img_proc #File that processes the images
from grasp_inf import inference, inference_cont #Inference - in other words contains the model of the neural network
import time
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

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
y_plot = []
plt.style.use('fivethirtyeight')
fig = plt.figure()
ax1 = fig.add_subplot(1,1,1)
def animate(epoch, loss):
    x_plot.append(epoch)
    y_plot.append(loss)
    plt.cla()
    plt.plot(x_plot, y_plot)


def run_training():
    tf.reset_default_graph()
    print(FLAGS.train_or_validation)
    if FLAGS.train_or_validation == 'train': 
        print('distorted_inputs')
        data_files_ = TRAIN_FILE #datafiles is the actual tfrdata
        images, bboxes = grasp_img_proc.distorted_inputs([data_files_], FLAGS.num_epochs, batch_size=FLAGS.batch_size) 

    else: 
        #Validation dataset
        print('inputs')
        data_files_ = VALIDATE_FILE
        images, bboxes = grasp_img_proc.inputs([data_files_]) 
    
    # These are the labels (uses the grasp_img_proc module with the disorted images), also everything is processed as bboxes as that the labels are in xy coords
    x, y, tan, h, w = bboxes_to_grasps(bboxes) 
    
    
    # These are the outputs of the model - used for the error minimisation (uses the grasp_inf model) - outputs the grasp representation directly
    x_hat, y_hat, tan_hat, h_hat, w_hat = tf.unstack(inference(images), axis=1) 
    
    
    
    # tangent of 85 degree is 11 
    tan_hat_confined = tf.minimum(11., tf.maximum(-11., tan_hat))
    tan_confined = tf.minimum(11., tf.maximum(-11., tan))
    
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
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord) 
    
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
    saver_g = tf.train.Saver(dg) 
    #my_models/test_crash/test_crash_continuation
    if FLAGS.continue_from_trained_model == 'yes':
        #saver.restore(sess, "/root/grasp/grasp-detection/models/imagenet/m2/m2.ckpt")
        saver_g.restore(sess, "my_models/test_save/test_save")
        '''Restores a previously trained model'''
        
    #Just a quick code to determine the ecurrent epoch
    steps_per_epoch = int(708/FLAGS.batch_size)
    model_no = 0
    try:
        count = 0
        step = 0
        epoch = 0
        start_time = time.time()
        while not coord.should_stop():
            start_batch = time.time()
            
            
            
            if FLAGS.train_or_validation == 'train':
                #so everytime the loss, x, x_hat, tan etc are called, their operations as well as their operation flows are also called implicitly - graph flow
                _, loss_value, x_value, x_model, tan_value, tan_model, h_value, h_model, w_value, w_model = sess.run([train_op, loss, x, x_hat, tan, tan_hat, h, h_hat, w, w_hat])
                duration = time.time() - start_batch
                
                #if step % 100 == 0:             
                    #print("Step %d | loss = %s\n | x = %s\n | x_hat = %s\n | tan = %s\n | tan_hat = %s\n | h = %s\n | h_hat = %s\n | w = %s\n | w_hat = %s\n | (%.3f sec/batch\n"%(step, loss_value, x_value[:3], x_model[:3], tan_value[:3], tan_model[:3], h_value[:3], h_model[:3], w_value[:3], w_model[:3], duration))
                    #How come the y values are not included? - does that not matter because the x's are already being called?
                    
                if step % 110 == 0:
                    saver_g.save(sess, "my_models/test_save/test_save"+str(model_no))
                    model_no += 1
                    
                if step % steps_per_epoch == 0:
                    
                    print("Step %d | loss = %s\n | x = %s\n | x_hat = %s\n | tan = %s\n | tan_hat = %s\n | h = %s\n | h_hat = %s\n | w = %s\n | w_hat = %s\n | (%.3f sec/batch\n)"%(step, loss_value, x_value[:3], x_model[:3], tan_value[:3], tan_model[:3], h_value[:3], h_model[:3], w_value[:3], w_model[:3], duration))                    
                    print("Current numbers of epoch: %d"%(epoch))
                    ani = FuncAnimation(fig, animate(epoch,loss_value), 1)
                    epoch += 1
                    plt.tight_layout()
                    plt.show() 
                    #MAKE A LIVE GRAPH MAKE IT EASIER TO SEE
                
            else:
                #VALIDATION 
                '''wont work yet as I have not edited the grasp_img_proc file yet'''
                #Converts output of NN to four corner vertices
                bbox_hat = grasp_to_bbox(x_hat, y_hat, tan_hat, h_hat, w_hat) 
                
                #Gets the value of the actual vertices (randomly), bbox from NN, the actual tan and the predicted tan
                bbox_value, bbox_model, tan_value, tan_model = sess.run([bboxes, bbox_hat, tan, tan_hat])
                
                #Turn the bbox value into a 1D array
                bbox_value = np.reshape(bbox_value, -1) 
                
                #Rescale it to the size of the 224x224 output of the neural net
                bbox_value = [(bbox_value[0]*0.35,bbox_value[1]*0.47),(bbox_value[2]*0.35,bbox_value[3]*0.47),(bbox_value[4]*0.35,bbox_value[5]*0.47),(bbox_value[6]*0.35,bbox_value[7]*0.47)] 
                
                #Takes in the x,y coordinates of the vertices, and creates rectangles from vertices
                p1 = Polygon(bbox_value)
                p2 = Polygon(bbox_model)
                
                #Jaccard Index/ if area is greater than 25% then it counds
                jaccard = p1.intersection(p2).area / (p1.area +p2.area -p1.intersection(p2).area) 
                
                #Also if the angle is within 30 degrees of the randomly picked rectangle then:
                angle_diff = np.abs(np.arctan(tan_model)*180/np.pi -np.arctan(tan_value)*180/np.pi)
                
                
                duration = time.time() -start_batch
                if angle_diff < 30. and jaccard >= 0.25:
                    #Add to the count of the 'correct' 
                    count+=1
                    print('image: %d | duration = %.2f | count = %d | jaccard = %.2f | angle_difference = %.2f' %(step, duration, count, jaccard, angle_diff))
                    
                    
                    
                    
            step +=1
        
    except tf.errors.OutOfRangeError: #some error
        saver_g.save(sess, "my_models/test_save/test_save"+str(model_no)) #Best to save it again at the end time
        if FLAGS.train_or_validation == 'train': 
            print('Done training for %d epochs, %d steps, %.1f min.' % (FLAGS.num_epochs, step, (time.time()-start_time)/60))
        else:
            #print("Number of validation data: ", step)
            error = ((1-(count/step))*100)
            print("\nError of %.2f%%" % (error))
    finally:
        coord.request_stop() #stops the threading/queueing
        
        

    coord.join(threads) #rejoins the threads

    sess.close()
    return error
     
'''Maybe add a function that evaluates the accuracy of the trained data as well as their confidence levels'''
#Trying to make it so loops the validation several times - so chance to compare with most of the correct rectangles

def main(_):
    if FLAGS.train_or_validation == 'train': 
        run_training()

    else: 
        errors = []
        for i in range(15):
            errors.append(run_training())
        print("The lowest error is:", min(errors),"%")
   
    
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
        default=50,
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
