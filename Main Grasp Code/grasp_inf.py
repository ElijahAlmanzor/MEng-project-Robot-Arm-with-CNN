'''
Inference model for grasping 
'''

import tensorflow as tf
FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_bool('trainable', True,
                            """Computes or not gradients for learning.""")

def conv2d_s2(x, W):
    return tf.nn.conv2d(x, W, strides=[1,2,2,1], padding='SAME') #makes one layer

def conv2d_s1(x, W):
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME') #makes another layer
    
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME') #this is just a pooling layer

#THESE FUNCTIONS ARE NOT ACTUALLY USED
#Some of my modifications
def create_weights(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.05))

def create_biases(size):
    return tf.Variable(tf.constant(0.05, shape=[size]))
    #makes a variable that is initialised to be a constant 0.05
#Model has been modified to be initialised with values from a random distribution

def inference(images):
    #Need to find a way to combine the RGB images with a single channel from the depth image! 
    if FLAGS.trainable:
        keep_prob = 0.5 
    else:
        keep_prob = 1.
    #print('keep_prob = %.1f' %keep_prob)
    
    #tf.variable just creates a variable with the name w1 in the tensorgraph
    w1 = tf.Variable(tf.truncated_normal(shape=[5,5,4,64], stddev=0.05), trainable=FLAGS.trainable, name='w1') 
    b1 = tf.Variable(tf.constant(0.1, shape=[64]), trainable=FLAGS.trainable, name='b1')
    h1 = tf.nn.relu(conv2d_s2(images, w1)+b1) #creates a convolution layer which is then outputted with a relu
    h1_pool = max_pool_2x2(h1) #this layer is then maxed pooled
    
    w2 = tf.Variable(tf.truncated_normal([3,3,64,128], stddev=0.05), trainable=FLAGS.trainable, name='w2')
    b2 = tf.Variable(tf.constant(0.1, shape=[128]), trainable=FLAGS.trainable, name='b2')
    h2 = tf.nn.relu(conv2d_s2(h1_pool,w2)+b2) #takes previous layer as input
    h2_pool = max_pool_2x2(h2)

    w3 = tf.Variable(tf.truncated_normal([3,3,128,128], stddev=0.05), trainable=FLAGS.trainable, name='w3')
    b3 = tf.Variable(tf.constant(0.1, shape=[128]), trainable=FLAGS.trainable, name='b3')
    h3 = tf.nn.relu(conv2d_s1(h2_pool,w3)+b3)
    
    w4 = tf.Variable(tf.truncated_normal([3,3,128,128], stddev=0.05), trainable=FLAGS.trainable, name='w4')
    b4 = tf.Variable(tf.constant(0.1, shape=[128]), trainable=FLAGS.trainable, name='b4')
    h4 = tf.nn.relu(conv2d_s1(h3,w4)+b4)

    w5 = tf.Variable(tf.truncated_normal([3,3,128,256], stddev=0.05), trainable=FLAGS.trainable, name='w5')
    b5 = tf.Variable(tf.constant(0.1, shape=[256]), trainable=FLAGS.trainable, name='b5')
    h5 = tf.nn.relu(conv2d_s1(h4,w5)+b5)
    h5_pool = max_pool_2x2(h5)
    
    #Underneath this are all the fully connected layer bits
    w_fc1 = tf.Variable(tf.truncated_normal([7*7*256,512], stddev = 0.05), trainable=FLAGS.trainable, name='w_fc1')
    b_fc1 = tf.Variable(tf.constant(0.1, shape=[512]), trainable=FLAGS.trainable, name='b_fc1')
    h5_flat = tf.reshape(h5_pool, [-1, 7*7*256])
    h_fc1 = tf.nn.relu(tf.matmul(h5_flat,w_fc1)+b_fc1)
    h_fc1_dropout = tf.nn.dropout(h_fc1, keep_prob)

    w_fc2 = tf.Variable(tf.truncated_normal([512,512], stddev=0.05), trainable=FLAGS.trainable, name='w_fc2')
    b_fc2 = tf.Variable(tf.constant(0.1, shape=[512]), trainable=FLAGS.trainable, name='b_fc2')
    h_fc2 = tf.nn.relu(tf.matmul(h_fc1_dropout, w_fc2)+b_fc2)
    h_fc2_dropout = tf.nn.dropout(h_fc2, keep_prob)
    #Eight hidden layers therefore it is the AlexNet
    
    #5 outputs to correspond to the x,y,tan(not the angle), and width
    w_output = tf.Variable(tf.truncated_normal([512, 5], stddev=0.05), trainable=FLAGS.trainable, name='w_output') #makes weights that are fc 512 to 5 outputs
    b_output = tf.Variable(tf.constant(0.1, shape=[5]), trainable=FLAGS.trainable, name='b_output')
    output = tf.matmul(h_fc2_dropout, w_output)+b_output
    
    return output

def inference_cont(images, train):
    if train == 'train':
        keep_prob = 0.5 
    else:
        keep_prob = 1.
    #print('keep_prob = %.1f' %keep_prob)
    #with tf.variable_scope('cnn', reuse=tf.AUTO_REUSE):
    w1 = tf.get_variable('w1', shape=[5,5,4,64], trainable=FLAGS.trainable)
    b1 = tf.get_variable('b1', initializer=tf.constant(0.1, shape=[64]), trainable=FLAGS.trainable)
    h1 = tf.nn.relu(conv2d_s2(images, w1)+b1)
    h1_pool = max_pool_2x2(h1)
    
    w2 = tf.get_variable('w2', [3,3,64,128], trainable=FLAGS.trainable)
    b2 = tf.get_variable('b2', initializer=tf.constant(0.1, shape=[128]), trainable=FLAGS.trainable)
    h2 = tf.nn.relu(conv2d_s2(h1_pool,w2)+b2)
    h2_pool = max_pool_2x2(h2)

    w3 = tf.get_variable('w3', [3,3,128,128], trainable=FLAGS.trainable)
    b3 = tf.get_variable('b3', initializer=tf.constant(0.1, shape=[128]), trainable=FLAGS.trainable)
    h3 = tf.nn.relu(conv2d_s1(h2_pool,w3)+b3)
    
    w4 = tf.get_variable('w4', [3,3,128,128], trainable=FLAGS.trainable)
    b4 = tf.get_variable('b4', initializer=tf.constant(0.1, shape=[128]), trainable=FLAGS.trainable)
    h4 = tf.nn.relu(conv2d_s1(h3,w4)+b4)

    w5 = tf.get_variable('w5', [3,3,128,256], trainable=FLAGS.trainable)
    b5 = tf.get_variable('b5', initializer=tf.constant(0.1, shape=[256]), trainable=FLAGS.trainable)
    h5 = tf.nn.relu(conv2d_s1(h4,w5)+b5)
    h5_pool = max_pool_2x2(h5)
    
    w_fc1 = tf.get_variable('w_fc1', [7*7*256,512], trainable=FLAGS.trainable)
    b_fc1 = tf.get_variable('b_fc1', initializer=tf.constant(0.1, shape=[512]), trainable=FLAGS.trainable)
    h5_flat = tf.reshape(h5_pool, [-1, 7*7*256])
    h_fc1 = tf.nn.relu(tf.matmul(h5_flat,w_fc1)+b_fc1)
    h_fc1_dropout = tf.nn.dropout(h_fc1, 1.0)
    
    w_fc2 = tf.get_variable('w_fc2', [512,512], trainable=FLAGS.trainable)
    b_fc2 = tf.get_variable('b_fc2', initializer=tf.constant(0.1, shape=[512]), trainable=FLAGS.trainable)
    h_fc2 = tf.nn.relu(tf.matmul(h_fc1_dropout, w_fc2)+b_fc2)
    h_fc2_dropout = tf.nn.dropout(h_fc2, 1.0)

    w_output = tf.get_variable('w_output', [512, 5], trainable=FLAGS.trainable)
    b_output = tf.get_variable('b_output', initializer=tf.constant(0.1, shape=[5]), trainable=FLAGS.trainable)
    output = tf.matmul(h_fc2_dropout, w_output)+b_output
    
    return output

