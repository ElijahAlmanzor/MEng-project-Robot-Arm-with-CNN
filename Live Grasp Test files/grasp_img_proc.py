import tensorflow as tf
import numpy as np


#The arguments/some predefined constants so that the rest of the code has 
FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer('image_size', 224,
                            """Provide square images of this size.""")
tf.app.flags.DEFINE_integer('num_preprocess_threads', 12, 
                            """Number of preprocessing threads per tower. """
                            """Please make this a multiple of 4.""")
tf.app.flags.DEFINE_integer('num_readers', 12,
                            """Number of parallel readers during train.""")
tf.app.flags.DEFINE_integer('input_queue_memory_factor', 12,
                            """Size of the queue of preprocessed images. """
                            """Default is ideal but try smaller values, e.g. """
                            """4, 2 or 1, if host memory is constrained. See """
                            """comments in code for more details.""")
#literally just a module of functions
#functions to load the data and do some funky image pre-processing

def parse_example_proto(examples_serialized):
    #First bit that needs to be modified
    feature_map={
        'image/encoded': tf.FixedLenFeature([], dtype=tf.string,
                                            default_value=''),
        'image/encoded_depth': tf.FixedLenFeature([], dtype=tf.string,default_value=''),
        'bboxes': tf.VarLenFeature(dtype=tf.float32)
        } #features
    
    features=tf.parse_single_example(examples_serialized, feature_map)  #converts the serialized_example curr in the queue, into the feature map
    bboxes = tf.sparse_tensor_to_dense(features['bboxes']) #take the bboxes label from the features
    
    r = 8*tf.random_uniform((1,), minval=0, maxval=tf.size(bboxes, out_type=tf.int32)//8, dtype=tf.int32)
    bbox = tf.gather_nd(bboxes, [r,r+1,r+2,r+3,r+4,r+5,r+6,r+7]) #this is just the bit that picks the random ground truth everytime
    
    
    return features['image/encoded'],features['image/encoded_depth'], bbox
    #returns the encoded images, and the current randomly picked bbox label

def eval_image(image, height, width):
    image = tf.image.central_crop(image, central_fraction=0.875)
    image = tf.expand_dims(image, 0)
    image = tf.image.resize_bilinear(image, [height, width],
                                     align_corners=False)
    image = tf.squeeze(image, [0])
 
    return image


def distort_color(image, thread_id):
    #This ensures the image is inbetween 0 and 1
    color_ordering = thread_id % 2
    if color_ordering == 0:
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_hue(image, max_delta=0.2)
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
    elif color_ordering == 1:
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_hue(image, max_delta=0.2)
    
    image = tf.clip_by_value(image, 0.0, 1.0) #been normalised to one



   #rgbd_image = tf.concat([image, image_depth], 2)
    return image


def distort_image(image, height, width, thread_id):
    distorted_image = tf.image.random_flip_left_right(image)
    distorted_image = distort_color(distorted_image, thread_id)
    return distorted_image

def normalize_depth(image_depth):
    normalized_depth = tf.div(
                    tf.subtract(image_depth, 
                            tf.reduce_min(image_depth)), 
                    tf.subtract(tf.reduce_max(image_depth), 
                                tf.reduce_min(image_depth)))
    return normalized_depth


#I dont want to modify the values of the depth - because that is just the depth
def distort_images(image, image_depth, height, width, thread_id):
    distorted_image = distort_color(image, thread_id)
    normalized_depth_image = normalize_depth(image_depth)
    rgbd_image = tf.concat([distorted_image, normalized_depth_image], 2)
    #print("HERE IS THE SHAPE OF THE RGBD IMAGE")
    #print(rgbd_image.get_shape())
    #rgbd_image = tf
    rgbd_image = tf.image.random_flip_left_right(rgbd_image)
    return rgbd_image

def image_preprocessing(image_buffer, image_buffer_depth, train, thread_id=0):
    #image buffer is the encoded image, train = true and thread_id is the current thread iteration
    height = FLAGS.image_size
    width = FLAGS.image_size
    #For the rgbd image
    #decode_png returns the PNG-encoded image as a tensor (matrix used by tensorflow)
    image = tf.image.decode_png(image_buffer, channels=3) #decodes the encoded image from the TFRecord, so how to combine this with the depth data?
    image = tf.image.convert_image_dtype(image, dtype=tf.float32) 
    image = tf.image.resize_images(image, [height,width]) #resize the image to 224x224
    #
    image_depth = tf.image.decode_png(image_buffer_depth, channels=1)
    image_depth = tf.image.convert_image_dtype(image, dtype=tf.float32)
    image_depth = tf.image.resize_images(image, [height,width])
    image_depth = tf.image.rgb_to_grayscale(image_depth)
    
    #Now to combine the channels when they are tensors
    #if train:
    #    image = distort_image(image, height, width, thread_id)
    #else:
    #    image = eval_image(image, height, width)
    
    if train:
        image = distort_images(image, image_depth, height, width, thread_id)
    else:
        normalized_depth_image = normalize_depth(image_depth)
        image = tf.concat([image, normalized_depth_image], 2)
    
    
    
    #this rescales from 0-1 to -1to1
    image = tf.subtract(image, 0.5)
    image = tf.multiply(image, 2.0)
    return image
    #Need to link this to the depth data somehow



def batch_inputs(data_files, train, num_epochs, batch_size,
                 num_preprocess_threads, num_readers):
    print(train)
    if train:
        #Create a shuffled FIFOQueue that has a capacity of 16 (with the queue runners automatically)
        filename_queue = tf.train.string_input_producer(data_files,
                                                        num_epochs,
                                                        shuffle=True,
                                                        capacity=16)
    else: #validation code
        filename_queue = tf.train.string_input_producer(data_files,
                                                        num_epochs,
                                                        shuffle=False,
                                                        capacity=1)
    
    examples_per_shard = 1024
    min_queue_examples = examples_per_shard * FLAGS.input_queue_memory_factor
    #before the shuffling and deqeueing can continue
    if train:
        print('pass')
        examples_queue = tf.RandomShuffleQueue(
            capacity=min_queue_examples+3*batch_size,
            min_after_dequeue=min_queue_examples,
            dtypes=[tf.string])
    
    #This all just threading stuff
    else:
        examples_queue = tf.FIFOQueue(
            capacity=examples_per_shard + 3 * batch_size,
            dtypes=[tf.string])

    if num_readers > 1:
        enqueue_ops = []
        for _ in range(num_readers):
            reader = tf.TFRecordReader()
            _, value = reader.read(filename_queue)
            enqueue_ops.append(examples_queue.enqueue([value]))
        tf.train.queue_runner.add_queue_runner(
            tf.train.queue_runner.QueueRunner(examples_queue,enqueue_ops))
        examples_serialized = examples_queue.dequeue()
    
    
    else:
        reader = tf.TFRecordReader()
        _, examples_serialized = reader.read(filename_queue)
    #This all just threading stuff
    
    images_and_bboxes=[]
    
    
    for thread_id in range(num_preprocess_threads):
        image_buffer, image_depth_buffer, bbox = parse_example_proto(examples_serialized) #really just loads the encoded image + picks a random ground truth rectangle
        
        #This is the bit that should be to ensure the RGB + D data are being used to train 
        #CGD has pointclouds as text

        image = image_preprocessing(image_buffer, image_depth_buffer, train, thread_id) #pre-processes the images (distorts them automatically!)
        images_and_bboxes.append([image, bbox])
    
    images, bboxes = tf.train.batch_join(
        images_and_bboxes,
        batch_size=batch_size,
        capacity=2*num_preprocess_threads*batch_size)
    
    height = FLAGS.image_size
    width = FLAGS.image_size
    depth = 4
    
    images = tf.cast(images, tf.float32)
    images = tf.reshape(images, shape=[batch_size, height, width, depth])

    return images, bboxes 


#These two are used in the training
    #Distorted inputs for training
def distorted_inputs(data_files, num_epochs, train=True, batch_size=None):
    with tf.device('/cpu:0'):
        print(train)
        images, bboxes = batch_inputs(
            data_files, train, num_epochs, batch_size,
            num_preprocess_threads=FLAGS.num_preprocess_threads,
            num_readers=FLAGS.num_readers)
  
    return images, bboxes
#returns the pre-processed images and the labels
#data_files contains the TFRecord file of all the images -> has the feature image/encoded_depth

#normal inputs for validation
def inputs(data_files, num_epochs=1, train=False, batch_size=1):
    #1 epoch as only evaluates once, 1 batch size - e.g transferred one by one
    with tf.device('/cpu:0'):
        print(train)
        images, bboxes = batch_inputs(
            data_files, train, num_epochs, batch_size,
            num_preprocess_threads=FLAGS.num_preprocess_threads,
            num_readers=1)
    
    return images, bboxes
