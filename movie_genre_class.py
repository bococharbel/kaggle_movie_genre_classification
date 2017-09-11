##

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
#from sklearn import metrics, cross_validation
#from __future__ import absolute_import
#from __future__ import division
#from __future__ import print_function

import os
import re
import sys
import tensorflow as tf
import tensorlayer as tl
import math
import time
from datetime import datetime
import numpy as np
from tensorflow.contrib.slim.python.slim.nets.inception_v3 import inception_v3_base, inception_v3_arg_scope
from build_model import *
from tensorflow.python.framework import ops
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from pandas.io.parsers import read_csv
#import tarfile
from six.moves import urllib
import logging
from scipy import misc

slim = tf.contrib.slim

train_inception = True
train_inception_learning_rate = 0.0005
sys.path.append(os.path.abspath('..'))
#reload(sys)
#sys.setdefaultencoding('utf-8')
MAX_PREDICTION_PER_IMG=1 #0#5

logger = logging.getLogger(__name__)

IMAGE_SIZE = 299
IMAGE_DEPTH = 3
DIR = "./"
BASE_DIR = './'
TRAIN_DATA_FILE = BASE_DIR + 'MovieGenre.csv'
IMAGE_DIR= BASE_DIR + 'SampleMoviePosters/' 
IMAGE_EXT= ".jpg"
image_shape=(IMAGE_SIZE,IMAGE_SIZE,IMAGE_DEPTH)
IMAGE_SHAPE_RESULT = (IMAGE_SIZE, IMAGE_SIZE, IMAGE_DEPTH)

# Directory to save the model.
MODEL_DIR = DIR + "model"

# Directory for saving and loading model checkpoints.
train_dir = MODEL_DIR + "/train/"
# Inception v3 checkpoint file.

#INCEPTION_CHECKPOINT = DIR + "data/inception_v3.ckpt"
#inception_checkpoint_file = INCEPTION_CHECKPOINT
# Whether to train inception submodel variables. If True : Fine Tune the Inception v3 Model
train_inception = False
# Number of training steps.
number_of_steps = 1000000
NUM_CLASSES = 10
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 16
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 16

# Frequency at which loss and global step are logged.
log_every_n_steps = 1
# Build the model.
mode = "train"
assert mode in ["train", "eval", "inference"]


# Number of examples per epoch of training data.
num_examples_per_epoch = 1000
# Optimizer for training the model.
optimizer = "SGD"
# Learning rate for the initial phase of training.
initial_learning_rate = 2.0
learning_rate_decay_factor = 0.5
num_epochs_per_decay = 8.0
# Learning rate when fine tuning the Inception v3 parameters.
train_inception_learning_rate = 0.0005
# If not None, clip gradients to this value.
clip_gradients = 5.0
# How many model checkpoints to keep.
max_checkpoints_to_keep = 5
MOVING_AVERAGE_DECAY = 0.9999     # The decay to use for the moving average.
SPLIT_RATIO=0.8

batch_size= 16
tf.logging.set_verbosity(tf.logging.INFO) # Enable tf.logging
FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('train_dir', train_dir,
                           """Directory where to write event logs """
                           """and checkpoint.""")

tf.app.flags.DEFINE_integer('batch_size', batch_size,
                            """Number of images to process in a batch.""")

tf.app.flags.DEFINE_boolean('use_fp16', False,
                            """Train the model using fp16.""")

# Create training directory.
if not tf.gfile.IsDirectory(train_dir):
  # if not Directory for saving and loading model checkpoints, create it
  tf.logging.info("Creating training directory: %s", train_dir)
  tf.gfile.MakeDirs(train_dir)


#use fp16 or not
def data_type():
  """Return the type of the activations, weights, and placeholder variables."""
  if FLAGS.use_fp16:
    return tf.float16
  else:
    return tf.float32


#download poster img file if not exists
def maybe_download(imageId, url):
  """Download the data from Yann's website, unless it's already here."""
  if not os.path.exists(IMAGE_DIR):
    os.makedirs(IMAGE_DIR)
  #filepath = os.path.join(IMAGE_DIR, filename)
  urlfilename = url.split('/')[-1].split("?")[0].strip()
  ext=urlfilename.split('.')[-1]
  if not ext:
    ext="jpg"
    print("No extension found {}".format(urlfilename))
  filename= "{}.{}".format(imageId,ext)
  filepath = os.path.join(IMAGE_DIR, filename)
  oldfilepath = os.path.join(IMAGE_DIR, urlfilename)
  #if os.path.exists(filepath):
  #  try:
  #    os.remove(filepath)
  #  except OSError:
  #    pass
  if os.path.exists(oldfilepath):
    os.rename(oldfilepath,filepath)
    #os.system('mv '+oldfilepath+' '+filepath)
  #return filepath 
  if not os.path.exists(filepath):
    def _progress(count, block_size, total_size):
      sys.stdout.write('\r>> Downloading %s %.1f%%' % (filename, float(count * block_size) / float(total_size) * 100.0))
      sys.stdout.flush()
    try:
      filepath, _ = urllib.request.urlretrieve(url, filepath, _progress)
      print()
    except  urllib.error.URLError as e:
      print("URL DOWNLOAD ERROR ({})".format( e.reason),"\n")
      try:
        os.remove(filepath)
      except OSError:
        pass
      except Exception:
        pass
      return None
    except  urllib.error.HTTPError as e:
      print("URL DOWNLOAD ERROR {}".format(e.code),"\n")
      try:
        os.remove(filepath)
      except OSError:
        pass
      except Exception:
        pass
      return None
    except Exception as e:
      print("URL DOWNLOAD ERROR ({0}): {1}".format(e.errno, e.strerror),"\n")
      try:
        os.remove(filepath)
        pass
      except Exception:
        pass
      return None
  statinfo = os.stat(filepath)
  if os.path.exists(filepath) and os.path.isfile(filepath):
    #print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
    return filepath
  #print('Successfully downloaded', filename)
  return None#filepath


#Load train and test csv files
def load_train_or_test(csvfile):
  df = pd.read_csv(csvfile, encoding = "ISO-8859-1")
  #df= df.dropna(axis=0, how='any')
  data_norm = pd.DataFrame(df)
  cols = df.columns
  labelsstr = data_norm["Genre"]
  urls=  data_norm["Poster"]
  imdburl= data_norm["Imdb Link"]
  imageid= data_norm["imdbId"]
  title= data_norm["Title"]
  score= data_norm["IMDB Score"]
  imagespath=[]
  imgext=[]
  #imagesdata=[]
  classes=[]
  labels=[]
  classes.append("0")
  #labelstab2=dict()
  ttotal= len(imageid)
  for i in range(len(imageid)):
    #if i==6000: #0#
    #  break     #0#
    #print("{}/{} getFile {} from {}\n".format(i,ttotal, imageid[i],urls[i]))#0#
    if urls[i]==None or (isinstance(urls[i], float ) and math.isnan(urls[i])):
      continue 
    imgpath=maybe_download(imageid[i], urls[i])
    if imgpath!=None and os.path.isfile(imgpath):
      imagespath.append(imgpath)
      ext=imgpath.split(".")[-1].lower()
      imgext.append(ext)
      if ext!="jpg":
        print("{} ext {}".format(imgpath, ext))
      #imagemat=  misc.imread(imgpath)
      #imagemat = misc.imresize(imagemat, size=image_shape)#resize(imagemat, output_shape=image_shape)
      #imagemat = np.transpose(imagemat, (1, 2, 0))#tf.transpose(imagemat, [1, 2, 0]).eval()
      ##imagemat = np.array(imagemat)
      ##imagemat = np.expand_dims(imagemat, axis=2)#axis=1
      #imagemat = np.expand_dims(imagemat, axis=0)#axis=1
      #imagesdata.append(imagemat)
      thislabels=np.zeros((1,MAX_PREDICTION_PER_IMG),dtype=np.int32)#len(labelspart)))
      thislabels[:]=0
      if labelsstr[i]==None or (isinstance(labelsstr[i], float ) and math.isnan(labelsstr[i])):
        thislabels[:]=0
      else:
        labelspart=labelsstr[i].strip().split("|")
        for j,lab in enumerate(labelspart):
          index= next((k for k, x in enumerate(classes) if x==lab), -1)
          if index<0 :
            classes.append(lab)
            index=len(classes)-1
          thislabels[0,j]=index
          break
      labels.append(thislabels[0,0])#0#labels.append(thislabels)
      #print("imagespath: {} labels: {}".format(len(imagespath), len(labels)))
      #labelstab2[imgpath]= thislabels
  #imagesdata= np.array(imagesdata)

#  return imagesdata, imagespath, labels, labelstab2, classes
  return  imagespath,imgext, labels,  classes#labelstab2

#split dataset into train and test
def split_dataset(images, exttab,labels, split_ratio=0.8):
    #train_img_set = []
    #train_labels_set = []
    #test_img_set = []
    #test_labels_set = []
    split = int(round(len(images) * split_ratio))
    train_img_set=images[0:split]
    train_labels_set=labels[0:split]
    train_ext_set=exttab[0:split]
    test_img_set=images[split:-1]
    test_labels_set=labels[split:-1]
    test_ext_set=exttab[split:-1]
    return train_img_set, train_ext_set, train_labels_set,  test_img_set, test_ext_set, test_labels_set

#read data function (will only call load_train_or_test function)  and split data in test and train datasets
# will do other operation after (not implemented yet) 
def getAllData(): 
  global NUM_CLASSES
  #trainimages, trainimagespath, trainlabels, trainlabelsdict, trainclasses = load_train_or_test(TRAIN_DATA_FILE)
  imagespath,imgext, labels, trainclasses = load_train_or_test(TRAIN_DATA_FILE)
  #trainimagespath,imgext, trainlabels, trainclasses = load_train_or_test(TRAIN_DATA_FILE)
  n_classes = len(trainclasses) + 1
  NUM_CLASSES= n_classes
  trainimagespath, trainimgext, trainlabels,  testimagespath, testimgext, testlabels=split_dataset(imagespath,imgext, labels, SPLIT_RATIO)
  ##for i,lab in enumerate(trainlabels):
  ##    trainlabels[i] = tf.one_hot(trainlabels, n_classes)
  #trainlabels=np.vstack(trainlabels)
  #trainlabels=tf.one_hot(trainlabels, n_classes)
  #trainlabels=np.array(trainlabels)
  #return trainimages, trainimagespath, trainlabels, trainlabelsdict, trainclasses,n_classes
  #return  trainimagespath,imgext, trainlabels,trainclasses, n_classes
  return  trainimagespath, trainimgext, trainlabels,  testimagespath, testimgext, testlabels, trainclasses, n_classes



#read data image data from disk with scipy and resize to 300*300*3
def read_data_in_batchnp(allimagepaths, allimagextension, alllabellist, image_size, batch_size, numstep, max_nrof_epochs,  shuffle, random_flip,
              random_brightness, random_contrast):

  num_examples=len(allimagepaths)
  debut= (numstep * batch_size) % num_examples
  fin =  ((numstep + 1) * batch_size) % num_examples
  image_paths=[]
  label_list=[]
  for i in range(batch_size):
    image_paths.append(allimagepaths[(debut+i)% num_examples])
    label_list.append(alllabellist[(debut+i)% num_examples])

  #images = ops.convert_to_tensor(image_paths, dtype=tf.string)
  #labels = ops.convert_to_tensor(label_list, dtype=tf.int32)


  images_labels = []
  imgs = []
  lbls = []
  for i in range(len(image_paths)):
    try:
      image = misc.imread(image_paths[i])
    except Exception as e:
      continue
    #print("image shape {} {}".format(image.shape, len(image.shape)))
    if len(image.shape)==2:
      #image = np.expand_dims(image, axis=2)#axis=1
      image = np.stack((image,)*IMAGE_DEPTH)
      #print("##new image shape {} {}".format(image.shape, len(image.shape)))
    image = misc.imresize(image, size=image_shape)#np.resize(ds, output_shape=image_shape)
    #image = np.expand_dims(image, axis=0)#axis=1
    #ds= ds[np.newaxis,:]
    #image = tf.random_crop(image, size=[image_size, image_size, IMAGE_DEPTH])
    #image.set_shape((image_size, image_size, IMAGE_DEPTH))
    #image = tf.image.per_image_standardization(image)
    #if random_flip:
    #  image = tf.image.random_flip_left_right(image)
    #if random_brightness:
    #  image = tf.image.random_brightness(image, max_delta=0.3)
    #if random_contrast:
    #  image = tf.image.random_contrast(image, lower=0.2, upper=1.8)
    
    #print("image {}".format(image.get_shape()))
    #image = tf.expand_dims(image, axis=0)#axis=1
    imgs.append(image)
    lbls.append(label_list[i])
    images_labels.append([image, label_list[i]])
    #imgs= ops.convert_to_tensor(imgs, tf.uint8)
    #lbls=ops.convert_to_tensor(lbls, tf.int32)
  while len(imgs)<len(image_paths):
    pos=len(imgs)-1
    lbls.append(lbls[pos])
    imgs.append(imgs[pos])
    images_labels.append([imgs[pos],lbls[pos]])

  image_batch=np.array(imgs)
  label_batch=np.array(lbls)
  #image_batch, label_batch = tf.train.batch_join(images_labels,
  #                                                 batch_size=batch_size,
  #                                                 capacity= batch_size,
  #                                                 enqueue_many=False,
  #                                                 allow_smaller_final_batch=True)
  
  return image_batch, label_batch

#function 
# #using thread to read data image data (not implemented yet)
def read_data(image_paths, imagextension,label_list, image_size, batch_size, max_nrof_epochs, num_threads, shuffle, random_flip,
              random_brightness, random_contrast):
  images = ops.convert_to_tensor(image_paths, dtype=tf.string)
  labels = ops.convert_to_tensor(label_list, dtype=tf.int32)

  # Makes an input queue
  #input_queue = tf.train.slice_input_producer((images, labels), num_epochs=max_nrof_epochs, shuffle=shuffle, )
  input_queue = tf.train.slice_input_producer((images, labels, imagextension), num_epochs=max_nrof_epochs, shuffle=shuffle )

  images_labels = []
  imgs = []
  lbls = []
#  for _ in range(len(num_threads)):
  for _ in range(batch_size*4):
    image, label = read_image_from_disk(filename_to_label_tuple=input_queue)
    if image==None:
      image, label = read_image_from_disk(filename_to_label_tuple=input_queue)
      if image==None:
        continue
    image = tf.random_crop(image, size=[image_size, image_size, IMAGE_DEPTH])
    image.set_shape((image_size, image_size, IMAGE_DEPTH))
    image = tf.image.per_image_standardization(image)
    if random_flip:
      image = tf.image.random_flip_left_right(image)

    if random_brightness:
      image = tf.image.random_brightness(image, max_delta=0.3)

    if random_contrast:
      image = tf.image.random_contrast(image, lower=0.2, upper=1.8)
    
    #print("image {}".format(image.get_shape()))
    imgs.append(image)
    lbls.append(label)
    images_labels.append([image, label])
    
  
  image_batch, label_batch = tf.train.batch_join(images_labels,
                                                   batch_size=batch_size,
                                                   capacity=4 * num_threads,
                                                   enqueue_many=False,
                                                   allow_smaller_final_batch=True)
  return image_batch, label_batch

#multi threaded
# #using thread to read data image data (not implemented yet)
def read_data_inthread(image_paths, imagextension,label_list, image_size, batch_size, max_nrof_epochs, num_threads, shuffle, random_flip,
              random_brightness, random_contrast):
  """
  Creates Tensorflow Queue to batch load images. Applies transformations to images as they are loaded.
  :param random_brightness: 
  :param random_flip: 
  :param image_paths: image paths to load
  :param label_list: class labels for image paths
  :param image_size: size to resize images to
  :param batch_size: num of images to load in batch
  :param max_nrof_epochs: total number of epochs to read through image list
  :param num_threads: num threads to use
  :param shuffle: Shuffle images
  :param random_flip: Random Flip image
  :param random_brightness: Apply random brightness transform to image
  :param random_contrast: Apply random contrast transform to image
  :return: images and labels of batch_size
  """
  images = ops.convert_to_tensor(image_paths, dtype=tf.string)
  labels = ops.convert_to_tensor(label_list, dtype=tf.int32)

  # Makes an input queue
  #input_queue = tf.train.slice_input_producer((images, labels), num_epochs=max_nrof_epochs, shuffle=shuffle, )
  input_queue = tf.train.slice_input_producer((images, labels, imagextension), num_epochs=max_nrof_epochs, shuffle=shuffle )

  images_labels = []
  imgs = []
  lbls = []
  for _ in range(num_threads):
    image, label = read_image_from_disk(filename_to_label_tuple=input_queue)
    if image==None:
      image, label = read_image_from_disk(filename_to_label_tuple=input_queue)
      if image==None:
        continue
    image = tf.random_crop(image, size=[image_size, image_size, IMAGE_DEPTH])
    image.set_shape((image_size, image_size, IMAGE_DEPTH))
    image = tf.image.per_image_standardization(image)
    if random_flip:
      image = tf.image.random_flip_left_right(image)

    if random_brightness:
      image = tf.image.random_brightness(image, max_delta=0.3)

    if random_contrast:
      image = tf.image.random_contrast(image, lower=0.2, upper=1.8)
    
    #print("image {}".format(image.get_shape()))
    imgs.append(image)
    lbls.append(label)
    images_labels.append([image, label])
    
  
  image_batch, label_batch = tf.train.batch_join(images_labels,
                                                   batch_size=batch_size,
                                                   capacity=4 * num_threads,
                                                   enqueue_many=False,
                                                   allow_smaller_final_batch=True)
  return image_batch, label_batch


#get file extension
def get_extension(filename):
    basename = os.path.basename(filename)  # os independent
    #ext = '.'.join(basename.split('.')[1:])
    ext = basename.split('.')[-1]
    return  ext if ext else None

# read poster img files from disk in a queue (not yed finished)
def read_image_from_disk(filename_to_label_tuple):
  """
  Consumes input tensor and loads image
  :param filename_to_label_tuple: 
  :type filename_to_label_tuple: list
  :return: tuple of image and label
  """
  label = filename_to_label_tuple[1]
  #file_contents = tf.read_file(filename_to_label_tuple[0])
  #example = tf.image.decode_jpeg(file_contents, channels=3)
  encoded_image= filename_to_label_tuple[0]
  #x = tf.Print(encoded_image, [encoded_image, tf.shape(encoded_image)])
  #print("read img {}".format(encoded_image.eval()))
  #print("read img {}".format(tf.string_split(encoded_image,delimiter='.')[-1]))
  #tf.Print(encoded_image)
  image_format=filename_to_label_tuple[2]
  #image_format="jpg"##tf.string_split(encoded_image,delimiter='.')[-1]
  if image_format == "jpeg" or image_format == "jpg" or image_format == "JPEG" or image_format == "JPG"  :
    try:
      image = tf.image.decode_jpeg(encoded_image, channels=IMAGE_DEPTH)
    except Exception as e:
      return None, label
  elif image_format == "png":
    image = tf.image.decode_png(encoded_image, channels=IMAGE_DEPTH)
  elif image_format == "gif":
    image = tf.image.decode_gif(encoded_image)
    image = image[0]
  else:
    raise ValueError("Invalid image format: %s" % image_format)
  
  image = tf.image.convert_image_dtype(image, dtype=tf.float32)

  return image, label

#Train function 
# Read data, create insception model and train
def train():
  g = tf.Graph()
  with g.as_default():
    # with tf.device('/cpu:0'):
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
    #sess=  tf.InteractiveSession()
    print("tl : Sets up the Global Step")
    global_step = tf.Variable(
      initial_value=0,
      dtype=tf.int32,
      name="global_step",
      trainable=False,
      collections=[tf.GraphKeys.GLOBAL_STEP, tf.GraphKeys.VARIABLES])
    #global_step = tf.contrib.framework.get_or_create_global_step()
    print("tl : Movie poster classification model based on inception 3")
    #trainimages, trainimagespath, trainlabels, trainlabelsdict, trainclasses,n_classes= getAllData()
    #trainimagespath, imgext, trainlabels, trainclasses, n_classes= getAllData()
    trainimagespath, trainimgext, trainlabels,  testimagespath, testimgext, testlabels, trainclasses, n_classes= getAllData()
    num_classes=n_classes
    print("trainimagespath:{} trainlabels:{} ".format(len(trainimagespath), len(trainlabels)))
    imagesinput = tf.placeholder(tf.float32, [batch_size, None, None, IMAGE_DEPTH])
    labels = tf.placeholder(tf.int64, shape=(batch_size))  #shape=(batch_size, num_classes)#tf.sparse_placeholder(tf.float32)

    net_image_embeddings = Build_Inception_Model(mode, imagesinput, train_inception,num_classes=NUM_CLASSES)
    network = net_image_embeddings.outputs
    train_params =  net_image_embeddings.all_params

    #probs = tf.nn.softmax(y)
    #labels = tf.cast(labels, tf.int64)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
       labels=labels, logits=network, name='cross_entropy_per_example')
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
    tf.add_to_collection('losses', cross_entropy_mean)
    # The total loss is defined as the cross entropy loss plus all of the weight
    # decay terms (L2 loss).
    total_loss= tf.add_n(tf.get_collection('losses'), name='total_loss')
    #net_image_embeddings.print_layers()

    tvar = tf.all_variables() # or tf.trainable_variables()
    #for idx, v in enumerate(tvar):
    #  print("  var {:3}: {:15}   {}".format(idx, str(v.get_shape()), v.name))

    # Sets up the function to restore inception variables from checkpoint.  setup_inception_initializer()
    inception_variables = tf.get_collection( tf.GraphKeys.VARIABLES, scope="InceptionV3")

    # Sets up the global step Tensor. setup_global_step()


    # Set up the learning rate.
    learning_rate_decay_fn = None
    if train_inception:
      learning_rate = tf.constant(train_inception_learning_rate)
    
    else:
      # when don't update inception_v3
      learning_rate = tf.constant(initial_learning_rate)
      if learning_rate_decay_factor > 0:
        num_batches_per_epoch = (num_examples_per_epoch / batch_size)
        decay_steps = int(num_batches_per_epoch * num_epochs_per_decay)
      def _learning_rate_decay_fn(learning_rate, global_step):
        return tf.train.exponential_decay(
                learning_rate,
                global_step,
                decay_steps=decay_steps,
                decay_rate=learning_rate_decay_factor,
                staircase=True)
      learning_rate_decay_fn = _learning_rate_decay_fn

        # with tf.device('/gpu:0'):
        # Set up the training ops.
    train_op = tf.contrib.layers.optimize_loss(
            loss=total_loss,
            global_step=global_step,
            learning_rate=learning_rate,
            optimizer=optimizer,
            clip_gradients=clip_gradients,
            learning_rate_decay_fn=learning_rate_decay_fn)

  
    #sess.run(tf.initialize_all_variables())
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    sess.run(init_op)

    if mode != "inference":
      #print("tl : Restore InceptionV3 model from: %s" % inception_checkpoint_file)
      #saver = tf.train.Saver(inception_variables)
      #saver.restore(sess, inception_checkpoint_file)
      print("tl : Restore the lastest ckpt model from: %s" % train_dir)
      try:
        saver = tf.train.Saver()
        saver.restore(sess, tf.train.latest_checkpoint(train_dir)) # train_dir+"/model.ckpt-960000")
      except Exception:
        print("     Not ckpt found")

    # Set up the Saver for saving and restoring model checkpoints.
    saver = tf.train.Saver(max_to_keep=max_checkpoints_to_keep)

  print('Start training') # the 1st epoch will take a while
  coord = tf.train.Coordinator()
  threads = tf.train.start_queue_runners(sess=sess, coord=coord)
  for step in range(sess.run(global_step), number_of_steps+1):
    start_time = time.time()
    imgfeed, labelfeed=read_data_in_batchnp(trainimagespath, trainimgext, trainlabels, image_size=IMAGE_SIZE, batch_size=batch_size, numstep=step, max_nrof_epochs=5,  shuffle=False, random_flip=True, random_brightness=True, random_contrast=True)
    feed = { imagesinput: imgfeed, labels: labelfeed }
    loss, _ = sess.run([total_loss, train_op],  feed_dict=feed) #, feed_dict={net_img_in:batch_images, :batch_labels}
    print("step %d: loss = %.4f (%.2f sec)" % (step, loss, time.time() - start_time))
    if (step % 10000) == 0 and step != 0:
      # save_path = saver.save(sess, MODEL_DIR+"/train/model.ckpt-"+str(step))
      save_path = saver.save(sess, train_dir+"modelmoviestof_"+str(step)+".ckpt", global_step=step)
      tl.files.save_npz(network.all_params , name=train_dir+'model_moviestof.npz')
  coord.request_stop()
  coord.join(threads)

def main(argv=None):  # pylint: disable=unused-argument
  train()

if __name__ == '__main__':
  tf.app.run()
