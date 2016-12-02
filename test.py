import os,time
from glob import glob
import tensorflow as tf
from ops import *
from utils import *
from network import networks
class EVAL(object):
    def __init__(self, sess,batch_size=1,ir_image_shape=[64, 64,1], normal_image_shape=[64, 64, 3],
                 df_dim=64,dataset_name='default',checkpoint_dir=None):

        self.sess = sess
        self.batch_size = batch_size
        self.normal_image_shape = normal_image_shape
        self.ir_image_shape = ir_image_shape
        self.df_dim = df_dim
        self.dataset_name = dataset_name
        self.checkpoint_dir = checkpoint_dir
        self.build_model()

    def build_model(self):

        self.ir_images = tf.placeholder(tf.float32, [self.batch_size] + self.ir_image_shape,
                                    name='ir_images')
        self.normal_images = tf.placeholder(tf.float32, [self.batch_size] + self.normal_image_shape,
                                    name='normal_images')
	self.keep_prob = tf.placeholder(tf.float32)

	net  = networks(self.batch_size,self.df_dim)
        self.G = net.generator(self.ir_images,self.keep_prob)
        self.sampler = net.sampler(self.ir_images,self.keep_prob)
        self.saver = tf.train.Saver()


    def load(self, checkpoint_dir,model):
        print(" [*] Reading checkpoints...")

        #model_dir = "%s_%s" % (self.dataset_name, 32)
        model_dir = "%s" % (self.dataset_name)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)
	#model_path = os.path.join(checkpoint_dir,model)
	if os.path.isfile(os.path.join(checkpoint_dir,model)):
	    print(' Success load network ')
	    self.saver.restore(self.sess, os.path.join(checkpoint_dir, model))
	    return True
	else:
	    print('Fail to load network')
	    return False
