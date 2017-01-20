
# coding: utf-8



import tensorflow as tf
from my_tf_layer import *
from PIL import Image
import numpy as np

from time import gmtime,strftime


class convNet():
    def __init__(self):
        self.nchannel = 1
        self.crop_size = 512
        self.batch = 1

        self.data = tf.placeholder(tf.float32, [self.batch, self.crop_size,self.crop_size,self.nchannel])
        self.label =  tf.placeholder(tf.int32, [self.batch, self.crop_size,self.crop_size])
        #self.expected = tf.expand_dims(self.label, -1)
        self.expected = self.label
        self.device = '/gpu:0'
        self.nclass = 2

        self.build_graph()

    def build_graph(self):
        out_size = 64
        self.conv1_1 = conv2d_layer(self.data,3,out_size,name = 'conv1_1')
        self.conv1_1_bn = batch_norm_layer(self.conv1_1,name = 'conv1_1_bn')

        out_size = 64
        self.conv1_2 = conv2d_layer(self.conv1_1_bn,3,out_size,name = 'conv1_2')
        self.conv1_2_bn = batch_norm_layer(self.conv1_2,name = 'conv1_2_bn')

        out_size = 128
        self.conv1_3 = conv2d_layer(self.conv1_2_bn,3,out_size,name = 'conv1_3')
        self.conv1_3_bn = batch_norm_layer(self.conv1_2,name = 'conv1_3_bn')

        # pool1
        self.pool1 = pool_layer(self.conv1_3_bn,name='pool1')

        out_size = 64
        self.conv2_1 = conv2d_layer(self.pool1,3,out_size,name = 'conv2_1')
        self.conv2_1_bn = batch_norm_layer(self.conv2_1,name = 'conv2_1_bn')

        out_size = 128
        self.conv2_2 = conv2d_layer(self.conv2_1_bn,3,out_size,name = 'conv2_2')
        self.conv2_2_bn = batch_norm_layer(self.conv2_2,name = 'conv2_2_bn')

        out_size = 128
        self.conv2_3 = conv2d_layer(self.conv2_2_bn,3,out_size,name = 'conv2_3')
        self.conv2_3_bn = batch_norm_layer(self.conv2_3,name = 'conv2_3_bn')

        self.pool2 = pool_layer(self.conv2_3_bn,name='pool2')

        out_size = 64
        self.conv3_1 = conv2d_layer(self.pool2,3,out_size,name = 'conv3_1')
        self.conv3_1_bn = batch_norm_layer(self.conv3_1,name = 'conv3_1_bn')

        out_size = 128
        self.conv3_2 = conv2d_layer(self.conv3_1_bn,3,out_size,name = 'conv3_2')
        self.conv3_2_bn = batch_norm_layer(self.conv3_2,name = 'conv3_2_bn')

        out_size = 128
        self.conv3_3 = conv2d_layer(self.conv3_2_bn,3,out_size,name = 'conv3_3')
        self.conv3_3_bn = batch_norm_layer(self.conv3_3,name = 'conv3_3_bn')

        # un pooling layer
        self.un_pool2 = up_pool_deconv_layer(self.conv3_3_bn,2,name = 'unpool2')

        out_size = 64
        self.conv4_1 = conv2d_layer(self.un_pool2,3,out_size,name = 'conv4_1')
        self.conv4_1_bn = batch_norm_layer(self.conv4_1,name = 'conv4_1_bn')

        out_size = 128
        self.conv4_2 = conv2d_layer(self.conv4_1_bn,3,out_size,name = 'conv4_2')
        self.conv4_2_bn = batch_norm_layer(self.conv4_2,name = 'conv4_2_bn')

        out_size = 128
        self.conv4_3 = conv2d_layer(self.conv4_2_bn,3,out_size,name = 'conv4_3')
        self.conv4_3_bn = batch_norm_layer(self.conv4_3,name = 'conv4_3_bn')

        self.un_pool1 = up_pool_deconv_layer(self.conv4_3_bn,2,name='unpool1')

        out_size = 64
        self.conv5_1 = conv2d_layer(self.un_pool1,3,out_size,name = 'conv5_1')
        self.conv5_1_bn = batch_norm_layer(self.conv5_1,name = 'conv5_1_bn')

        out_size = 128
        self.conv5_2 = conv2d_layer(self.conv5_1_bn,3,out_size,name = 'conv5_2')
        self.conv5_2_bn = batch_norm_layer(self.conv5_2,name = 'conv5_2_bn')

        out_size = 128
        self.conv5_3 = conv2d_layer(self.conv5_2_bn,3,out_size,name = 'conv5_3')
        self.conv5_3_bn = batch_norm_layer(self.conv5_3,name = 'conv5_3_bn')

        out_size = 64
        self.conv5_4 = conv2d_layer(self.conv5_3_bn,3,out_size,name = 'conv5_4')
        self.conv5_4_bn = batch_norm_layer(self.conv5_4,name = 'conv5_4_bn')

        self.score = conv2d_layer(self.conv5_4_bn,1,self.nclass,name='score')

        self.logits = tf.reshape(self.score,(-1,2))
        print 'loghts size: ',self.logits.get_shape()


        self.cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(self.logits, tf.reshape(self.expected, [-1]), name='x_entropy')
        print 'cross entropy size: ',self.cross_entropy.get_shape()
        self.loss = tf.reduce_mean(self.cross_entropy)

        rate = 0.0001
        self.optimize = tf.train.AdamOptimizer(rate,0.5,name = 'optimize').minimize(self.loss)
        #optimize = tf.train.GradientDescentOptimizer(learning_rate=0.005).minimize(loss)
        th = 0.5
        self.prediction = tf.argmax(tf.reshape(tf.nn.softmax(self.logits), tf.shape(self.score)), dimension=3)
        print 'prediction size: ',self.prediction.get_shape()
        self.accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.cast(self.prediction,tf.int32),self.expected),tf.float32))

        print 'Graph build done'


    def init_session(self,model = None):
        '''
        If a model file is specified, will restore the model
        '''
        if model == None:
            self.sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
            init = tf.global_variables_initializer()
            self.sess.run(init)
        else:
            self.sess = tf.Session()
            saver = tf.train.Saver()
            saver.restore(self.sess,model)
            print "Model: {} restored".format(model)


    def feed_data(self,data,label,info=False):
        self.sess.run(self.optimize,feed_dict = {self.data:data,self.label:label})
        if info:
            return self.sess.run(self.accuracy,feed_dict = {self.data:data,self.label:label})

    def save_checkpoint(self,path):
        save_path = path
        saver = tf.train.Saver()
        save_path = saver.save(self.sess,save_path)
        print 'Model saved in: ',save_path


    def restore(self,model):
        self.sess = tf.Session()
        saver = tf.train.Saver()
        md = saver.restore(self.sess,model)
        print "Model: {} restored".format(md)


    def predict(self,data):
        return self.sess.run(self.prediction,feed_dict = {self.data:data})


    def list_operations(self):
        #[n.name for n in tf.get_default_graph().as_graph_def().node]
        gf = tf.get_default_graph()
        [n.name for n in gf.get_operations()]


    def get_operation(self,name):
        op = tf.get_default_graph().get_operation_by_name(name)
        op.values()