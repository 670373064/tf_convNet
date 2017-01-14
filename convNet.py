
# coding: utf-8

# In[1]:

import tensorflow as tf
from my_tf_layer import *
from PIL import Image
import numpy as np

from time import gmtime,strftime


seed = 15  # random seed
random.seed(seed)

nchannel = 1
crop_size = 512
batch = 1

data = tf.placeholder(tf.float32, [batch, crop_size,crop_size,nchannel])
x = tf.placeholder(tf.float32, [batch, crop_size,crop_size,nchannel])
label =  tf.placeholder(tf.int32, [1, crop_size,crop_size])
expected = tf.expand_dims(label, -1)
device = '/gpu:0'

out_size = 64
phase_train = True
shape = x.get_shape().as_list()
n_out = shape[3]
beta = tf.Variable(tf.constant(0.0, shape=[n_out]),
                                name='beta', trainable=True)
gamma = tf.Variable(tf.constant(1.0, shape=[n_out]),
                                name='gamma', trainable=True)
batch_mean, batch_var = tf.nn.moments(x, [0,1,2], name='moments')


ema = tf.train.ExponentialMovingAverage(decay=0.5)


def mean_var_with_update():
    ema_apply_op = ema.apply([batch_mean, batch_var])
    with tf.control_dependencies([ema_apply_op]):
        return tf.identity(batch_mean), tf.identity(batch_var)

#phase_train
bb = tf.constant(True)

mean, var = tf.cond( bb,
                    mean_var_with_update,
                    lambda: (ema.average(batch_mean), ema.average(batch_var)))
                    
'''
normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-3,name = "normed")
'''

# enable multiple device
#sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)
#sess.run(tf.global_variables_initializer())
    
iter = 0
idx = random.randint(0, len(train_data)-1)
batch_xs = read_image(train_data[idx][0],size=crop_size)
batch_ys = read_label(train_data[idx][1],size=crop_size)
sess.run(optimize,feed_dict = {data: batch_xs,label: batch_ys})
print 'step{} -- accuracy is {}'.format(iter,sess.run(accuracy,feed_dict = {data: batch_xs,label: batch_ys}))


# In[4]:

for iter in range(50000):
    idx = random.randint(0, len(train_data)-1)
    rotate = random.randint(0,360)
    batch_xs = read_image(train_data[idx][0],rotate = rotate,size=crop_size)
    batch_ys = read_label(train_data[idx][1],rotate = rotate,size=crop_size)
    sess.run(optimize,feed_dict = {data: batch_xs,label: batch_ys})
    if (iter+1)%5 == 0:
        print '{}-step{} -- accuracy is {}'.format(iter,sess.run(strftime("%a, %d %b %Y %H:%M:%S", gmtime()),
                                                                 accuracy,
                                                                 feed_dict = {data: batch_xs,label: batch_ys}))
        strftime("checkpoint/%H-%M-%S.ckpt",gmtime())
        saver = tf.train.Saver()
        save_path = saver.save(sess,'checkpoint/tmp.ckpt')
        print("Model saved in file: %s" % save_path)
        ss = score.eval(session = sess,feed_dict = {data: batch_xs,label: batch_ys})
        ss2 = ss.reshape(-1,crop_size,crop_size)
        show_image(ss2[0])


ss = score.eval(session = sess,feed_dict = {data: batch_xs,label: batch_ys})
ss2 = ss.reshape(-1,512,512)
truth = batch_ys.reshape(512,512)
show_image(ss2[0],truth)





