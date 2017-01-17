import tensorflow as tf
import os
import random
import numpy as np
import cv2
import time
from PIL import Image
import matplotlib.pyplot as plt



def weight_variable(shape):
    initial = tf.random_normal(shape, stddev=1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.5, shape=shape)
    return tf.Variable(initial)

def conv2d_layer(x, kernel_size,out_channel, use_gpu = True,stride = 1,name = None,padding='SAME'):
    '''
    x: [batch, in_height, in_width, in_channels]
    W_shape: [filter_height, filter_width, in_channels, out_channels]
    b_shape: in_channels
    '''
    if use_gpu:
        device = '/gpu:0'
    else:
        device = '/cpu:0'
    with tf.device(device):
        sh = x.get_shape().as_list()
        W_shape = [kernel_size,kernel_size,sh[3],out_channel]
        b_shape = out_channel
        W = weight_variable(W_shape)
        b = bias_variable([b_shape])
        ret = tf.nn.relu(tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding=padding) +b , name = name)
        print 'layer: {}  size: {}'.format(name, ret.get_shape())
        return ret

def batch_norm_layer(x,use_gpu = True, phase_train = True ,name = None):
    '''
    Batch normalization
    reference: https://gist.github.com/tomokishii/0ce3bdac1588b5cca9fa5fbdf6e1c412 
    '''
    shape = x.get_shape().as_list()
    n_out = shape[3]
    #if use_gpu:
    #    device = '/gpu:0'
    #else:
    #    device = '/cpu:0'
    #with tf.device(device):
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
    #phase_train_tf = tf.Variable(phase_train)
    phase_train_tf = tf.constant(phase_train)
    mean, var = tf.cond( phase_train_tf,
                        mean_var_with_update,
                        lambda: (ema.average(batch_mean), ema.average(batch_var)))
    normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-3,name = name)
    return normed

def deconv_layer(x, W_shape, b_shape, name, padding='SAME'):
    '''
    [batch,width,height,channel]
    '''
    W = weight_variable(W_shape)
    b = bias_variable([b_shape])

    x_shape = tf.shape(x)
    out_shape = tf.pack([x_shape[0], x_shape[1], x_shape[2], W_shape[2]])

    return tf.nn.conv2d_transpose(x, W, out_shape, [1, 1, 1, 1], padding=padding) + b


def pool_layer(x, shrink_size = 2,name=None,padding = 'SAME'):
    '''
    see description of build method
    '''
    with tf.device('/gpu:0'):
        ret = tf.nn.max_pool(x, 
            ksize=[1, shrink_size, shrink_size, 1], 
            strides=[1, shrink_size, shrink_size, 1], 
            name = name,
            padding = padding)
        print 'layer: {}  size: {}'.format( name, ret.get_shape())
        return ret


def up_pool2x2_layer(mat_var,use_gpu = True,name=None,padding = 'SAME'):
    '''
    my unpool layer
    expand 
    NHWC
    '''
    if use_gpu:
        device = '/gpu:0'
    else:
        device = '/cpu:0'
    with tf.device(device):
        #### begin
        shape = mat_var.get_shape().as_list()
        batch = shape[0]
        height = shape[1]
        width = shape[2]
        channel = shape[3]

        # expand row
        mat_row_ext = tf.concat(3,[mat_var,mat_var])
        # expand colum
        mat_colum_ext = tf.concat(2,[mat_row_ext,mat_row_ext])
        # put back
        mat_back = tf.reshape(mat_colum_ext,[batch,2*height,2*width,channel],name = name)  
        print 'layer: {}  size: {}'.format( name, mat_back.get_shape())
        return mat_back

def up_pool_deconv_layer(x,up_mag,out_channel = None,use_gpu = True, name = None,padding = "SAME"):
    '''
    Use deconvolution to implement up-sampling
    shape: NHWC
    '''
    if use_gpu:
        device = '/gpu:0'
    else:
        device = '/cpu:0'
    with tf.device(device):
        #### begin
        shape = x.get_shape().as_list()
        batch = shape[0]
        height = shape[1]
        width = shape[2]
        in_channel = shape[3]
        if out_channel == None:
            out_channel = in_channel            

        W_shape = [up_mag,up_mag,out_channel,in_channel]
        W = weight_variable(W_shape)
        b = bias_variable([out_channel])

        out_shape = [batch,up_mag*height,up_mag*width,out_channel]
        stride = [1,up_mag,up_mag,1]
        deconv = tf.nn.conv2d_transpose(x,W,out_shape,stride,name = name,padding = padding)+b
        print 'layer: {}  size: {}'.format(name,deconv.get_shape())
        return deconv


def FixedUnPooling(x, shape, unpool_mat=None):
    """
    Reference: https://github.com/ppwwyyxx/tensorpack/blob/master/tensorpack/models/pool.py 
    
    Unpool the input with a fixed matrix to perform kronecker product with.
    Args:
        x (tf.Tensor): a NHWC tensor
        shape: int or (h, w) tuple
        unpool_mat: a tf.Tensor or np.ndarray 2D matrix with size=shape.
            If is None, will use a matrix with 1 at top-left corner.
    """
    def shape2d(shape):
        try:
            len(shape)
            return shape
        except:
            return shape,shape

    shape = shape2d(shape)

    # a faster implementation for this special case
    def UnPooling2x2ZeroFilled(x):
        # https://github.com/tensorflow/tensorflow/issues/2169
        out = tf.concat(3,[x, tf.zeros_like(x)])
        out = tf.concat(2,[out, tf.zeros_like(out)])

        sh = x.get_shape().as_list()
        if None not in sh[1:]:
            out_size = [-1, sh[1] * 2, sh[2] * 2, sh[3]]
            return tf.reshape(out, out_size)
        else:
            shv = tf.shape(x)
            ret = tf.reshape(out, tf.stack([-1, shv[1] * 2, shv[2] * 2, sh[3]]))
            ret.set_shape([None, None, None, sh[3]])
            return ret
    if shape[0] == 2 and shape[1] == 2 and unpool_mat is None:
        return UnPooling2x2ZeroFilled(x)

    input_shape = tf.shape(x)
    if unpool_mat is None:
        mat = np.zeros(shape, dtype='float32')
        mat[0][0] = 1
        unpool_mat = tf.constant(mat, name='unpool_mat')
    elif isinstance(unpool_mat, np.ndarray):
        unpool_mat = tf.constant(unpool_mat, name='unpool_mat')
    assert unpool_mat.get_shape().as_list() == list(shape)

    # perform a tensor-matrix kronecker product
    fx = symbf.flatten(tf.transpose(x, [0, 3, 1, 2]))
    fx = tf.expand_dims(fx, -1)       # (bchw)x1
    mat = tf.expand_dims(symbf.flatten(unpool_mat), 0)  # 1x(shxsw)
    prod = tf.matmul(fx, mat)  # (bchw) x(shxsw)
    prod = tf.reshape(prod, tf.stack(
        [-1, input_shape[3], input_shape[1], input_shape[2], shape[0], shape[1]]))
    prod = tf.transpose(prod, [0, 2, 4, 3, 5, 1])
    prod = tf.reshape(prod, tf.stack(
        [-1, input_shape[1] * shape[0], input_shape[2] * shape[1], input_shape[3]]))
    return prod




####### helper functions  #######
# show image
def show_image(*img_arrays):
    num = len(img_arrays)
    plt.figure(figsize=[20,5])
    for i in range(num):
        plt.subplot(num/4 + 1,4,i+1)
        data = np.squeeze(img_arrays[i])
        filt_min, filt_max = data.min(), data.max()
        try:
            plt.title("filter #{} output".format(i))
            plt.imshow(data, vmin=filt_min, vmax=filt_max,cmap="gray")
            plt.tight_layout()
            plt.axis('off')
        except:
            print "index {} cannot display".format(i)
            pass


def read_img_raw(im_name,size = 512,type=np.float32):
    im = Image.open(im_name,'r')
    im = im.convert('L')
    im = im.resize((size,size),Image.NEAREST)
    in_ = np.array(im, dtype=type)
    return in_

def read_image(img_name,size = 512):
    batch_xs = read_img_raw(img_name,size = size,type=np.float32)
    batch_xs = np.expand_dims(batch_xs,axis=0)
    batch_xs = np.expand_dims(batch_xs,axis=3)
    return batch_xs

def read_label(label_name,size = 512):
    batch_ys = read_img_raw(label_name,size = size,type=np.uint8)
    batch_ys = np.expand_dims(batch_ys,axis=0)
    return batch_ys