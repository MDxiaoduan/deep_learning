import tensorflow as tf
import math


def Batch_Normalization(in_img, dimension):
    offset = tf.get_variable('offset', [dimension], initializer=tf.constant_initializer(0.0, tf.float32))
    scale = tf.get_variable('scale', [dimension], initializer=tf.constant_initializer(1.0, tf.float32))
    mean, variance = tf.nn.moments(in_img, [0, 1, 2])
    img = tf.nn.batch_normalization(in_img, mean, variance, offset, scale, 0.001)
    return img


# 定义卷积，卷积后尺寸不变
def BN_ReLU_Conv(in_img, weight, biases, dimension, strides=1, active="relu"):
    img = Batch_Normalization(in_img, dimension)
    if active == "sigmoid":
        active_img = tf.nn.sigmoid(img)
    else:
        active_img = tf.nn.relu(img)
    return tf.nn.conv2d(active_img, weight, strides=[1, strides, strides, 1], padding='SAME') + biases


# 定义卷积，卷积后尺寸不变
def Conv_BN_Relu(in_img, weight, dimension, biases=None, strides=1, active="relu", padding='SAME'):
    if biases is None:
        img = tf.nn.conv2d(in_img, weight, strides=[1, strides, strides, 1], padding=padding)
    else:
        img = tf.nn.conv2d(in_img, weight, strides=[1, strides, strides, 1], padding=padding) + biases
    img = Batch_Normalization(img, dimension)
    if active == "sigmoid":
        return tf.nn.sigmoid(img)
    else:
        return tf.nn.relu(img)


# 池化，大小k*k
def max_pool(x, k_size=(2, 2), stride=(2, 2), pad = 'VALID'):
    return tf.nn.max_pool(x, ksize=[1, k_size[0], k_size[1], 1],
                          strides=[1, stride[0], stride[1], 1], padding=pad)


def ave_pool(x, k_size=(2, 2), stride=(2, 2), pad = 'VALID'):
    return tf.nn.avg_pool(x, ksize=[1, k_size[0], k_size[1], 1],
                          strides=[1, stride[0], stride[1], 1], padding=pad)


def Bin_conv(img, weight, strides=1):
    w, h, in_size, out_size = weight.get_shape().as_list()
    B = tf.sign(weight)
    alpha = tf.div(tf.reduce_sum(abs(weight), [0, 1]), w * h)
    B_alpha = tf.multiply(B, alpha)          # 刚好对应位置相乘
    conv_img = tf.nn.conv2d(img, B_alpha, strides=[1, strides, strides, 1], padding='SAME')
    return conv_img


def hard_tanh(img):
    active_img = tf.maximum(tf.minimum(img, 1), -1)
    return active_img


# Bit_count operation:统计二进制表达式中”1“的个数  bit运算的kernel
def Bit_count(A, B):
    # 判断多维数组对应元素是否相等  tf.cast 保证A B精度相等  XNOR运算
    C = tf.equal(tf.cast(A, tf.int8), tf.cast(B, tf.int8))
    count = tf.reduce_sum(tf.to_float(C))
    return count


def XNOR_convolutional(img, weight, strides=1):
    w, h, in_channel, out_channel = weight.get_shape().as_list()
    _, w_in, h_in, img_channel = img.get_shape().as_list()
    pad_w = math.ceil((w_in/strides-1)*strides + w - w_in)   # math.ceil  向上取整
    pad_h = math.ceil((h_in/strides - 1) * strides + h - h_in)
    assert in_channel == img_channel
    conv_img = img
    batch_list = []
    conv_out = tf.nn.conv2d(img, weight, strides=[1, strides, strides, 1], padding='SAME')
    conv_conv = tf.div(tf.add(conv_out, tf.ones_like(conv_out) * (w**2)), 2)
    # for ii in range(batch_size):
    #     output_list = []
    #     for pp in range(out_channel):
    #         sum_img = []
    #         for qq in range(in_channel):
    #             each_img = []
    #             for jj in range(0, w_in, strides):
    #                 for kk in range(0, h_in, strides):
    #                     print(pp, qq, jj, kk)
    #                     if w == 1:
    #                         padding = conv_img[ii, :, :, qq]
    #                     else:
    #                         # print(ii, pp, qq, jj, kk, int(pad_w/2), int(pad_w-int(pad_w/2)))
    #                         padding = tf.pad(conv_img[ii, :, :, qq], paddings=[[int(pad_w/2), int(pad_w-int(pad_w/2))], [int(pad_h/2), int(pad_h-int(pad_h/2))]])
    #                     B = Bit_count(padding[jj:jj + w, kk:kk + h], weight[:, :, qq, pp])
    #                     each_img.append(B)
    #             sum_img.append(each_img)
    #         output_list.append(tf.reduce_sum(sum_img, 0))
    #     batch_list.append(output_list)
    # con_out = tf.stack(batch_list)                          # [batch_size, out_size, weight, height]
    # reshape_out = tf.reshape(con_out, [batch_size,  out_channel, w_in, h_in])
    # out_img = tf.transpose(reshape_out, perm=[0, 2, 3, 1])   # 转置为标准维度顺序[batch_size, weight, height , out_channel]
    return conv_conv


def XNOR_Active(img, weight, strides=1):   # img [batch_size, w, h, channel]
    batch_size, w_in, h_in, in_channel = img.get_shape().as_list()
    w, h, _, _ = weight.get_shape().as_list()
    # c = tf.constant(in_channel)
    A = tf.reduce_sum(abs(img), [3])
    A = tf.cast(tf.div(A, in_channel), tf.float32)
    k_weight = tf.ones([w, h])*(1/(w*h))  # tf.constant(1/(w*h), )
    K = tf.nn.conv2d(tf.expand_dims(A, 3), tf.expand_dims(tf.expand_dims(k_weight, 2), 3), strides=[1, strides, strides, 1], padding="SAME")
    I = tf.sign(img)
    return I, K  # [batch_size, w_in, h_in, in_channel], [batch_size, w_in, h_in, 1]

