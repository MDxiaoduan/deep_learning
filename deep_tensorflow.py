import tensorflow as tf
import math


# 定义卷积，卷积后尺寸不变
def conv(img, weight, biases, offset, scale, strides=1):
    conv_conv = tf.nn.conv2d(img, weight, strides=[1, strides, strides, 1], padding='SAME') + biases
    mean, variance = tf.nn.moments(conv_conv, [0, 1, 2])
    conv_batch = tf.nn.batch_normalization(conv_conv, mean, variance, offset, scale, 1e-10)
    return tf.nn.sigmoid(conv_batch)


# 池化，大小k*k
def max_pool(x, k_size=(2, 2), stride=(2, 2)):
    return tf.nn.max_pool(x, ksize=[1, k_size[0], k_size[1], 1],
                          strides=[1, stride[0], stride[1], 1], padding='VALID')


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


def XNOR_convolutional(img, weight, batch_size, strides=1):
    w, h, in_channel, out_channel = weight.get_shape().as_list()
    _, w_in, h_in, img_channel = img.get_shape().as_list()
    pad_w = math.ceil((w_in/strides-1)*strides + w - w_in)   # math.ceil  向上取整
    pad_h = math.ceil((h_in/strides - 1) * strides + h - h_in)
    assert in_channel == img_channel
    conv_img = img
    batch_list = []
    for ii in range(batch_size):
        output_list = []
        for pp in range(out_channel):
            sum_img = []
            for qq in range(in_channel):
                each_img = []
                for jj in range(0, w_in, strides):
                    for kk in range(0, h_in, strides):
                        if w == 1:
                            padding = conv_img[ii, :, :, qq]
                        else:
                            print(ii, pp, qq, jj, kk, int(pad_w/2), int(pad_w-int(pad_w/2)))
                            padding = tf.pad(conv_img[ii, :, :, qq], paddings=[[int(pad_w/2), int(pad_w-int(pad_w/2))], [int(pad_h/2), int(pad_h-int(pad_h/2))]])
                        B = Bit_count(padding[jj:jj + w, kk:kk + h], weight[:, :, qq, pp])
                        each_img.append(B)
                sum_img.append(each_img)
            output_list.append(tf.reduce_sum(sum_img, 0))
        batch_list.append(output_list)
    con_out = tf.stack(batch_list)                          # [batch_size, out_size, weight, height]
    reshape_out = tf.reshape(con_out, [batch_size,  out_channel, w_in, h_in])
    out_img = tf.transpose(reshape_out, perm=[0, 2, 3, 1])   # 转置为标准维度顺序[batch_size, weight, height , out_channel]
    return out_img


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

