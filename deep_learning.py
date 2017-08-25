from __future__ import division        # for 1/2=0.5
import numpy as np


def Batch_Normalization(batch_data, gama=1, beta=0):
    mean = np.mean(batch_data)
    std = np.std(batch_data)
    batch_norm_data = (batch_data - mean) / std
    return gama*batch_norm_data + beta


def convolve(image, kernel):       # image, kernel 都是二维矩阵(不一定是正方形)  stride = 1
    (iH, iW) = image.shape[:2]
    (kH, kW) = kernel.shape[:2]
    output = np.zeros((iH-kH+1, iW-kW+1), dtype="float32")
    for y in np.arange(0, iH-kH+1):
        for x in np.arange(0, iW-kW+1):
            roi = image[y:y+kH, x:x+kW]
            k = (np.array(roi)*np.array(kernel)).sum()
            output[y, x] = k
    return output


def sigmoid(x):
    return 1./(1+np.exp(-x))


def ReLu(x):
    if x < 0:
        return 0
    else:
        return x


def sigmoid_derivative(x):     # sigmoid(x)的导数
    return (1./(1+np.exp(-x)))*(1-(1./(1+np.exp(-x))))


def ReLu_derivative(x):        # ReLu(x)的导数
    if x < 0:
        return 0
    else:
        return 1


def pooling(image, p_size, stride):                      # 最好是简单的可以整除
    (iH, iW) = image.shape[:2]
    (kH, kW) = p_size
    out = np.zeros((int(iH/stride), int(iW/stride)))     # 输入输出的大小关系
    for ii in range(0, iH, stride):
        for jj in range(0, iW, stride):                 # 相当于和np.ones((2,2))做卷积
            out[int(ii/stride), int(jj/stride)] = image[ii:(ii+kH), jj:(jj+kW)].sum()
    return out/4


def expand(inputs, stride):  # 二维矩阵
    (w, h) = inputs.shape
    out = np.zeros((w*stride, h*stride))
    for ii in range(0, w*stride, stride):
        for jj in range(0, h*stride, stride):
            out[ii:(ii + stride), jj:(jj+stride)] = inputs[int(ii/2), int(jj/2)]*np.ones((stride, stride))
    return out


def deconvolution(image, weight):    # 残差和卷积核得到上一层残差   in: [8, 8] [5, 5] out :[12, 12]
    (iH, iW) = image.shape[:2]
    (kH, kW) = weight.shape[:2]
    image_exp = np.zeros((iH+2*(kH-1), iW+2*(kW-1)))
    image_exp[4:12, 4:12] = image
    out = convolve(image_exp, weight)
    return out


def learning_rate(kk, name):
    if name == "exp":       # 这里必须用==不能用is
        return 0.0001 + (1 - 0.0001) * np.exp(-kk / 100)
    elif name == "my":
        if kk < 500:
            return 1
        elif 500 < kk < 2000:
            return 0.1
        elif 2000 < kk < 5000:
            return 0.01
        elif 5000 < kk < 10000:
            return 0.001
        else:
            return 0.0001
    elif name == "0.01":
        return 0.01
    elif name == "0.001":
        return 0.001
    elif name == "0.0001":
        return 0.0001


def one_hot(batch_label, class_num):            # label 是batch  class_num 是分类数  输出one_hot的label
    length = batch_label.shape[0]
    Class_list = [kk for kk in range(class_num)]
    out_label = np.zeros((length, class_num))
    for kk in range(length):
        out_label[kk, Class_list.index(int(batch_label[kk]))] = 1.0
    return out_label


def Sign(x):
    if x < 0:
        return -1
    else:
        return 1


def Sign_derivative(x):
    if abs(x) > 1:
        return 0
    else:
        return 1


def hard_sigmoid(x):
    return max(0, min(1, (x + 1)/2))


def hard_sigmoid_derivative(x):
    if abs(x) <= 1:
        return 1./2
    else:
        return 0


def hard_tanh(x):
    return max(-1, min(1, x))   # 2*hard_sigmoid(x) - 1


def hard_tanh_derivative(x):
    if abs(x) <= 1:
        return 1
    else:
        return 0


def bin_function(x):   # 输入x可以是一维、二维、三维
    x = np.array(x)
    shape = len(x.shape)
    assert shape < 4
    output = np.ones_like(x)
    dim_1 = x.shape[0]
    if shape == 1:
        for ii in range(dim_1):
            if hard_sigmoid(x[ii]) < 0.5:
                output[ii] = -1
    if shape == 2:
        dim_2 = x.shape[1]
        for ii in range(dim_1):
            for jj in range(dim_2):
                if hard_sigmoid(x[ii, jj]) < 0.5:
                    output[ii, jj] = -1
    elif shape == 3:
        dim_2 = x.shape[1]
        dim_3 = x.shape[2]
        for ii in range(dim_1):
            for jj in range(dim_2):
                for kk in range(dim_3):
                    if hard_sigmoid(x[ii, jj, kk]) < 0.5:
                        output[ii, jj, kk] = -1
    return output
