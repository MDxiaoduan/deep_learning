import numpy as np
import random
from math import *

#  归一化
def Norm(vec):  # 默认变量值None!!
    if np.amax(vec) == np.min(vec):
        if np.amax(vec) == 0:
            return vec
        else:
            return vec / np.amax(vec)
    else:
        return (vec - np.min(vec)) / (np.amax(vec) - np.min(vec))


# 按行打乱矩阵
def shuffle_matrix(A, order=None):   # A is numpy matrix
    row, col = A.shape
    B = np.zeros((row, col))
    if order is None:
        li = [i for i in range(row)]
        random.shuffle(li)
        for kk in range(row):
            B[kk, :] = A[li[kk], :]
        order = li
    else:
        for kk in range(row):
            B[kk, :] = A[order[kk], :]
    return B, order


def list_save(content, filename, mode='a'):
    # Try to save a list  into txt file.
    # content:list  filename:path and txt name to save as txt such as : "list.txt"
    file = open(filename, mode)   # filename could existence or not
    for i in range(len(content)):
        file.write(str(content[i]))
    file.close()


def Bin_img(x):
    if x < 0:
        return -1
    else:
        return 1


def XNOR(A, B):  # 同或：相同为1 不同为0
    if A == B:
        return 1
    else:
        return 0


def Gaussian(x, mu=0, sigma2=1):    # sigma2:方差
    return 1./sqrt(2.*pi*sigma2)*exp(-.5*(x-mu)**2/sigma2)
