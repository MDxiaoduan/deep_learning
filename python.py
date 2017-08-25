import numpy as np
import random


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
    x, y = A.shape
    B = np.zeros((x, y))
    if order is None:
        li = [i for i in range(x)]
        random.shuffle(li)
        for kk in range(x):
            B[kk, :] = A[li[kk], :]
        order = li
    else:
        for kk in range(x):
            B[kk, :] = A[order[kk], :]
    return B, order


def list_save(content, filename, mode='a'):
    # Try to save a list  into txt file.
    # content:list  filename:path and txt name to save as txt such as : "list.txt"
    file = open(filename, mode)   # filename could existence or not
    for i in range(len(content)):
        file.write(str(content[i]))
    file.close()
