import numpy as np
import random
import os

from math import *

class python_numpy:
    #  归一化
    @staticmethod
    def Norm(vec):  # 默认变量值None!!
        if np.amax(vec) == np.min(vec):
            if np.amax(vec) == 0:
                return vec
            else:
                return vec / np.amax(vec)
        else:
            return (vec - np.min(vec)) / (np.amax(vec) - np.min(vec))


    # 按行打乱矩阵
    @staticmethod
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

    @staticmethod
    def list_save(content, filename, mode='a'):
        # Try to save a list  into txt file.
        # content:list  filename:path and txt name to save as txt such as : "list.txt"
        if os.path.exists(filename):   # filename could existence or not but i will remove it haha
            os.remove(filename)
        file = open(filename, mode)
        for i in range(len(content)):
            file.write(str(content[i]) + '\n')
        file.close()

    @staticmethod
    def text_read(filename):
        # Try to read a txt file and return a list.Return [] if there was a mistake.
        try:
            file = open(filename, 'r')
        except IOError:
            error = []
            return error
        content = file.readlines()

        for i in range(len(content)):
            content[i] = content[i][:len(content[i])-1]

        file.close()
        return content

    @staticmethod
    def Bin_img(x):
        if x < 0:
            return -1
        else:
            return 1

    @staticmethod
    def XNOR(A, B):  # 同或：相同为1 不同为0
        if A == B:
            return 1
        else:
            return 0

    @staticmethod
    def Gaussian(x, mu=0, sigma2=1):    # sigma2:方差
        return 1./sqrt(2.*pi*sigma2)*exp(-.5*(x-mu)**2/sigma2)

    @staticmethod
    def find_same(vec):         # 找到向量中两个相同的元素的位置
        for index1, m in enumerate(vec):
            for index2, n in enumerate(vec):
                if m == n and index1 != index2:
                    return index1, index2
                else:
                    continue
        return None

    @staticmethod
    def find_near(x, vec):   # 找到vec中和x最近的数
        diff = []
        for tx in vec:
            diff.append(abs(tx - x))
        return vec[np.argmin(diff)]

    # ------------------------------------------------------------------------------------------------
    @staticmethod
    def mean_filter(arr, step):
        """
        平滑滤波函数，输入是一个列表，输出是这个列表平滑之后的值。即取step个数的平均值
        :param arr:列表
        :param step:以多大步长取平均
        :return:平滑后的列表
        """
        new_arr = arr[:int(step / 2)]
        for kk in range(int(step / 2), len(arr) - step + int(step / 2)):
            new_arr.append(int(sum(arr[kk - int(step / 2):kk + step - int(step / 2)]) / step))
        new_arr.extend(arr[-(step - int(step / 2)):])
        return new_arr

    def HIST(self, im, direction="weight"):
        """
        计算图像x或y方向的像素累加值并归一化到255
        """
        h = im.shape[0]
        w = im.shape[1]
        # print(h, w)
        if direction == "weight":  # 向图片宽的方向轴投影
            x = np.zeros((1, w))
            y = np.zeros((1, w))
            for i in range(w):
                x[0, i] = i
                y[0, i] = sum(im[:, i])  # y方向相加
            return x, self.Norm(y) * 255
        elif direction == "height":
            x = np.zeros((1, h))
            y = np.zeros((1, h))
            for i in range(h):
                y[0, i] = i
                x[0, i] = sum(im[i, :])  # x方向相加
            return self.Norm(x) * 255, y
        else:
            return print("dir is wrong")

# 把一个列表元素全变为int类型
int_fun = lambda x_list: [int(x) for x in x_list]


