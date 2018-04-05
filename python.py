import numpy as np
import random
import os

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
    if os.path.exists(filename):   # filename could existence or not but i will remove it haha
        os.remove(filename)
    file = open(filename, mode)
    for i in range(len(content)):
        file.write(str(content[i]) + '\n')
    file.close()


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


def find_same(vec):         # 找到向量中两个相同的元素的位置
    for index1, m in enumerate(vec):
        for index2, n in enumerate(vec):
            if m == n and index1 != index2:
                return index1, index2
            else:
                continue
    return None


def find_near(x, vec):   # 找到vec中和x最近的数
    diff = []
    for tx in vec:
        diff.append(abs(tx - x))
    return vec[np.argmin(diff)]

# ------------------------------------------------------------------------------------------------
from moviepy.editor import VideoClip, ImageSequenceClip
from moviepy.video.io.bindings import mplfig_to_npimage
from moviepy.editor import VideoFileClip
from moviepy.video.VideoClip import ImageClip


def gen_gif(imgs_path, save_path, fps):
    '''  把一串连续的图片保存为gif动图
    :param imgs_path: 图片路径  ,比如"G:\\UCLA\\png"
    :param save_path: 保存的gif路径（包括文件名.gif） ,比如"G:\\UCLA\\ucla.gif"
    :param fps: 以多少副图片保存一帧
    :return: 一个gif图片，在save_path文件夹下
    '''

    image_list = []
    for im in os.listdir(imgs_path):
        image_list.append(os.path.join(imgs_path, im))

    clip = ImageSequenceClip(image_list, fps=fps)
    clip.write_gif(save_path, fps=fps)
# ------------------------------------------------------------------------------------------------
# 把一个列表元素全变为int类型
int_fun = lambda x_list: [int(x) for x in x_list]
