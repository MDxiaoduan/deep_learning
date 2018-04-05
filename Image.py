from DeepLearning.python import *
import matplotlib.pyplot as plt
import math
import cv2

def HIST(im, direction="weight"):
    """
    计算图像x或y方向的像素累加值并归一化到255
    """
    h = im.shape[0]
    w = im.shape[1]
    # print(h, w)
    if direction == "weight":    # 向图片宽的方向轴投影
        x = np.zeros((1, w))
        y = np.zeros((1, w))
        for i in range(w):
            x[0, i] = i
            y[0, i] = sum(im[:, i])  # y方向相加
        return x, Norm(y) * 255
    elif direction == "height":
        x = np.zeros((1, h))
        y = np.zeros((1, h))
        for i in range(h):
            y[0, i] = i
            x[0, i] = sum(im[i, :])  # x方向相加
        return Norm(x) * 255, y
    else:
        return print("dir is wrong")


def plot_images(images, labels=None, show_color="gray"):
    """
    将一个batch图片输出
        输入images: [batch_size, weight, height](GRAY) or [batch_size, weight, height, 3](RGB)
        输入labels：[batch_size, ...]
    Possible show_color values are: Accent, Accent_r, Blues, Blues_r, BrBG, BrBG_r, BuGn, BuGn_r, BuPu, BuPu_r, CMRmap,
    CMRmap_r, Dark2, Dark2_r, GnBu, GnBu_r, Greens, Greens_r, Greys, Greys_r, OrRd, OrRd_r, Oranges, Oranges_r, PRGn,
    PRGn_r, Paired, Paired_r, Pastel1, Pastel1_r, Pastel2, Pastel2_r, PiYG, PiYG_r, PuBu, PuBuGn, PuBuGn_r, PuBu_r,
    PuOr, PuOr_r, PuRd, PuRd_r, Purples, Purples_r, RdBu, RdBu_r, RdGy, RdGy_r, RdPu, RdPu_r, RdYlBu, RdYlBu_r, RdYlGn,
    RdYlGn_r, Reds, Reds_r, Set1, Set1_r, Set2, Set2_r, Set3, Set3_r, Spectral, Spectral_r, Wistia, Wistia_r, YlGn,
    YlGnBu, YlGnBu_r, YlGn_r, YlOrBr, YlOrBr_r, YlOrRd, YlOrRd_r, afmhot, afmhot_r, autumn, autumn_r, binary, binary_r,
    bone, bone_r, brg, brg_r, bwr, bwr_r, cool, cool_r, coolwarm, coolwarm_r, copper, copper_r, cubehelix, cubehelix_r,
    flag, flag_r, gist_earth, gist_earth_r, gist_gray, gist_gray_r, gist_heat, gist_heat_r, gist_ncar, gist_ncar_r,
    gist_rainbow, gist_rainbow_r, gist_stern, gist_stern_r, gist_yarg, gist_yarg_r, gnuplot, gnuplot2, gnuplot2_r,
    gnuplot_r, gray, gray_r, hot, hot_r, hsv, hsv_r, inferno, inferno_r, jet, jet_r, magma, magma_r, nipy_spectral,
    nipy_spectral_r, ocean, ocean_r, pink, pink_r, plasma, plasma_r, prism, prism_r, rainbow, rainbow_r, seismic,
    seismic_r, spectral, spectral_r, spring, spring_r, summer, summer_r, terrain, terrain_r, viridis, viridis_r, winter,
    winter_r
    """
    batch_size = images.shape[0]
    weight = math.sqrt(batch_size)
    if weight % int(weight) == 0:
        height = int(weight)
    else:
        height = int(weight) + 1
        weight = int(weight) + 1
    for i in np.arange(0, batch_size):
        plt.subplot(height, weight, i + 1)
        plt.axis('off')
        if labels != None:
            plt.title(labels[i], fontsize=14)
        plt.subplots_adjust(top=1.5)
        plt.imshow(images[i], cmap=show_color)
    plt.show()


# draw bbox
def draw_box(_im, bbox, color_box=(0, 0, 0), thick_bbox=2, thick_circle=8):
    bbox = int_fun(bbox)
    im_box = cv2.rectangle(_im, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color=color_box, thickness=thick_bbox)
    im_box = cv2.circle(im_box, (bbox[0], bbox[1]), thick_circle, (0, 255, 0), -1)
    im_box = cv2.circle(im_box, (bbox[0], bbox[3]), thick_circle, (0, 255, 0), -1)
    im_box = cv2.circle(im_box, (bbox[2], bbox[1]), thick_circle, (0, 255, 0), -1)
    im_box = cv2.circle(im_box, (bbox[2], bbox[3]), thick_circle, (0, 255, 0), -1)

    im_box = cv2.circle(im_box, (bbox[0], bbox[1]), thick_circle, (0, 0, 0), 1)
    im_box = cv2.circle(im_box, (bbox[0], bbox[3]), thick_circle, (0, 0, 0), 1)
    im_box = cv2.circle(im_box, (bbox[2], bbox[1]), thick_circle, (0, 0, 0), 1)
    im_box = cv2.circle(im_box, (bbox[2], bbox[3]), thick_circle, (0, 0, 0), 1)
    return im_box

# -----------------------------------------------------------------------------------------------
# 准确度曲线图
import matplotlib
#matplotlib.use('Agg')

plt.style.available
matplotlib.style.use('seaborn-darkgrid')


def plot_learning_curves(fig_path, n_epochs, data_list, name,  color_list, title, xlabel='iter', ylabel='Acc', style=''):
    """
    :param fig_path: 图片保存位置（包括文件名）
    :param n_epochs: 数据长度
    :param data_list: 数据列表，可以是多个数据，但是每个数据长度必须一样, [[数据集1], [数据集2]， ...]
    :param name: 每个数据的名称 也是一个列表
    :param color_list: 每个数据对应颜色列表,可以选：['dodgerblue', 'red', 'aqua', 'orange']
    :param title: 图形标题
    :param xlabel: x坐标名称
    :param ylabel：y坐标名称
    :param style: ...
    :return:
    """
    if color_list is None:
        color_list = ['dodgerblue', 'red', 'aqua', 'orange']
    measure = ylabel
    steps_measure = xlabel

    plt.figure(dpi=400)
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['axes.labelweight'] = 'bold'
    plt.rcParams['axes.titlesize'] = 12
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10
    plt.rcParams['legend.fontsize'] = 10
    plt.rcParams['figure.titlesize'] = 12

    steps = range(1, n_epochs + 1)
    plt.title(title + style)
    for ii in range(len(data_list)):
        assert ii < len(color_list)
        plt.plot(steps, data_list[ii], linewidth=1, color=color_list[ii], linestyle='-', marker='o',
                 markeredgecolor='black',
                 markeredgewidth=0.5, label=name[ii])

    eps = int((steps[-1]-1)/5)
    plt.xlabel(steps_measure)
    plt.xticks([0, eps, eps*2, eps*3, eps*4, eps*5], [0, eps, eps*2, eps*3, eps*4, eps*5])  # 前面一个数组表示真真实的值，后面一个表示在真实值处显示的值
    plt.ylabel(measure)
    plt.legend(loc='best', numpoints=1, fancybox=True)
    plt.savefig(fig_path)  # 这一句要在plt.show()之前
    plt.show()

# ----------------------------------------------------------------------------------------------------
def plot_without_axis(fig_path, n_epochs, data_list, name,  color_list, title, xlabel='iter', ylabel='Acc', style=''):
    """
    :param fig_path: 图片保存位置（包括文件名）
    :param n_epochs: 数据长度
    :param data_list: 数据列表，可以是多个数据，但是每个数据长度必须一样, [[数据集1], [数据集2]， ...]
    :param name: 每个数据的名称 也是一个列表
    :param color_list: 每个数据对应颜色列表,可以选：['dodgerblue', 'red', 'aqua', 'orange']
    :param title: 图形标题
    :param xlabel: x坐标名称
    :param ylabel：y坐标名称
    :param style: ...
    :return:
    """
    if color_list is None:
        color_list = ['dodgerblue', 'red', 'aqua', 'orange']
    measure = ylabel
    steps_measure = xlabel

    plt.figure(dpi=400)
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['axes.labelweight'] = 'bold'
    plt.rcParams['axes.titlesize'] = 12
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10
    plt.rcParams['legend.fontsize'] = 10
    plt.rcParams['figure.titlesize'] = 12

    steps = range(1, n_epochs + 1)
    # plt.title(title + style)
    for ii in range(len(data_list)):
        assert ii < len(color_list)
        plt.plot(steps, data_list[ii], linewidth=1, color=color_list[ii], linestyle='-', marker='o',
                 markeredgecolor='black',
                 markeredgewidth=0.5)

    eps = int((steps[-1]-1)/5)
    #plt.xlabel(steps_measure)
    #plt.xticks([0, eps, eps*2, eps*3, eps*4, eps*5], [0, eps, eps*2, eps*3, eps*4, eps*5])  # 前面一个数组表示真真实的值，后面一个表示在真实值处显示的值
    #plt.ylabel(measure)
    plt.xticks([])
    plt.yticks([])
    plt.axis('off')
    # frame = plt.gca()
    # frame.axes.get_yaxis().set_visible(False)
    # # x 轴不可见
    # frame.axes.get_xaxis().set_visible(False)
    plt.legend(loc='best', numpoints=1, fancybox=True)
    plt.savefig(fig_path)  # 这一句要在plt.show()之前
    plt.show()

# ----------------------------------------------------------------------------------------------------
def video_to_png(video_path, out_path, step):
    """
        :param video_path: 输入视频名称，如 G:\\video.mp4
        :param out_path: 输出PNG保存位置，如 G:\\PNG\\
        :param step: 图片以多大步长保存，int型
    """
    i = 0
    ret = True
    cap = cv2.VideoCapture(video_path)
    while ret:
        ret, frame = cap.read()
        if i % step == 0:
            cv2.imwrite(out_path + '{:05d}'.format(i//step) + '.png', frame)
        i += 1

    cap.release()