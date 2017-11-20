from DeepLearning.python import *
import matplotlib.pyplot as plt
import math


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


def plot_images(images, labels, show_color="gray"):
    """
    将一个batch图片输出
        输入images: [batch_size, weight, height](GRAY) or [batch_size, weight, height, 3](RGB)
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
        plt.title(labels[i], fontsize=14)
        plt.subplots_adjust(top=1.5)
        plt.imshow(images[i], cmap=show_color)
    plt.show()
