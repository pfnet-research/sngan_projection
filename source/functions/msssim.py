# -*- coding: utf-8 -*-
# Created by @ken-nakanishi
# MSSSIM implementation in Chainer, which follows the implementation in TF: https://github.com/tensorflow/models/blob/master/research/compression/image_encoder/msssim.py
from __future__ import division
from __future__ import print_function
import numpy as np
import chainer.functions as F
from chainer import cuda


def _gaussian_filter(size, sigma):
    x = np.linspace(-(size - 1) / 2, (size - 1) / 2, size)
    gauss = np.exp(-((x * x) / (2.0 * sigma * sigma)))
    gauss_2d = np.tile(gauss, (size, 1)) * np.tile(gauss.reshape(size, 1), (1, size))
    return gauss_2d / np.sum(gauss_2d)


class SSIM():
    def __init__(self, window_size=11, sigma=1.5, max_val=1, k1=0.01, k2=0.03):
        self.window_size = window_size
        self.sigma = sigma
        self.c1 = (k1 * max_val) ** 2
        self.c2 = (k2 * max_val) ** 2
        self.window = None

    def __call__(self, x0, x1, cs_map=False):
        xp = cuda.get_array_module(x0.data)

        x0 = F.mean(x0, axis=1, keepdims=True)
        x1 = F.mean(x1, axis=1, keepdims=True)

        if self.window is None:
            g_filter = _gaussian_filter(self.window_size, self.sigma)
            self.window = xp.asarray(g_filter.reshape(1, 1, self.window_size, self.window_size), dtype=np.float32)

        mu0 = F.convolution_2d(x0, self.window)
        mu1 = F.convolution_2d(x1, self.window)
        sigma00 = F.convolution_2d(x0 * x0, self.window)
        sigma11 = F.convolution_2d(x1 * x1, self.window)
        sigma01 = F.convolution_2d(x0 * x1, self.window)

        mu00 = mu0 * mu0
        mu11 = mu1 * mu1
        mu01 = mu0 * mu1
        sigma00 = sigma00 - mu00
        sigma11 = sigma11 - mu11
        sigma01 = sigma01 - mu01

        v1 = 2 * sigma01 + self.c2
        v2 = sigma00 + sigma11 + self.c2

        if cs_map:
            cs = v1 / v2
            cs = F.mean(cs, axis=3)
            cs = F.mean(cs, axis=2)
            cs = F.mean(cs, axis=1)
            return cs

        w1 = 2 * mu01 + self.c1
        w2 = mu00 + mu11 + self.c1

        ssim = (w1 * v1) / (w2 * v2)
        ssim = F.mean(ssim, axis=3)
        ssim = F.mean(ssim, axis=2)
        ssim = F.mean(ssim, axis=1)

        return ssim


class RGBSSIM(SSIM):
    def __init__(self, window_size=11, sigma=1.5, max_val=1, k1=0.01, k2=0.03):
        super(RGBSSIM, self).__init__(window_size, sigma, max_val, k1, k2)

    def __call__(self, x0, x1, cs_map=False):
        ssim0 = super(RGBSSIM, self).__call__(x0[:, :1, :, :], x1[:, :1, :, :])
        ssim1 = super(RGBSSIM, self).__call__(x0[:, 1:2, :, :], x1[:, 1:2, :, :])
        ssim2 = super(RGBSSIM, self).__call__(x0[:, 2:, :, :], x1[:, 2:, :, :])
        return (ssim0 + ssim1 + ssim2) / 3


class MSSSIM(SSIM):  # need 256x256 at least
    def __init__(self, window_size=11, sigma=1.5, max_val=1, k1=0.01, k2=0.03, n_levels=5):
        super(MSSSIM, self).__init__(window_size, sigma, max_val, k1, k2)
        self.weight = [0.0448, 0.2856, 0.3001, 0.2363, 0.1333]
        self.weight = self.weight[:n_levels]
        self.level = len(self.weight)

    def __call__(self, x0, x1, cs_map=False):
        xp = cuda.get_array_module(x0.data)

        h0 = x0
        h1 = x1
        msssim = 1

        for i in range(self.level - 1):
            cs = super(MSSSIM, self).__call__(h0, h1, cs_map=True)
            cs = F.maximum(cs, xp.zeros_like(cs.data))
            msssim *= cs ** self.weight[i]
            h0 = F.average_pooling_2d(h0, 2)
            h1 = F.average_pooling_2d(h1, 2)

        ssim = super(MSSSIM, self).__call__(h0, h1)
        ssim = F.maximum(ssim, xp.zeros_like(ssim.data))
        msssim *= ssim ** self.weight[-1]
        return msssim


class RGBMSSSIM(MSSSIM):  # need 256x256 at least
    def __init__(self, window_size=11, sigma=1.5, max_val=1, k1=0.01, k2=0.03, n_levels=5):
        super(RGBMSSSIM, self).__init__(window_size, sigma, max_val, k1, k2, n_levels)

    def __call__(self, x0, x1, cs_map=False):
        msssim0 = super(RGBMSSSIM, self).__call__(x0[:, :1, :, :], x1[:, :1, :, :])
        msssim1 = super(RGBMSSSIM, self).__call__(x0[:, 1:2, :, :], x1[:, 1:2, :, :])
        msssim2 = super(RGBMSSSIM, self).__call__(x0[:, 2:, :, :], x1[:, 2:, :, :])
        return (msssim0 + msssim1 + msssim2) / 3


def calc_ssim(type, im0, im1, scale=255):
    mdl = __import__('msssim', fromlist=[type])
    cls = getattr(mdl, type)
    f = cls()
    if im0.ndim == 3:
        im0 = im0.transpose(2, 0, 1)
        im1 = im1.transpose(2, 0, 1)
    elif im0.ndim == 2:
        im0 = im0[None, :, :]
        im1 = im1[None, :, :]
    else:
        print('invalid dim')
        exit(-1)
    im0 = im0.astype(np.float32) / scale
    im1 = im1.astype(np.float32) / scale
    im0 = im0[None, :, :, :]
    im1 = im1[None, :, :, :]
    ssim = f(im1, im0).data[0]
    return ssim


if __name__ == '__main__':
    import sys
    from PIL import Image

    args = sys.argv
    _im0 = np.array(Image.open(args[2]))
    _im1 = np.array(Image.open(args[3]))

    print('im0.shape: {}'.format(_im0.shape))
    print('im1.shape: {}'.format(_im1.shape))
    print('max im0: {}'.format(max(_im0.flatten())))
    print('max im1: {}'.format(max(_im1.flatten())))

    _ssim = calc_ssim(args[1], _im0, _im1)

    print('{}: {}'.format(args[1], _ssim))


    # import chainer
    # _, test = chainer.datasets.get_cifar10(withlabel=False, ndim=3)
    # test = test[:200]
    # test = np.tile(test, (1, 1, 8, 8))
    # x = test[0:10]
    # y = test[10:20]
    # f = SSIM()
    # print('SSIM: {} {}'.format(f(x, x), f(y, x)))
    # f = RGBSSIM()
    # print('RGBSSIM: {} {}'.format(f(x, x), f(y, x)))
    # f = MSSSIM()
    # print('MSSSIM: {} {}'.format(f(x, x), f(y, x)))
    # f = RGBMSSSIM()
    # print('RGBMSSSIM: {} {}'.format(f(x, x), f(y, x)))

    # import matplotlib.pyplot as plt
    # plt.style.use('seaborn-poster')
    # output = _gaussian_filter(3, 0.5)
    # print(output)
    # plt.imshow(output)
    # plt.show()
