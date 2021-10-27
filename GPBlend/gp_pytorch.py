import math
import time
import torch

import numpy as np
# from scipy.fftpack import dct, idct
from GPBlend.gp_gan import laplacian_param, gaussian_param
from t7_dct import dct, idct
from scipy.ndimage import correlate
from skimage.transform import resize

# ################## Gradient Operator #########################
# normal_h = lambda im: correlate(im, np.asarray([[0, -1, 1]]), mode='nearest')
# normal_v = lambda im: correlate(im, np.asarray([[0, -1, 1]]).T, mode='nearest')
#
# gradient_operator = {
#     'normal': (normal_h, normal_v),
#     'sobel': (sobel_h, sobel_v),
#     'scharr': (scharr_h, scharr_v),
#     'roberts': (roberts_pos_diag, roberts_neg_diag),
#     'prewitt': (prewitt_h, prewitt_v)
# }


###########################################################
from GPBlend.module import normal_h, normal_w, GaussianSmoothing


def preprocess(im):
    im = np.transpose(im * 2 - 1, (2, 0, 1)).astype(np.float32)
    return im


def ndarray_resize(im, image_size, order=3, dtype=None):
    im = resize(im, image_size, preserve_range=True, order=order, mode='constant')

    if dtype:
        im = im.astype(dtype)
    return im


def gradient_feature(im, color_feature):
    result = torch.zeros((*im.shape, 5)).cuda()
    normal_conv_h = normal_h().cuda()
    normal_conv_w = normal_w().cuda()

    result[:, :, :, :, 0] = color_feature
    result[:, :, :, :, 1] = normal_conv_h(im)
    result[:, :, :, :, 2] = normal_conv_w(im)
    result[:, :, :, :, 3] = torch.roll(result[:, :, :, :, 1], shifts=1, dims=3)
    result[:, :, :, :, 4] = torch.roll(result[:, :, :, :, 2], shifts=1, dims=2)

    return result


def fft2(K, size, dtype):
    w, h = size
    # param = torch.fft(K,signal_ndim=1)      #torch 1.4 错，这是复数到复数
    param = torch.rfft(K, signal_ndim=1)  # torch 1.4
    # param = np.fft.fft2(K)
    # param = torch.fft.fft2(K)    #torch 1.1  1.9
    param = torch.real(param[0:w, 0:h])

    return param


def laplacian_param_torch(size, dtype):
    w, h = size
    K = torch.zeros((2 * w, 2 * h)).to(dtype)

    laplacian_k = torch.tensor([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])
    kw, kh = laplacian_k.shape
    K[:kw, :kh] = laplacian_k

    K = torch.roll(K, -(kw // 2), 0)
    K = torch.roll(K, -(kh // 2), 1)

    return fft2(K, size, dtype)


def gaussian_param_torch(size, dtype, sigma):
    w, h = size
    K = torch.zeros((2 * w, 2 * h))

    K[1, 1] = 1
    g = GaussianSmoothing(channels=1, kernel_size=3, sigma=sigma)
    K[:3, :3] = g(K[:3, :3].unsqueeze(dim=0).unsqueeze(dim=0))[0][0]

    K = torch.roll(K, -1, 0)
    K = torch.roll(K, -1, 1)

    return fft2(K, size, dtype)


def dct2(x, norm='ortho'):
    return dct(dct(x, norm=norm).T, norm=norm).T


def idct2(x, norm='ortho'):
    return idct(idct(x, norm=norm).T, norm=norm).T


def gaussian_poisson_editing(X, param_l, param_g, color_weight=1, eps=1e-12):
    Fh = (X[:, :, :, :, 1] + torch.roll(X[:, :, :, :, 3], -1, 3)) / 2
    Fv = (X[:, :, :, :, 2] + torch.roll(X[:, :, :, :, 4], -1, 2)) / 2
    L = torch.roll(Fh, 1, 3) + torch.roll(Fv, 1, 2) - Fh - Fv

    param = param_l + color_weight * param_g
    param[(param >= 0) & (param < eps)] = eps
    param[(param < 0) & (param > -eps)] = -eps

    Y = torch.zeros(X.shape[1:4])

    for i in range(3):
        Xdct = dct2(X[0, i, :, :, 0])  # 原图每个通道
        Ydct = (dct2(L[0, i, :, :]) + color_weight * Xdct) / param
        Y[i, :, :] = idct2(Ydct)
    return Y


def run_GP_editing(src_im, dst_im, mask_im, bg_for_color, color_weight, sigma, gradient_kernel='normal'):
    T_min = time.time()

    dst_feature = gradient_feature(dst_im, bg_for_color)
    src_feature = gradient_feature(src_im, bg_for_color)  # 两个 gradient_feature 耗时1s
    mask_im = mask_im.unsqueeze(dim=-1).float()
    feature = dst_feature * (1 - mask_im) + src_feature * mask_im
    print('T_min', time.time() - T_min)
    size = feature.shape[-3:-1]
    dtype=float
    param_l = laplacian_param(size, dtype)  # 拉普拉斯的傅里叶变换
    param_g = gaussian_param(size, dtype, sigma)
    param_l = torch.from_numpy(param_l).cuda()
    param_g = torch.from_numpy(param_g).cuda()

    gan_im = gaussian_poisson_editing(feature, param_l, param_g, color_weight=color_weight)

    gan_im = np.clip(gan_im, 0, 1)

    return gan_im


def laplacian_pyramid(im, max_level, image_size, smooth_sigma):
    im_pyramid = [im]
    diff_pyramid = []
    for i in range(max_level - 1, -1, -1):
        smoothed = gaussian(im_pyramid[-1], smooth_sigma, multichannel=True)
        diff_pyramid.append(im_pyramid[-1] - smoothed)
        smoothed = ndarray_resize(smoothed, (image_size * 2 ** i, image_size * 2 ** i))
        im_pyramid.append(smoothed)

    im_pyramid.reverse()
    diff_pyramid.reverse()

    return im_pyramid, diff_pyramid


@torch.no_grad()
def GP_GPU_fusion(obj, bg, mask, gpu=0, color_weight=1, sigma=0.5, gradient_kernel='normal', smooth_sigma=1,
                  supervised=True, nz=100, n_iteration=1000):
    device = f'cuda:{gpu}'
    w_orig, h_orig, _ = obj.shape
    obj = torch.from_numpy(obj)[np.newaxis].to(device).permute(0, 3, 1, 2)
    bg = torch.from_numpy(bg)[np.newaxis].to(device).permute(0, 3, 1, 2)
    mask = torch.from_numpy(mask)[np.newaxis][np.newaxis].to(device)
    ############################ Gaussian-Poisson GAN Image Editing ###########################
    # pyramid
    # gauss = GaussianSmoothing(channels=3, kernel_size=3, sigma=smooth_sigma, dim=2)
    # Start pyramid
    gan_im = bg
    T1 = time.time()
    gan_im = run_GP_editing(obj, bg, mask, gan_im, color_weight, sigma,
                            gradient_kernel)
    print('TIME T1', time.time() - T1)
    gan_im=gan_im.permute(1,2,0).numpy()[::-1]
    gan_im = np.clip(gan_im * 255, 0, 255).astype(np.uint8)

    return gan_im
