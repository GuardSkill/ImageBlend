import math
import time
import torch

import numpy as np
# from scipy.fftpack import dct, idct
from gp_gan import laplacian_param, gaussian_param
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
from module import normal_h, normal_w, GaussianSmoothing


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


def fft2(K, size):
    w, h = size
    # param = torch.fft(K,signal_ndim=1)      #torch 1.4 错，这是复数到复数
    # param = torch.rfft(K, signal_ndim=1,onesided=False)  # torch 1.4
    # param = np.fft.fft2(K)
    param = torch.fft.fft2(K)  # torch 1.1  1.9
    param = torch.real(param[0:w, 0:h])

    return param


def laplacian_param_torch(size, device='cuda'):
    w, h = size
    K = torch.zeros((2 * w, 2 * h)).to(device)

    laplacian_k = torch.tensor([[0, -1, 0], [-1, 4, -1], [0, -1, 0]]).cuda()
    kw, kh = laplacian_k.shape
    K[:kw, :kh] = laplacian_k

    K = torch.roll(K, -(kw // 2), 0)
    K = torch.roll(K, -(kh // 2), 1)

    return fft2(K, size)


def gaussian_param_torch(size, sigma, device='cuda'):
    w, h = size
    K = torch.zeros((2 * w, 2 * h)).to(device)

    # K[1, 1] = 1
    # g = GaussianSmoothing(channels=1, kernel_size=3, sigma=sigma).cuda()
    # K[:3, :3] = g(K[:3, :3].unsqueeze(dim=0).unsqueeze(dim=0))[0][0]
    # K = torch.zeros((2 * w, 2 * h)).cuda()
    # K[1, 1] = 1
    # from torchvision import transforms
    # T_guassian=transforms.GaussianBlur(kernel_size=(3,3), sigma=(sigma,sigma))
    # K[:3, :3] = T_guassian(K[:3, :3].unsqueeze(dim=0).unsqueeze(dim=0))[0][0]
    K[:3, :3] = torch.tensor([[0.01133, 0.08373, 0.01133],
                              [0.08373, 0.61869, 0.08373],
                              [0.01133, 0.08373, 0.01133]])
    K = torch.roll(K, -1, 0)
    K = torch.roll(K, -1, 1)
    return fft2(K, size)


def dct2(x, norm='ortho'):
    return dct(dct(x, norm=norm).T, norm=norm).T


def idct2(x, norm='ortho'):
    return idct(idct(x, norm=norm).T, norm=norm).T


def gaussian_poisson_editing(X, param, color_weight=1):
    Fh = (X[:, :, :, :, 1] + torch.roll(X[:, :, :, :, 3], -1, 3)) / 2
    Fv = (X[:, :, :, :, 2] + torch.roll(X[:, :, :, :, 4], -1, 2)) / 2
    L = torch.roll(Fh, 1, 3) + torch.roll(Fv, 1, 2) - Fh - Fv
    Y = torch.zeros(X.shape[1:4])
    for i in range(3):
        Xdct = dct2(X[0, i, :, :, 0])  # 原图每个通道
        Ydct = (dct2(L[0, i, :, :]) + color_weight * Xdct) / param
        Y[i, :, :] = idct2(Ydct)
    return Y


class Gradient_Caculater(torch.nn.Module):
    def __init__(self, img_shape):
        super(Gradient_Caculater, self).__init__()
        self.h_conv = normal_h()
        self.w_conv = normal_w()
        self.feature = torch.nn.Parameter(torch.zeros((*img_shape, 5)))

    def forward(self, im, color_feature):
        self.feature[:, :, :, :, 0] = color_feature
        self.feature[:, :, :, :, 1] = self.h_conv(im)
        self.feature[:, :, :, :, 2] = self.w_conv(im)
        self.feature[:, :, :, :, 3] = torch.roll(self.feature[:, :, :, :, 1], shifts=1, dims=3)
        self.feature[:, :, :, :, 4] = torch.roll(self.feature[:, :, :, :, 2], shifts=1, dims=2)
        return self.feature


class GP_model(torch.nn.Module):
    def __init__(self, img_shape, color_weight=8e-10, sigma=0.5, eps=1e-12):
        # img_shape :(B, C, H, W)
        super(GP_model, self).__init__()
        self.gradient_caculater_Dst = Gradient_Caculater(img_shape)
        self.gradient_caculater_Src = Gradient_Caculater(img_shape)
        size = img_shape[-2:]
        param_l = laplacian_param_torch(size)
        param_g = gaussian_param_torch(size, sigma)
        self.param_total = param_l + color_weight * param_g
        self.param_total[(self.param_total >= 0) & (self.param_total < eps)] = eps
        self.param_total[(self.param_total < 0) & (self.param_total > -eps)] = -eps
        self.Y = torch.zeros(img_shape[1:])
        # self.Y = torch.nn.Parameter(torch.zeros(img_shape[1:]))


    @torch.no_grad()
    def forward(self, src_im, dst_im, mask_im, bg_for_color, color_weight, sigma):
        dst_feature = self.gradient_caculater_Dst(dst_im, color_feature=bg_for_color)
        src_feature = self.gradient_caculater_Src(src_im, color_feature=bg_for_color)
        # (B, C, H, W,5)
        mask_im = mask_im.unsqueeze(dim=-1).float()
        X = dst_feature * (1 - mask_im) + src_feature * mask_im
        # fusion
        Fh = (X[:, :, :, :, 1] + torch.roll(X[:, :, :, :, 3], -1, 3)) / 2
        Fv = (X[:, :, :, :, 2] + torch.roll(X[:, :, :, :, 4], -1, 2)) / 2
        L = torch.roll(Fh, 1, 3) + torch.roll(Fv, 1, 2) - Fh - Fv
        for i in range(3):
            Xdct = dct2(X[0, i, :, :, 0])  # 原图每个通道
            Ydct = (dct2(L[0, i, :, :]) + color_weight * Xdct) / self.param_total
            self.Y[i, :, :] = idct2(Ydct)
        return self.Y


@torch.no_grad()
def GP_GPU_Model_fusion(obj, bg, mask, gpu=0, color_weight=8e-10, sigma=0.5, gradient_kernel='normal', smooth_sigma=1,
                        supervised=True, nz=100, n_iteration=1000):
    T0 = time.time()
    device = f'cuda:{gpu}'
    w_orig, h_orig, _ = obj.shape
    obj = torch.from_numpy(obj[np.newaxis]).to(device).permute(0, 3, 1, 2)
    bg = torch.from_numpy(bg[np.newaxis]).to(device).permute(0, 3, 1, 2)
    mask = torch.from_numpy(mask[np.newaxis][np.newaxis]).to(device)
    print('CPU TO GPU', time.time() - T0)
    ############################ Gaussian-Poisson GAN Image Editing ###########################
    # pyramid
    # gauss = GaussianSmoothing(channels=3, kernel_size=3, sigma=smooth_sigma, dim=2)
    # Start pyramid
    T1 = time.time()
    infer_model = GP_model(img_shape=obj.shape, color_weight=color_weight, sigma=sigma).to(device)
    print('Init TIME T1', time.time() - T1)

    gan_im = infer_model(obj, bg, mask, bg, color_weight, sigma)
    gan_im = gan_im.permute(1, 2, 0).cpu().numpy()
    gan_im = np.clip(gan_im * 255, 0, 255).astype(np.uint8)
    print('Init + infer TIME', time.time() - T1)

    return gan_im
