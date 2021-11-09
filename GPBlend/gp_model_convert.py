import math
import time
import torch
from torch2trt import torch2trt
import numpy as np
# from scipy.fftpack import dct, idct
from GPBlend.dct_module import dctmodule2D, idctmodule2D
from gp_gan import laplacian_param, gaussian_param
from t7_dct import dct, idct, dct_2d, idct_2d
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
    return dct_2d(x, norm=norm)
    # return dct(dct(x, norm=norm).T, norm=norm).T


def idct2(x, norm='ortho'):
    return idct_2d(x, norm=norm)
    # return idct(idct(x, norm=norm).T, norm=norm).T


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
        # self.feature = torch.zeros((*img_shape, 5)).cuda()

    def forward(self, im, color_feature):
        self.feature[:, :, :, :, 0] = color_feature
        h_feature = self.h_conv(im)
        w_feature = self.w_conv(im)
        self.feature[:, :, :, :, 3] = torch.roll(h_feature, shifts=1, dims=3)
        self.feature[:, :, :, :, 4] = torch.roll(w_feature, shifts=1, dims=2)
        self.feature[:, :, :, :, 1] = h_feature
        self.feature[:, :, :, :, 2] = w_feature
        return self.feature


class GP_model_T1(torch.nn.Module):
    def __init__(self, img_shape, color_weight=8e-10, sigma=0.5, device='cuda', eps=1e-12):
        # img_shape :(B, C, H, W)
        super(GP_model_T1, self).__init__()
        self.gradient_caculater_Dst = Gradient_Caculater(img_shape)
        self.gradient_caculater_Src = Gradient_Caculater(img_shape)
        # self.half()

    def _apply(self, fn):
        super(GP_model_T1, self)._apply(fn)
        return self

    @torch.no_grad()
    def forward(self, src_im, dst_im, mask_im, bg_for_color):
        # src_im = src_im.half()
        # dst_im = dst_im.half()
        # bg_for_color = bg_for_color.half()
        mask_im = mask_im.unsqueeze(dim=-1).float()
        dst_feature = self.gradient_caculater_Dst(dst_im, color_feature=bg_for_color)
        src_feature = self.gradient_caculater_Src(src_im, color_feature=bg_for_color)

        # (B, C, H, W,5)
        # mask_im =mask_im.half()
        X = dst_feature * (1 - mask_im) + src_feature * mask_im

        # fusion

        Fh = (X[:, :, :, :, 1] + torch.roll(X[:, :, :, :, 3], -1, 3)) / 2
        Fv = (X[:, :, :, :, 2] + torch.roll(X[:, :, :, :, 4], -1, 2)) / 2
        # Fh = X[:, :, :, :, 1]
        # Fv = X[:, :, :, :, 2]
        L = torch.roll(Fh, 1, 3) + torch.roll(Fv, 1, 2) - Fh - Fv
        # L = X[:, :, :, :, 3] + X[:, :, :, :, 4] - Fh - Fv

        # Xdct = dct2(X[:, :, :, :, 0])  # 原图每个通
        return X[:, :, :, :, 0],L  # BGR To RGB，+++time


class GP_model_T2(torch.nn.Module):
    def __init__(self, img_shape, color_weight=8e-10, sigma=0.5, device='cuda', eps=1e-12):
        # img_shape :(B, C, H, W)
        super(GP_model_T2, self).__init__()
        size = img_shape[-2:]
        param_l = laplacian_param_torch(size, device)
        param_g = gaussian_param_torch(size, sigma, device)
        self.param_total = param_l + color_weight * param_g
        self.param_total[(self.param_total >= 0) & (self.param_total < eps)] = eps
        self.param_total[(self.param_total < 0) & (self.param_total > -eps)] = -eps
        self.color_weight = color_weight
        self.dct_2d_module = dctmodule2D(img_shape)
        self.idct_2d_module = idctmodule2D(img_shape)
        # self.Y = torch.zeros(img_shape[1:])
        # self.Y = torch.nn.Parameter(torch.zeros(img_shape))
        # self.half()

    def _apply(self, fn):
        super(GP_model_T2, self)._apply(fn)
        self.param_total = fn(self.param_total)
        return self

    @torch.no_grad()
    def forward(self, X, L):
        Xdct = self.dct_2d_module(X)  # 7 ms
        # Ydct = (dct2(L) + self.color_weight * Xdct) / self.param_total
        Ydct = (self.dct_2d_module(L) + self.color_weight * Xdct) / self.param_total
        # results = idct2(Ydct)  # 15 ms
        results =self.idct_2d_module(Ydct)
        results = torch.clamp(results, min=-0, max=255)
        return results  # BGR To RGB，+++time

class GPU_model_GP_MultiStage():
    def __init__(self, img_shape, color_weight=8e-10, sigma=0.5, gpu=0):
        if gpu >= 0:
            device = f'cuda:{gpu}'
        else:
            device = 'cpu'
        self.infer_model_T1 = GP_model_T1(img_shape=img_shape, color_weight=color_weight, sigma=sigma,
                                    device=device).eval().to(device)
        self.infer_model_T2 = GP_model_T2(img_shape=img_shape, color_weight=color_weight, sigma=sigma,
                                    device=device).eval().to(device)

        # -----------------------------------
        # input_name = ['src_im', 'dst_im', 'mask_im', 'bg_for_color']
        # output_name = ['output']
        # from torch.autograd import Variable
        # input = Variable(torch.randn(1, 3, 1080, 1920)).cuda()
        # mask = Variable(torch.randn(1, 1, 1080, 1920)).cuda()
        # torch.onnx.export(self.infer_model, (input,input,mask,input), 'GP_model.onnx',
        #                   input_names=input_name, output_names=output_name, verbose=True, opset_version=13)
        # -----------------------------------
        # print(torch2trt.CONVERTERS)
        x = torch.ones((1, 3, 2160, 3840)).float().cuda()
        mask = torch.ones((1, 1, 2160, 3840)).float().cuda()
        model_trt = torch2trt(self.infer_model_T1, [x, x, mask, x])

    @torch.no_grad()
    def GP_GPU_Model_fusion(self, obj, bg, mask, gpu=0):
        torch.cuda.synchronize()
        T00 = time.time()
        if gpu >= 0:
            device = f'cuda:{gpu}'
        else:
            device = 'cpu'
        w_orig, h_orig, _ = obj.shape
        obj = torch.from_numpy(obj[np.newaxis]).to(device).float().permute(0, 3, 1, 2)
        bg = torch.from_numpy(bg[np.newaxis]).to(device).float().permute(0, 3, 1, 2)
        mask = torch.from_numpy(mask[np.newaxis][np.newaxis]).to(device)
        torch.cuda.synchronize()
        print('Data TO GPU TIME', time.time() - T00)
        ############################ Gaussian-Poisson GAN Image Editing ###########################
        torch.cuda.synchronize()
        T1 = time.time()
        X,L = self.infer_model_T1(obj, bg, mask, bg)
        gan_ims = self.infer_model_T2(X,L)
        gan_ims = gan_ims.permute(0, 2, 3, 1).int()
        torch.cuda.synchronize()
        print('Infer TIME (Not CPU to GPU and GPU to CPU)', time.time() - T1)
        torch.cuda.synchronize()
        T2 = time.time()
        gan_ims = gan_ims.cpu().numpy()  #
        # gan_ims = gan_ims.astype(np.uint8)
        # gan_im = np.clip(gan_im * 255, 0, 255).astype(np.uint8)
        torch.cuda.synchronize()
        print('Data to Memory', time.time() - T2)
        print('Infer TIME', time.time() - T00)
        return gan_ims
