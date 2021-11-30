import time
import torch
import numpy as np
from module.dct_module import dctmodule2D, idctmodule2D
from module.conv_module import normal_h, normal_w


def fft2(K, size):
    w, h = size
    # param = torch.fft(K,signal_ndim=1)      #torch 1.4 错，这是复数到复数
    # param = torch.rfft(K, signal_ndim=1,onesided=False)  # torch 1.4
    # param = np.fft.fft2(K)
    param = torch.fft.fft2(K)  # torch 1.1  1.9
    param = torch.real(param[0:w, 0:h])

    return param


def laplacian_param_init(size, device='cuda'):
    w, h = size
    K = torch.zeros((2 * w, 2 * h)).to(device)

    laplacian_k = torch.tensor([[0, -1, 0], [-1, 4, -1], [0, -1, 0]]).cuda()
    kw, kh = laplacian_k.shape
    K[:kw, :kh] = laplacian_k

    K = torch.roll(K, -(kw // 2), 0)
    K = torch.roll(K, -(kh // 2), 1)

    return fft2(K, size)


def gaussian_param_init(size, sigma, device='cuda'):
    w, h = size
    K = torch.zeros((2 * w, 2 * h)).to(device)
    K[:3, :3] = torch.tensor([[0.01133, 0.08373, 0.01133],
                              [0.08373, 0.61869, 0.08373],
                              [0.01133, 0.08373, 0.01133]])
    K = torch.roll(K, -1, 0)
    K = torch.roll(K, -1, 1)
    return fft2(K, size)


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


# GP的Pyotorch模型
class GP_model(torch.nn.Module):
    def __init__(self, img_shape, color_weight=8e-10, sigma=0.5, device='cuda', eps=1e-12, half_flag=True):
        # img_shape :(B, C, H, W)
        super(GP_model, self).__init__()
        self.gradient_caculater_Dst = Gradient_Caculater(img_shape)
        self.gradient_caculater_Src = Gradient_Caculater(img_shape)
        size = img_shape[-2:]
        param_l = laplacian_param_init(size, device)
        param_g = gaussian_param_init(size, sigma, device)
        self.param_total = param_l + color_weight * param_g
        self.param_total[(self.param_total >= 0) & (self.param_total < eps)] = eps
        self.param_total[(self.param_total < 0) & (self.param_total > -eps)] = -eps
        self.color_weight = color_weight
        self.dct_2d_module = dctmodule2D(img_shape)
        self.idct_2d_module = idctmodule2D(img_shape)
        # self.Y = torch.zeros(img_shape[1:])
        # self.Y = torch.nn.Parameter(torch.zeros(img_shape))
        if half_flag:
            self.half_flag = True
            self.half()

    def _apply(self, fn):
        super(GP_model, self)._apply(fn)
        self.param_total = fn(self.param_total)
        return self

    @torch.no_grad()
    def forward(self, src_im, dst_im, mask_im, bg_for_color):
        if self.half_flag:
            src_im = src_im.half()
            dst_im = dst_im.half()
            bg_for_color = bg_for_color.half()
            mask_im = mask_im.unsqueeze(dim=-1).float().half()

        dst_feature = self.gradient_caculater_Dst(dst_im, color_feature=bg_for_color)
        src_feature = self.gradient_caculater_Src(src_im, color_feature=bg_for_color)
        # src_feature = self.gradient_caculater_Src(src_im, color_feature=src_im)

        # (B, C, H, W,5)
        X = dst_feature * (1 - mask_im) + src_feature * mask_im

        # fusion

        Fh = (X[:, :, :, :, 1] + torch.roll(X[:, :, :, :, 3], -1, 3)) / 2
        Fv = (X[:, :, :, :, 2] + torch.roll(X[:, :, :, :, 4], -1, 2)) / 2
        # Fh = X[:, :, :, :, 1]
        # Fv = X[:, :, :, :, 2]
        L = torch.roll(Fh, 1, 3) + torch.roll(Fv, 1, 2) - Fh - Fv
        # L = X[:, :, :, :, 3] + X[:, :, :, :, 4] - Fh - Fv

        # Xdct = dct2(X[:, :, :, :, 0].float())  # 原图每个通道
        Xdct = self.dct_2d_module(X[:, :, :, :, 0])  # 7 ms
        #
        # Ydct = (dct2(L.float()) + self.color_weight * Xdct) / self.param_total
        Ydct = (self.dct_2d_module(L) + self.color_weight * Xdct) / self.param_total
        #
        # results = idct2(Ydct)  # 15 ms
        results = self.idct_2d_module(Ydct).half()

        # results = results[:, [2, 1, 0], :, :]
        results = torch.clamp(results, min=-0, max=255)

        return results  # BGR To RGB，+++time


# 包裹类
class GPU_GP_Container():
    def __init__(self, img_shape, color_weight=8e-10, sigma=0.5, gpu=0):
        if gpu >= 0:
            device = f'cuda:{gpu}'
        else:
            device = 'cpu'
        self.infer_model = GP_model(img_shape=img_shape, color_weight=color_weight, sigma=sigma,
                                    device=device).eval().to(device)
        for param in self.infer_model.parameters():
            param.grad = None

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
        gan_ims = self.infer_model(obj, bg, mask, bg)
        gan_ims = gan_ims.permute(0, 2, 3, 1).int()
        torch.cuda.synchronize()
        print('Infer TIME (Not CPU to GPU and GPU to CPU)', time.time() - T1)
        torch.cuda.synchronize()
        T2 = time.time()
        gan_ims = gan_ims.cpu().numpy()  #
        torch.cuda.synchronize()
        print('Data to Memory', time.time() - T2)
        print('Infer TIME', time.time() - T00)
        return gan_ims
