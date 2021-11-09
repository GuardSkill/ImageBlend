import time

import torch
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import numbers
import numpy as np


class dctmodule(nn.Module):
    def __init__(self, shape):
        super(dctmodule, self).__init__()
        N = shape[-1]
        k = torch.nn.Parameter(- torch.arange(N, dtype=float)[None, :] * np.pi / (2 * N))
        self.W_r = torch.cos(k)
        self.W_i = torch.sin(k)

    def forward(self, x, norm=None):
        x_shape = x.shape
        N = x_shape[-1]
        x = x.contiguous().view(-1, N)
        v = torch.cat([x[:, ::2], x[:, 1::2].flip([1])], dim=1)

        # Vc = torch.rfft(v, 1, onesided=False)              #
        Vc = torch.view_as_real(torch.fft.fft(v, dim=1))  # pytoch 1.9
        V = Vc[:, :, 0] * self.W_r - Vc[:, :, 1] * self.W_i

        if norm == 'ortho':
            V[:, 0] /= np.sqrt(N) * 2
            V[:, 1:] /= np.sqrt(N / 2) * 2
        V = 2 * V.view(*x_shape)
        return V


class dctmodule2D(nn.Module):
    def __init__(self, shape):
        super(dctmodule2D, self).__init__()
        N_weight = shape[-1]
        k_weight = - torch.arange(N_weight, dtype=float)[None, :] * np.pi / (2 * N_weight)
        self.W_r_weight = torch.nn.Parameter(torch.cos(k_weight))
        self.W_i_weight = torch.nn.Parameter(torch.sin(k_weight))

        self.inverted_index_weight = torch.arange(N_weight - 1, 0, -2)
        self.index_weight = torch.arange(0, N_weight, 2)

        N_height = shape[-2]
        k_height = - torch.arange(N_height, dtype=float)[None, :] * np.pi / (2 * N_height)
        self.W_r_height = torch.nn.Parameter(torch.cos(k_height))
        self.W_i_height = torch.nn.Parameter(torch.sin(k_height))
        self.inverted_index_height = torch.arange(N_height - 1, 0, -2)
        self.index_height = torch.arange(0, N_height, 2)

        self.N_weight = N_weight
        self.N_height = N_height

    def _apply(self, fn):
        super(dctmodule2D, self)._apply(fn)
        self.inverted_index_weight = fn(self.inverted_index_weight)
        self.inverted_index_height = fn(self.inverted_index_height)
        self.W_r_weight = fn(self.W_r_weight)
        self.W_i_weight = fn(self.W_i_weight)
        self.W_r_height = fn(self.W_r_height)
        self.W_i_height = fn(self.W_i_height)
        return self

    def forward(self, x, norm='ortho'):

        x_shape = x.shape
        N = self.N_weight
        x = x.contiguous().view(-1, N)
        # v = torch.cat([x[:, ::2], x[:, 1::2].flip([1])], dim=1)
        v = torch.cat([x[:, self.index_weight], x[:, self.inverted_index_weight]], dim=1)

        # Vc = torch.rfft(v, 1, onesided=False)              #

        Vc = torch.view_as_real(torch.fft.fft(v.float(), dim=1))  # pytoch 1.9
        # Vc = torch.view_as_real(torch.fft.fft(v, dim=1))
        V = Vc[:, :, 0] * self.W_r_weight - Vc[:, :, 1] * self.W_i_weight

        if norm == 'ortho':
            V[:, 0] /= np.sqrt(N) * 2
            V[:, 1:] /= np.sqrt(N / 2) * 2
        V = 2 * V.view(*x_shape)

        x = V.transpose(-1, -2)

        x_shape = x.shape
        N = self.N_height
        x = x.contiguous().view(-1, N)
        # v = torch.cat([x[:, ::2], x[:, 1::2].flip([1])], dim=1)
        v = torch.cat([x[:, self.index_height], x[:, self.inverted_index_height]], dim=1)

        # Vc = torch.rfft(v, 1, onesided=False)
        # torch.cuda.synchronize()
        # TEST_TIME = time.time()
        Vc = torch.view_as_real(torch.fft.fft(v, dim=1))  # pytoch 1.9   第二次fft最耗时
        # Vc = torch.view_as_real(torch.fft.fft(v, dim=1))
        # Vc = Vc
        # torch.cuda.synchronize()
        # print('TEST TIME', (time.time() - TEST_TIME) * 1000)
        V = Vc[:, :, 0] * self.W_r_height - Vc[:, :, 1] * self.W_i_height

        if norm == 'ortho':
            V[:, 0] /= np.sqrt(N) * 2
            V[:, 1:] /= np.sqrt(N / 2) * 2
        V = 2 * V.view(*x_shape)
        V = V.transpose(-1, -2)

        return V


class idctmodule2D(nn.Module):
    def __init__(self, shape):
        super(idctmodule2D, self).__init__()
        N_weight = shape[-1]
        k_weight = torch.arange(N_weight, dtype=float)[None, :] * np.pi / (2 * N_weight)
        self.W_r_weight = torch.nn.Parameter(torch.cos(k_weight))
        self.W_i_weight = torch.nn.Parameter(torch.sin(k_weight))

        self.inverted_index_weight = torch.arange(N_weight - 1, 0, -2)
        self.index_weight = torch.arange(0, N_weight, 2)

        N_height = shape[-2]
        k_height = torch.arange(N_height, dtype=float)[None, :] * np.pi / (2 * N_height)
        self.W_r_height = torch.nn.Parameter(torch.cos(k_height))
        self.W_i_height = torch.nn.Parameter(torch.sin(k_height))
        self.inverted_index_height = torch.arange(N_height - 1, 0, -2)
        self.index_height = torch.arange(0, N_height, 2)

        self.N_weight = N_weight
        self.N_height = N_height

    def _apply(self, fn):
        super(idctmodule2D, self)._apply(fn)
        self.inverted_index_weight = fn(self.inverted_index_weight)
        self.inverted_index_height = fn(self.inverted_index_height)
        self.W_r_weight = fn(self.W_r_weight)
        self.W_i_weight = fn(self.W_i_weight)
        self.W_r_height = fn(self.W_r_height)
        self.W_i_height = fn(self.W_i_height)
        return self

    def forward(self, X, norm='ortho'):
        x_shape = X.shape
        N = self.N_weight
        X_v = X.contiguous().view(-1, x_shape[-1]) / 2

        if norm == 'ortho':
            X_v[:, 0] *= np.sqrt(N) * 2
            X_v[:, 1:] *= np.sqrt(N / 2) * 2

        V_t_r = X_v
        V_t_i = torch.cat([X_v[:, :1] * 0, -X_v.flip([1])[:, :-1]], dim=1)

        V_r = V_t_r * self.W_r_weight - V_t_i * self.W_i_weight
        V_i = V_t_r * self.W_i_weight + V_t_i * self.W_r_weight

        V = torch.cat([V_r.unsqueeze(2), V_i.unsqueeze(2)], dim=2)

        # v = torch.irfft(V, 1, onesided=False)
        v = torch.fft.irfft(torch.view_as_complex(V), n=V.shape[1], dim=1)  # torch 1.9
        # v = torch.fft.irfft(torch.view_as_complex(V.float()), n=V.shape[1], dim=1).half()  # torch 1.9
        x = v.new_zeros(v.shape)
        x[:, ::2] += v[:, :N - (N // 2)]
        x[:, 1::2] += v.flip([1])[:, :N // 2]

        x = x.view(*x_shape)

        X = x.transpose(-1, -2)

        x_shape = X.shape
        N = self.N_height
        X_v = X.contiguous().view(-1, x_shape[-1]) / 2

        if norm == 'ortho':
            X_v[:, 0] *= np.sqrt(N) * 2
            X_v[:, 1:] *= np.sqrt(N / 2) * 2

        V_t_r = X_v
        V_t_i = torch.cat([X_v[:, :1] * 0, -X_v.flip([1])[:, :-1]], dim=1)

        V_r = V_t_r * self.W_r_height - V_t_i * self.W_i_height
        V_i = V_t_r * self.W_i_height + V_t_i * self.W_r_height

        V = torch.cat([V_r.unsqueeze(2), V_i.unsqueeze(2)], dim=2)

        # v = torch.irfft(V, 1, onesided=
        v = torch.fft.irfft(torch.view_as_complex(V), n=V.shape[1], dim=1)
        # v = torch.fft.irfft(torch.view_as_complex(V.float()), n=V.shape[1], dim=1).half()  # torch 1.9

        x = v.new_zeros(v.shape)
        x[:, ::2] += v[:, :N - (N // 2)]
        x[:, 1::2] += v.flip([1])[:, :N // 2]

        x = x.view(*x_shape)

        return x.transpose(-1, -2)
