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
        N_height = shape[-2]
        k_height = - torch.nn.Parameter(torch.arange(N_height, dtype=float)[None, :] * np.pi / (2 * N_height))
        self.W_r_height = torch.nn.Parameter(torch.cos(k_height))
        self.W_i_height = torch.nn.Parameter(torch.sin(k_height))

        self.N_weight = N_weight
        self.N_height = N_height

    def forward(self, x, norm='ortho'):
        x_shape = x.shape
        N = self.N_weight
        x = x.contiguous().view(-1, N)
        v = torch.cat([x[:, ::2], x[:, 1::2].flip([1])], dim=1)

        # Vc = torch.rfft(v, 1, onesided=False)              #
        Vc = torch.view_as_real(torch.fft.fft(v, dim=1))  # pytoch 1.9
        V = Vc[:, :, 0] * self.W_r_weight - Vc[:, :, 1] * self.W_i_weight

        if norm == 'ortho':
            V[:, 0] /= np.sqrt(N) * 2
            V[:, 1:] /= np.sqrt(N / 2) * 2
        V = 2 * V.view(*x_shape)

        x = V.transpose(-1, -2)
        x_shape = x.shape
        N = self.N_height
        x = x.contiguous().view(-1, N)
        v = torch.cat([x[:, ::2], x[:, 1::2].flip([1])], dim=1)

        # Vc = torch.rfft(v, 1, onesided=False)              #
        Vc = torch.view_as_real(torch.fft.fft(v, dim=1))  # pytoch 1.9
        V = Vc[:, :, 0] * self.W_r_height - Vc[:, :, 1] * self.W_i_height

        if norm == 'ortho':
            V[:, 0] /= np.sqrt(N) * 2
            V[:, 1:] /= np.sqrt(N / 2) * 2
        V = 2 * V.view(*x_shape)

        return V.transpose(-1, -2)
