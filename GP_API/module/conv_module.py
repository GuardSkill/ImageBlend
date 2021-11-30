import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import numbers

# 图向的宽上面做梯度1920-> 1922 -> 1920,逐行求梯度
class normal_h(nn.Module):
    def __init__(self, channels=3, dim=2):
        super(normal_h, self).__init__()
        # Reshape to depthwise convolutional
        self.weight =torch.tensor([[0., -1., 1.]])
        self.weight = self.weight.view(1, 1, *self.weight.size())
        self.weight = self.weight.repeat(channels, *[1] * (self.weight.dim() - 1))
        self.weight = torch.nn.Parameter(self.weight)
        self.p2d = (1, 1, 0, 0)
        self.groups = channels
        if dim == 1:
            self.conv = F.conv1d
        elif dim == 2:
            self.conv = F.conv2d
        elif dim == 3:
            self.conv = F.conv3d
        else:
            raise RuntimeError(
                'Only 1, 2 and 3 dimensions are supported. Received {}.'.format(dim)
            )

    def forward(self, input):
        """
        Apply gaussian filter to input.
        Arguments:
            input (torch.Tensor): Input to apply gaussian filter on.
        Returns:
            filtered (torch.Tensor): Filtered output.
        """
        input = F.pad(input, self.p2d, 'replicate')
        return self.conv(input, weight=self.weight, groups=self.groups)


class normal_w(nn.Module):
    def __init__(self, channels=3, dim=2):
        super(normal_w, self).__init__()

        self.weight = torch.tensor([[0., -1., 1.]]).T
        self.weight = self.weight.view(1, 1, *self.weight.size())
        self.weight = self.weight.repeat(channels, *[1] * (self.weight.dim() - 1))
        self.weight = torch.nn.Parameter(self.weight)
        self.p2d = (0, 0, 1, 1)

        self.groups = channels
        if dim == 1:
            self.conv = F.conv1d
        elif dim == 2:
            self.conv = F.conv2d
        elif dim == 3:
            self.conv = F.conv3d
        else:
            raise RuntimeError(
                'Only 1, 2 and 3 dimensions are supported. Received {}.'.format(dim)
            )

    def forward(self, input):
        """
        Apply gaussian filter to input.
        Arguments:
            input (torch.Tensor): Input to apply gaussian filter on.
        Returns:
            filtered (torch.Tensor): Filtered output.
        """

        input = F.pad(input, self.p2d, 'replicate')
        return self.conv(input, weight=self.weight, groups=self.groups)
