import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import numbers


class GaussianSmoothing(nn.Module):
    """
    Apply gaussian smoothing on a
    1d, 2d or 3d tensor. Filtering is performed seperately for each channel
    in the input using a depthwise convolution.
    Arguments:
        channels (int, sequence): Number of channels of the input tensors. Output will
            have this number of channels as well.
        kernel_size (int, sequence): Size of the gaussian kernel.
        sigma (float, sequence): Standard deviation of the gaussian kernel.
        dim (int, optional): The number of dimensions of the data.
            Default value is 2 (spatial).
    """

    def __init__(self, channels, kernel_size, sigma, dim=2):
        super(GaussianSmoothing, self).__init__()
        if isinstance(kernel_size, numbers.Number):
            kernel_size = [kernel_size] * dim
        if isinstance(sigma, numbers.Number):
            sigma = [sigma] * dim

        # The gaussian kernel is the product of the
        # gaussian function of each dimension.
        kernel = 1
        meshgrids = torch.meshgrid(
            [
                torch.arange(size, dtype=torch.float32)
                for size in kernel_size
            ]
        )
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= 1 / (std * math.sqrt(2 * math.pi)) * \
                      torch.exp(-((mgrid - mean) / (2 * std)) ** 2)

        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / torch.sum(kernel)

        # Reshape to depthwise convolutional weight
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

        self.register_buffer('weight', kernel.float())
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
            filtered (torch.Tensor).weight: Filtered output.
        """
        return self.conv(input, weight=self.weight, groups=self.groups)


# 图向的宽上面做梯度1920-> 1922 -> 1920,逐行求梯度
class normal_h(nn.Module):
    def __init__(self, channels=3, dim=2):
        super(normal_h, self).__init__()
        # Reshape to depthwise convolutional
        self.weight =torch.tensor([[0., -1., 1.]])
        self.weight = self.weight.view(1, 1, *self.weight.size())
        self.weight = self.weight.repeat(channels, *[1] * (self.weight.dim() - 1))
        self.weight = torch.nn.Parameter(self.weight.double())

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
        p2d = (1, 1, 0, 0)
        input = F.pad(input, p2d, 'replicate')
        return self.conv(input, weight=self.weight, groups=self.groups)


class normal_w(nn.Module):
    def __init__(self, channels=3, dim=2):
        super(normal_w, self).__init__()

        self.weight = torch.tensor([[0., -1., 1.]]).T
        self.weight = self.weight.view(1, 1, *self.weight.size())
        self.weight = self.weight.repeat(channels, *[1] * (self.weight.dim() - 1))
        self.weight = torch.nn.Parameter(self.weight.double())


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
        p2d = (0, 0, 1, 1)
        input = F.pad(input, p2d, 'replicate')
        return self.conv(input, weight=self.weight, groups=self.groups)
