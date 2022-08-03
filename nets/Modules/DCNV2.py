"""Deformable ConvNets v2 in PyTorch."""

import math
from typing import Tuple, Union

import torch
import torchvision.ops
from torch import nn
from torch.nn.modules.utils import _pair


class DeformableConv2d(nn.Module):
    """Deformable ConvNets v2 as described in
    `Deformable ConvNets v2: More Deformable, Better Results
    <https://arxiv.org/abs/1811.11168>`__.

    Attributes:
        in_channels (int): Number of channels in the input image.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (Tuple[int, int]): Size of the convolving kernel.
        stride (Tuple[int, int]): Stride of the convolution.
        padding (Tuple[int, int]): Padding added to all four sides of the
            input.
        dilation (Tuple[int, int]): Spacing between kernel elements.
        deformable_groups (int): Used to determine the number of offset groups.
        weight (torch.nn.Parameter): Convolution weights.
        bias (torch.nn.Parameter): Bias terms for convolution.
        conv_offset_mask (torch.nn.Conv2d): 2D convolution to generate the
            offset and mask.

    Args:
        in_channels (int): Number of channels in the input image.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (Union[int, Tuple[int, int])): Size of the convolving
            kernel.
        stride (Union[int, Tuple[int, int]]): Stride of the convolution.
        padding (Union[int, Tuple[int, int]]): Padding added to all four sides
            of the input.
        dilation (Union[int, Tuple[int, int]]): Spacing between kernel
            elements.
        deformable_groups (int): Used to determine the number of offset groups.

    Reference:
        PyTorch-Deformable-Convolution-v2:
        https://github.com/developer0hye/PyTorch-Deformable-Convolution-v2

        Deformable Convolutional Networks V2 with Pytorch 1.0:
        https://github.com/ifzhang/DCNv2/tree/pytorch_1.7
    """

    # pylint: disable=redefined-builtin,too-many-instance-attributes

    num_chunks = 3  # Num channels for offset + mask

    def __init__(  # pylint: disable=too-many-arguments
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Union[int, Tuple[int, int]],
        padding: Union[int, Tuple[int, int]],
        dilation: Union[int, Tuple[int, int]] = 1,
        deformable_groups: int = 1,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.deformable_groups = deformable_groups

        self.weight = nn.Parameter(
            torch.Tensor(out_channels, in_channels, *self.kernel_size)
        )
        self.bias = nn.Parameter(torch.Tensor(out_channels))
        self.reset_parameters()

        num_offset_mask_channels = (
            self.deformable_groups * self.num_chunks * self.kernel_size[0] * self.kernel_size[1]
        )
        self.conv_offset_mask = nn.Conv2d(
            self.in_channels,
            num_offset_mask_channels,
            self.kernel_size,
            self.stride,
            self.padding,
            self.dilation,
            bias=True,
        )
        self.init_offset()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Defines the computation performed at every call.

        Args:
            input (torch.Tensor): Input from the previous layer.

        Returns:
            (torch.Tensor): Result of convolution.
        """
        out = self.conv_offset_mask(input)
        offset_1, offset_2, mask = torch.chunk(out, self.num_chunks, dim=1)
        offset = torch.cat((offset_1, offset_2), dim=1)
        mask = torch.sigmoid(mask)

        return torchvision.ops.deform_conv2d(
            input=input,
            offset=offset,
            weight=self.weight,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            mask=mask,
        )

    def init_offset(self) -> None:
        """Initializes the weight and bias for `conv_offset_mask`."""
        self.conv_offset_mask.weight.data.zero_()
        if self.conv_offset_mask.bias is not None:
            self.conv_offset_mask.bias.data.zero_()

    def reset_parameters(self) -> None:
        """Re-initialize parameters using a method similar to He
        initialization with mode='fan_in' and gain=1.
        """
        fan_in = self.in_channels
        for k in self.kernel_size:
            fan_in *= k
        std = 1.0 / math.sqrt(fan_in)
        self.weight.data.uniform_(-std, std)
        self.bias.data.zero_()
