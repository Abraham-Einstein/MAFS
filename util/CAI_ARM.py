# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import collections


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, dilation=1, groups=1):
        padding = ((kernel_size - 1) * dilation + 1) // 2
        # padding = (kernel_size - 1) // 2
        super(ConvBNReLU, self).__init__(collections.OrderedDict([
            ('conv', nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, dilation=dilation, groups=groups,
                               bias=False)),
            ('bn', nn.BatchNorm2d(out_planes)),
            ('relu', nn.ReLU6(inplace=True))]))


class ARM_I(nn.Module):
    def __init__(self, low_channel, high_channel, middle):
        super(ARM_I, self).__init__()

        self.dilation_conv1 = ConvBNReLU(low_channel+high_channel+middle, low_channel, dilation=1)
        self.dilation_conv2 = ConvBNReLU(low_channel+high_channel+middle, low_channel, dilation=2)
        self.dilation_conv3 = ConvBNReLU(low_channel+high_channel+middle, low_channel, dilation=4)
        self.conv1 = ConvBNReLU(low_channel*3, low_channel, kernel_size=1)
        self.upx2 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.high_out = ConvBNReLU(middle, low_channel, kernel_size=1)

    def forward(self, low, previous_arm, high):

        if previous_arm.size()[2] != low.size()[2]:
            previous_arm = self.upx2(previous_arm)
            high = self.upx2(high)
        # print(low.shape, previous_arm.shape, high.shape)
        x = torch.cat((low, previous_arm, high), dim=1)
        x1 = self.dilation_conv1(x)
        x2 = self.dilation_conv2(x)
        x3 = self.dilation_conv3(x)
        low = self.conv1(torch.cat((x1, x2, x3), dim=1))
        high = self.high_out(high)
        out1 = low*high
        out2 = out1+high
        return out1, out2


class ARM_II(nn.Module):
    def __init__(self, low_channel, high_channel):
        super(ARM_II, self).__init__()

        self.dilation_conv1 = ConvBNReLU(low_channel+2*high_channel, low_channel, dilation=1)
        self.dilation_conv2 = ConvBNReLU(low_channel+2*high_channel, low_channel, dilation=2)
        self.dilation_conv3 = ConvBNReLU(low_channel+2*high_channel, low_channel, dilation=4)
        self.conv1 = ConvBNReLU(low_channel*3, low_channel, kernel_size=1)
        self.upx2 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.high_out = ConvBNReLU(high_channel, low_channel, kernel_size=1)

    def forward(self, low, previous_arm, high):

        if previous_arm.size()[2] != low.size()[2]:
            previous_arm = self.upx2(previous_arm)
            high = self.upx2(high)
        x = torch.cat((low, previous_arm, high), dim=1)
        x1 = self.dilation_conv1(x)
        x2 = self.dilation_conv2(x)
        x3 = self.dilation_conv3(x)
        low = self.conv1(torch.cat((x1, x2, x3), dim=1))
        high = self.high_out(high)
        out1 = low*high
        out2 = out1+high
        return out1, out2