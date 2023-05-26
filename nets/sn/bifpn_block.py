import torch

from nets import sn
import torch.nn.functional as F

class ByPass(sn.Module):
    @staticmethod
    def forward(x):
        return x


# Similar to SeparableConv of MobileNet V1,
#    but has no activation for intermediate...

class SeparableConv(sn.WrapModule):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding='same',
                 bias=True,
                 momentum=0.01,
                 use_batchnorm=False,
                 use_se=None,
                 activation=None):
        super().__init__()

        modules = []

        # Depthwise 3x3
        modules.append(sn.Conv2dBlock(in_channels,
                                      in_channels,
                                      kernel_size,
                                      stride=stride,
                                      padding=padding,
                                      groups=in_channels,
                                      bias=False,
                                      use_batchnorm=False,
                                      use_se=use_se,
                                      activation=None))

        # Pointwise
        modules.append(sn.Conv2dBlock(in_channels,
                                      out_channels,
                                      1,
                                      bias=bias,
                                      momentum=momentum,
                                      use_batchnorm=use_batchnorm,
                                      use_se=use_se,
                                      activation=activation))

        self.net = sn.Sequential(*modules)

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.stride = stride
        self.padding = padding

    def forward(self, x):
        return self.net(x)


class BottomUpBlock(sn.Module):
    def __init__(self, channels, activation='swish'):
        super().__init__()

        self.upsample = sn.Upsample(scale_factor=2.)

        self.conv = SeparableConv(channels,
                                  channels,
                                  3,
                                  padding='same',
                                  use_batchnorm=True)

        self.weights = sn.Parameter(torch.ones(2))

        if activation == 'swish':
            self.act = sn.Swish(inplace=True)
        elif activation == 'relu':
            self.act = sn.ReLU(inplace=True)
        elif activation == 'relu6':
            self.act = sn.ReLU6(inplace=True)
        elif activation is None:
            self.act = None
        else:
            raise Exception('not supported activation')

    def forward(self, *x):
        w = F.relu(self.weights)
        w = w / (w.sum() + 1e-16)

        y = x[0] * w[0] + self.upsample(x[1]) * w[1]

        if self.act:
            y = self.act(y)

        return self.conv(y)


class TopDownBlock(sn.Module):
    def __init__(self,
                 channels,
                 kernel_size,
                 is_bottom=False,
                 activation='swish'):
        super().__init__()

        num_inputs = 3 if not is_bottom else 2
        downsample = sn.MaxPool2d(kernel_size, 2, 'same')

        weights = sn.Parameter(torch.ones(num_inputs))

        self.is_bottom = is_bottom
        self.weights = weights

        self.downsample = downsample

        self.conv = SeparableConv(channels,
                                  channels,
                                  3,
                                  padding='same',
                                  use_batchnorm=True)

        if activation == 'swish':
            self.act = sn.Swish(inplace=True)
        elif activation == 'relu':
            self.act = sn.ReLU(inplace=True)
        elif activation == 'relu6':
            self.act = sn.ReLU6(inplace=True)
        elif activation is None:
            self.act = None
        else:
            raise Exception('not supported activation')

    def forward(self, *x):
        w = F.relu(self.weights)
        w = w / (w.sum() + 1e-16)

        y = self.downsample(x[0]) * w[0]
        y = y + x[1] * w[1]

        if not self.is_bottom:
            y = y + x[2] * w[2]

        if self.act:
            y = self.act(y)

        return self.conv(y)


class BiFpnBlock(sn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 num_level,
                 activation='swish'):
        super().__init__()

        bottom_up_stems = []
        bottom_up = []
        for i in range(0, num_level-1):
            if in_channels[i] != out_channels:
                bottom_up_stems.append(sn.Conv2dBlock(in_channels[i],
                                                      out_channels,
                                                      1,
                                                      momentum=0.01,
                                                      use_batchnorm=True))
            else:
                bottom_up_stems.append(ByPass())

            bottom_up.append(BottomUpBlock(out_channels,
                                           activation=activation))

        top_down_stems = []
        top_down = []

        for i in range(1, num_level):
            if in_channels[i] != out_channels:
                top_down_stems.append(sn.Conv2dBlock(in_channels[i],
                                                     out_channels,
                                                     1,
                                                     momentum=0.01,
                                                     use_batchnorm=True))
            else:
                top_down_stems.append(ByPass())

            is_bottom = i == (num_level - 1)
            top_down.append(TopDownBlock(out_channels,
                                         kernel_size,
                                         is_bottom=is_bottom,
                                         activation=activation))

        self.num_level = num_level

        self.bottom_up_stems = sn.ModuleList(bottom_up_stems)
        self.bottom_up = sn.ModuleList(bottom_up)

        self.top_down_stems = sn.ModuleList(top_down_stems)
        self.top_down = sn.ModuleList(top_down)

    def forward(self, x):
        num_level = self.num_level

        # bottom up path
        tmp = []
        y = x[-1]

        for i in range(1, num_level):
            x_ = self.bottom_up_stems[-i](x[-i-1])
            y = self.bottom_up[-i](x_, y)
            tmp.insert(0, y)

        # top down path
        out = []

        y = tmp[0]
        out.append(y)

        for i in range(1, num_level-1):
            x_ = self.top_down_stems[i-1](x[i])
            y = self.top_down[i-1](y, x_, tmp[i])
            out.append(y)

        x_ = self.top_down_stems[-1](x[-1])
        y = self.top_down[-1](y, x_)
        out.append(y)

        return out


class BiFpn(sn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 num_level,
                 repeat,
                 activation='swish'):
        super().__init__()

        bifpn = []

        for i in range(0, repeat):
            bifpn.append(BiFpnBlock(in_channels,
                                    out_channels,
                                    kernel_size,
                                    num_level,
                                    activation=activation))

            in_channels = [out_channels for _ in in_channels]

        self.bifpn = sn.Sequential(*bifpn)

    def forward(self, x):
        return self.bifpn(x)
