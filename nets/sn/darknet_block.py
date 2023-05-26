from torch import nn


class DarknetConv2d(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 groups=1,
                 bias=True,
                 has_batchnorm=False,
                 activation=None):

        super().__init__()

        layers = []

        if has_batchnorm:
            bias = False

        conv2d = nn.Conv2d(in_channels,
                           out_channels,
                           kernel_size,
                           stride=stride,
                           padding=padding,
                           groups=groups,
                           bias=bias)

        layers.append(conv2d)

        if has_batchnorm:
            layers.append(nn.BatchNorm2d(out_channels))

        if activation is None or activation == "linear":
            pass
        elif activation == "relu":
            layers.append(nn.ReLU(inplace=True))
        elif activation == "leaky":
            layers.append(nn.LeakyReLU(0.1, inplace=True))
        elif activation == "relu6":
            layers.append(nn.ReLU6(inplace=True))
        else:
            raise Exception("Unknown activation function %s" % activation)

        self.subnet = nn.Sequential(*layers)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.has_batchnorm = has_batchnorm

    def forward(self, x):
        return self.subnet(x)


class DarknetReorg(nn.Module):
    def __init__(self, stride):
        super().__init__()

        self.stride = stride

    def forward(self, x):
        n, c, h, w = x.shape
        stride = self.stride

        h_ = h // stride
        w_ = w // stride

        # Bug patched version of Reorg(=Reorg3d)
        x = x.reshape(n, c, h_, stride, w_, stride)
        x = x.permute(0, 3, 5, 1, 2, 4).reshape(n, -1, h_, w_)

        return x
