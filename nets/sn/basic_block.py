import math

import torch
import torch.nn as nn
import torch.nn.functional as F
#from backbone.vggnet import PruneConv2d

class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self, vectorSize=0):
        self.avg = torch.zeros(vectorSize).cuda()
        self.sum = torch.zeros(vectorSize).cuda()
        self.count = torch.zeros(vectorSize).cuda()
    def update(self, val, idx, n=1):
        self.sum[idx] += val * n
        self.count[idx] += n
        self.avg[idx] = self.sum[idx] / self.count[idx]

class WrapModule(nn.Module):
    def forward(self, x):
        return self.net(x)

    def __getitem__(self, idx):
        if not hasattr(self, 'net'):
            raise NotImplementedError()

        if isinstance(self.net, (nn.Sequential, nn.ModuleList)):
            return self.net[idx]

        raise NotImplementedError()

    @property
    def weight(self):
        return self.net.weight

    @property
    def bias(self):
        return self.net.bias


class TfOp2d(WrapModule):
    def get_same_pad(self, h, w):
        r, s = self.pair(self.kernel_size)
        stride = self.pair(self.stride)

        h = int(h)
        w = int(w)

        out_h = h // stride[0]
        out_w = w // stride[1]

        pad_h = max((out_h - 1)*stride[0] + r - h, 0)
        pad_w = max((out_w - 1)*stride[1] + s - w, 0)

        pad_t = pad_h // 2
        pad_b = pad_h - pad_t
        pad_l = pad_w // 2
        pad_r = pad_w - pad_l

        return (pad_l, pad_r, pad_t, pad_b)

    def __repr__(self):
        return "{}({})".format(type(self).__name__, self.net.extra_repr())

    def extra_repr(self):
        inherit = self.net.extra_repr().replace(', padding=0', '')

        s = (', padding={padding}')

        return inherit + s.format(**self.__dict__)

    @staticmethod
    def pair(value):
        if isinstance(value, int):
            return (value, value)
        else:
            return value


class TfConv2d(TfOp2d):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding='valid',
                 groups=1,
                 bias=True):
        super().__init__()

        if padding != 'same' and padding != 'valid':
           raise NotImplementedError('Unknown padding method')

        '''
        self.net = PruneConv2d(in_channels,
                             out_channels,
                             kernel_size,
                             stride=stride,
                             padding=0,
                             groups=groups,
                             bias=bias)
        '''
        self.net = nn.Conv2d(in_channels,
                             out_channels,
                             kernel_size,
                             stride=stride,
                             padding=0,
                             groups=groups,
                             bias=bias)


        self.in_channels = self.net.in_channels
        self.out_channels = self.net.out_channels
        self.kernel_size = self.net.kernel_size
        self.stride = self.net.stride
        self.padding = padding
        self.groups = self.net.groups

    def forward(self, x):
        if self.padding == 'same':
            padding = self.get_same_pad(*x.shape[2:4])
            x = F.pad(x, padding)

        return self.net(x)


class TfMaxPool2d(TfOp2d):
    def __init__(self,
                 kernel_size,
                 stride=None,
                 padding='valid'):
        super().__init__()

        if padding != 'same' and padding != 'valid':
           raise NotImplementedError('Unknown padding method')

        self.net = nn.MaxPool2d(kernel_size,
                                 stride=stride,
                                 padding=0)

        self.kernel_size = self.net.kernel_size
        self.stride = self.net.stride
        self.padding = padding

    def forward(self, x):
        if self.padding == 'same':
            padding = self.get_same_pad(*x.shape[2:4])
            x = F.pad(x, padding)

        return self.net(x)


class Conv2d(WrapModule):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 groups=1,
                 bias=True):
        super().__init__()

        if padding in ('same', 'valid'):
            module = TfConv2d
        else:
            #module = PruneConv2d
            module = nn.Conv2d

        self.net = module(in_channels,
                          out_channels,
                          kernel_size,
                          stride=stride,
                          padding=padding,
                          groups=groups,
                          bias=bias)

    def __repr__(self):
        return "{}({})".format(type(self).__name__, self.extra_repr())

    def extra_repr(self):
        return self.net.extra_repr()


class MaxPool2d(WrapModule):
    def __init__(self,
                 kernel_size,
                 stride=None,
                 padding='valid'):
        super().__init__()

        if padding in ('same', 'valid'):
            module = TfMaxPool2d
        else:
            module = nn.MaxPool2d

        self.net = module(kernel_size,
                          stride=stride,
                          padding=padding)

    def __repr__(self):
        return "{}({})".format(type(self).__name__, self.extra_repr())

    def extra_repr(self):
        return self.net.extra_repr()

class Sigmoid(nn.Sigmoid):
    def __init__(self, inplace=True):
        super().__init__()
###        

class SiLU(nn.Module):
    def __init__(self, inplace: bool = False):
        super().__init__()
        self.inplace = inplace

    def forward(self, input):
        return F.silu(input, inplace=self.inplace)

    def extra_repr(self) -> str:
        inplace_str = 'inplace=True' if self.inplace else ''
        return inplace_str

class Swish(SiLU):
    pass

class SE(nn.Module):
    def __init__(self, channels, reduction_ratio=1.0, inplace=True):
        super().__init__()

        exp_channels = int(math.ceil(channels / reduction_ratio))

        se = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                           nn.Conv2d(channels, exp_channels, 1),
                           nn.ReLU(inplace=inplace),
                           nn.Conv2d(exp_channels, channels, 1),
                           Sigmoid())

        self.se = se

        self.reduction_ratio = reduction_ratio

    def forward(self, x):
        return x * self.se(x)


class HardSigmoid(nn.Module):
    def __init__(self, inplace=True):
        super().__init__()

        self.inplace = inplace

    def forward(self, x):
        return F.relu6(x + 3., inplace=self.inplace) / 6.


class HardSwish(nn.Module):
    def __init__(self, inplace=True):
        super().__init__()

        self.hsigmoid = HardSigmoid(inplace=inplace)

    def forward(self, x):
        return self.hsigmoid(x) * x


class HardSE(nn.Module):
    def __init__(self, channels, reduction_ratio=1., inplace=True):
        super().__init__()

        exp_channels = int(math.ceil(channels / reduction_ratio))

        se = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                           nn.Conv2d(channels, exp_channels, 1, bias=False),
                           nn.ReLU(inplace=inplace),
                           nn.Conv2d(exp_channels, channels, 1, bias=False),
                           HardSigmoid(inplace=inplace))

        self.se = se

    def forward(self, x):
        return x * self.se(x)


class SwishSE(nn.Module):
    def __init__(self, channels, reduction_ratio=1.0, inplace=True):
        super().__init__()

        exp_channels = int(math.ceil(channels / reduction_ratio))

        se = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                           nn.Conv2d(channels, exp_channels, 1),
                           nn.ReLU6(inplace=inplace),
                           nn.Conv2d(exp_channels, channels, 1),
                           Sigmoid(inplace=inplace))
        # se = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
        #                    nn.Conv2d(channels, exp_channels, 1),
        #                    Swish(inplace=inplace),
        #                    nn.Conv2d(exp_channels, channels, 1),
        #                    Sigmoid(inplace=inplace))

        self.se = se

        self.reduction_ratio = reduction_ratio

    def forward(self, x):
        return x * self.se(x)


class Residual(nn.Module):
    def __init__(self, drop_path_rate=None):
        super().__init__()

        self.drop_path_rate = drop_path_rate

    def forward(self, x, y):
        if self.training and self.drop_path_rate is not None:
            shape = (y.shape[0],) + (1,) * (y.dim() - 1)

            keep_rate = 1. - self.drop_path_rate
            keep = torch.rand(shape).add_(keep_rate).floor_()

            keep = keep.to(y.device)

            y = keep * y / keep_rate

        return x + y

    def extra_repr(self):
        s = ('drop_path_rate={drop_path_rate}')
        return s.format(**self.__dict__)


class Conv2dBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 groups=1,
                 bias=True,
                 momentum=0.1,
                 reduction_ratio=1.,
                 use_batchnorm=False,
                 use_se=None,
                 activation=None):
        super().__init__()

        if use_batchnorm:
            bias = False

        modules = [Conv2d(in_channels,
                          out_channels,
                          kernel_size,
                          stride=stride,
                          padding=padding,
                          groups=groups,
                          bias=bias)]

        if use_batchnorm:
            modules.append(nn.BatchNorm2d(out_channels,
                                          momentum=momentum))

        if activation == 'relu':
            modules.append(nn.ReLU(inplace=True))
        elif activation == 'relu6':
            modules.append(nn.ReLU6(inplace=True))
        elif activation == 'leaky':
            modules.append(nn.LeakyReLU(0.1, inplace=True))
        elif activation == 'swish':
            modules.append(Swish(inplace=True))
        elif activation == 'hard-swish':
            modules.append(HardSwish(inplace=True))
        elif activation is not None and activation != 'linear':
            raise Exception('unknown activation')

        if use_se == 'hard-se':
            modules.append(HardSE(out_channels,
                                  reduction_ratio=reduction_ratio))
        elif use_se == 'swish-se':
            modules.append(SwishSE(out_channels,
                                   reduction_ratio=reduction_ratio))
        elif use_se == 'se':
            modules.append(SE(out_channels,
                              reduction_ratio=reduction_ratio))

        self.net = nn.Sequential(*modules)

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.stride = stride
        self.padding = padding
        self.groups = groups

        self.use_batchnorm = use_batchnorm
        self.use_se = use_se
        self.activation = activation

    def forward(self, x):
        return self.net(x)

    def __repr__(self):
        return "{}({})".format(type(self).__name__, self.extra_repr())

    def extra_repr(self):
        s = ''

        if self.use_batchnorm:
            s += (', use_batchnorm={use_batchnorm}')

        if self.use_se:
            s += (', use_se={use_se}')

        if self.activation:
            s += (', use_se={activation}')

        return self.net[0].extra_repr() + s.format(**self.__dict__)

    @property
    def weight(self):
        return self.net[0].weight

    @property
    def bias(self):
        return self.net[0].bias
