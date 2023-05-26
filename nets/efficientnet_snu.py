"""
    My Module
    ~~~~~~~~~
"""

import nets.sn as sn
import torch

from .model_snu import Model
#from backbone.vggnet import PruneConv2d
import pickle

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

class Conv(sn.Module):
    """ Convolution 연산을 수행하는 클래스
    :param in_channels: input channel
    :param out_channels: output channel
    """

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, groups=1,
                 use_batchnorm=True, activation='relu6'):
        super().__init__()

        layers = []
        layers.append(sn.Conv2d(in_channels, out_channels, kernel_size,
                                stride=stride, padding=padding,
                                groups=groups))

        if use_batchnorm:
            layers.append(sn.BatchNorm2d(out_channels, momentum=0.01))

        if activation == 'relu':
            layers.append(sn.ReLU(inplace=True))
        elif activation == 'relu6':
            layers.append(sn.ReLU6(inplace=True))
        elif activation == 'swish':
            layers.append(sn.Swish(inplace=True))
        elif activation == 'hswish':
            layers.append(sn.HardSwish(inplace=True))
        elif activation is not None:
            raise NotImplementedError

        self.net = sn.Sequential(*layers)

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def forward(self, x):
        return self.net(x)


class MbConv(sn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 t=1,
                 se=None,
                 activation='relu6',
                 drop_path_rate=None,
                 ex_channels=None):
        super().__init__()

        layers = []

        params = {
            'use_batchnorm': True,
            'activation': activation
        }

        if ex_channels is None:
            ex_channels = in_channels * t

        if in_channels != ex_channels:
            layers.append(Conv(in_channels, ex_channels, 1,
                               stride=1, **params))

        layers.append(Conv(ex_channels, ex_channels, kernel_size,
                           stride=stride, padding=padding,
                           groups=ex_channels, activation=activation))

        ratio = 4. * t

        if se == "se":
            layers.append(sn.SE(ex_channels, reduction_ratio=ratio))
        elif se == "swish-se":
            layers.append(sn.SwishSE(ex_channels, reduction_ratio=ratio))
        elif se == "hse":
            layers.append(sn.HardSE(ex_channels, reduction_ratio=ratio))

        layers.append(Conv(ex_channels, out_channels, 1, activation=None))

        if in_channels == out_channels and stride == 1:
            residual = sn.Residual(drop_path_rate)
        else:
            residual = None

        self.net = sn.Sequential(*layers)
        self.residual = residual

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def forward(self, x):
        y = self.net(x)

        if self.residual:
            y = self.residual(x, y)

        return y


class MbConvN(sn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 t=1,
                 n=1,
                 se=None,
                 activation='relu6',
                 drop_path_rate=None,
                 mid_channels = None):
        super().__init__()

        if mid_channels is None:
            layers = [MbConv(in_channels,
                             out_channels,
                             kernel_size,
                             stride,
                             padding,
                             t,
                             se,
                             activation,
                             drop_path_rate)]

            for i in range(1, n):
                layers.append(MbConv(out_channels,
                                     out_channels,
                                     kernel_size,
                                     1,
                                     padding,
                                     t,
                                     se,
                                     activation,
                                     drop_path_rate))
        else:
            layers = [MbConv(in_channels,
                             out_channels,
                             kernel_size,
                             stride,
                             padding,
                             t,
                             se,
                             activation,
                             drop_path_rate, mid_channels[0])]

            for i in range(1, n):
                layers.append(MbConv(out_channels,
                                     out_channels,
                                     kernel_size,
                                     1,
                                     padding,
                                     t,
                                     se,
                                     activation,
                                     drop_path_rate, mid_channels[i]))

        self.net = sn.Sequential(*layers)

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def forward(self, x):
        return self.net(x)


class EfficientNet(Model):
    # layout: type, channels, kernel_size, stride, padding, t, n
    presets = {
        'default': {
            'se': 'swish-se',
            'activation': 'swish',#relu6
            'stochastic_depth': 0.2,
            'drop_rate': 0.2,
            'num_class': 1000,

            'layout': (['c', 32, 3, 2, 'same', 0, 1],
                       ['i', 16, 3, 1, 'same', 1, 1],
                       ['i', 24, 3, 2, 'same', 6, 2],
                       ['i', 40, 5, 2, 'same', 6, 2],
                       ['i', 80, 3, 2, 'same', 6, 3],
                       ['i', 112, 5, 1, 'same', 6, 3],
                       ['i', 192, 5, 2, 'same', 6, 4],
                       ['i', 320, 3, 1, 'same', 6, 1])
        },

        'enlight': {
            'layout': (['c', 32, 3, 2, 'same', 0, 1],
                       ['i', 16, 3, 1, 'same', 1, 1],
                       ['i', 24, 3, 2, 'same', 6, 2],
                       ['i', 40, 3, 2, 'same', 6, 4],
                       ['i', 80, 3, 2, 'same', 6, 3],
                       ['i', 112, 3, 1, 'same', 6, 6],
                       ['i', 192, 3, 2, 'same', 6, 8],
                       ['i', 320, 3, 1, 'same', 6, 1]),

            'se': None,
        },

        'efficientnet-b0': {
            'width': 224,
            'height': 224,
            'scale_w': 1.0,
            'scale_d': 1.0,
            'drop_rate': 0.2,
        },

        'efficientnet-b1': {
            'width': 224,
            'height': 224,
            'input_size': 240,
            'scale_w': 1.0,
            'scale_d': 1.1,
            'drop_rate': 0.286,
        },

        'efficientnet-b2': {
            'width': 260,
            'height': 260,
            'scale_w': 1.1,
            'scale_d': 1.2,
            'drop_rate': 0.371,
        },

        'efficientnet-b3': {
            'width': 300,
            'height': 300,
            'scale_w': 1.2,
            'scale_d': 1.4,
            'drop_rate': 0.457,
        },

        'efficientnet-b4': {
            'width': 380,
            'height': 380,
            'scale_w': 1.4,
            'scale_d': 1.8,
            'drop_rate': 0.543,
        },

        'efficientnet-b5': {
            'width': 456,
            'height': 456,
            'scale_w': 1.6,
            'scale_d': 2.2,
            'drop_rate': 0.629,
        },

        'efficientnet-b6': {
            'width': 528,
            'height': 528,
            'scale_w': 1.8,
            'scale_d': 2.6,
            'drop_rate': 0.714,
        },

        'efficientnet-b7': {
            'width': 600,
            'height': 600,
            'scale_w': 2.0,
            'scale_d': 3.1,
            'drop_rate': 0.8,
        },

        'efficientnet-enlight-b0': {
            'inherit': ('efficientnet-b0', 'enlight')
        },

        'efficientnet-enlight-b1': {
            'inherit': ('efficientnet-b1', 'enlight')
        },

        'efficientnet-enlight-b2': {
            'inherit': ('efficientnet-b2', 'enlight')
        },

        'efficientnet-enlight-b3': {
            'inherit': ('efficientnet-b3', 'enlight')
        },

        'efficientnet-enlight-b4': {
            'inherit': ('efficientnet-b4', 'enlight')
        },

        'efficientnet-enlight-b5': {
            'inherit': ('efficientnet-b5', 'enlight')
        },

        'efficientnet-enlight-b6': {
            'inherit': ('efficientnet-b6', 'enlight')
        },

        'efficientnet-enlight-b7': {
            'inherit': ('efficientnet-b7', 'enlight')
        }
    }

    def __init__(self, num_classes, preset='efficientnet-enlight-b0', params=None,
                 pretrained=False, prune_config=None):
        super().__init__(preset=preset, params=params)

        # extract parameters
        params = self.params

        num_class = num_classes#params['num_class']
        scale_w = params['scale_w']
        scale_d = params['scale_d']

        se = params['se']
        activation = params['activation']

        drop_rate = params['drop_rate']
        layout = params['layout']

        for layer in layout:
            layer[1] = self._align(layer[1] * scale_w, 8)
            layer[6] = self._ceil(layer[6] * scale_d)

        if prune_config is not None:
            layout = prune_config['prune_config']

        in_channels = 3

        features = []
        layerIdx = 0
        for layer in layout:

            if layer[5] != -1:
                if layer[0] == 'i':
                    features.append(MbConvN(in_channels, *layer[1:],
                                            se=se,
                                            activation=activation,
                                            drop_path_rate=drop_rate))
                elif layer[0] == 'c':
                    features.append(Conv(in_channels, *layer[1:5],
                                         activation=activation))
                else:
                    raise Exception('Invalid parameter')
            else:
                if layer[0] == 'i':
                    features.append(MbConvN(in_channels, *layer[1:],
                                            se=se,
                                            activation=activation,
                                            drop_path_rate=drop_rate, mid_channels=prune_config['prune_midchs'][layerIdx]))

            in_channels = layer[1]
            layerIdx += 1

        if prune_config is None:
            regressions = (Conv(in_channels, 1280, 1, activation=activation),
                           sn.AdaptiveAvgPool2d((1, 1)),
                           sn.Dropout(drop_rate),
                           sn.Conv2d(1280, num_class, 1))
        else:
            regression_ch = prune_config['prune_midchs'][-1][0]
            regressions = (Conv(in_channels, regression_ch, 1, activation=activation),
                           sn.AdaptiveAvgPool2d((1, 1)),
                           sn.Dropout(drop_rate),
                           sn.Conv2d(regression_ch, num_class, 1))

        self.features = sn.Sequential(*features)
        self.regression = sn.Sequential(*regressions)

        self.initialize_model(pretrained)

        self.stochastic_depth(params['stochastic_depth'])

    def getPrunedArchitectureList(self):
        retList = list()
        midChList = list()

        retList.append(['c', self.features[0].net[0].net.net.out_channels, 3, 2, 'same', 0, 1])
        midChList.append([0])

        retList.append(['i', self.features[1].net[0].net[1].net[0].net.out_channels, 3, 1, 'same', 1, 1])
        midChList.append([0])

        retList.append(['i', self.features[2].net[1].net[2].net[0].net.out_channels, 3, 2, 'same', -1, 2])
        midChList.append([self.features[2].net[0].net[1].net[0].net.net.out_channels, self.features[2].net[1].net[1].net[0].net.net.out_channels])

        retList.append(['i', self.features[3].net[3].net[2].net[0].net.out_channels, 3, 2, 'same', -1, 4])
        midChList.append([self.features[3].net[0].net[1].net[0].net.net.out_channels, self.features[3].net[1].net[1].net[0].net.net.out_channels,
                          self.features[3].net[2].net[1].net[0].net.net.out_channels, self.features[3].net[3].net[1].net[0].net.net.out_channels ])

        retList.append(['i', self.features[4].net[2].net[2].net[0].net.out_channels, 3, 2, 'same', -1, 3])
        midChList.append([self.features[4].net[0].net[1].net[0].net.net.out_channels, self.features[4].net[1].net[1].net[0].net.net.out_channels,
                          self.features[4].net[2].net[1].net[0].net.net.out_channels])

        retList.append(['i', self.features[5].net[5].net[2].net[0].net.out_channels, 3, 1, 'same', -1, 6])
        midChList.append([self.features[5].net[0].net[1].net[0].net.net.out_channels, self.features[5].net[1].net[1].net[0].net.net.out_channels,
                          self.features[5].net[2].net[1].net[0].net.net.out_channels, self.features[5].net[3].net[1].net[0].net.net.out_channels,
                          self.features[5].net[4].net[1].net[0].net.net.out_channels, self.features[5].net[5].net[1].net[0].net.net.out_channels])

        retList.append(['i', self.features[6].net[7].net[2].net[0].net.out_channels, 3, 2, 'same', -1, 8])
        midChList.append([self.features[6].net[0].net[1].net[0].net.net.out_channels, self.features[6].net[1].net[1].net[0].net.net.out_channels,
                          self.features[6].net[2].net[1].net[0].net.net.out_channels, self.features[6].net[3].net[1].net[0].net.net.out_channels,
                          self.features[6].net[4].net[1].net[0].net.net.out_channels, self.features[6].net[5].net[1].net[0].net.net.out_channels,
                          self.features[6].net[6].net[1].net[0].net.net.out_channels, self.features[6].net[7].net[1].net[0].net.net.out_channels])

        retList.append(['i', self.features[7].net[0].net[2].net[0].net.out_channels, 3, 1, 'same', -1, 1])
        midChList.append([self.features[7].net[0].net[1].net[0].net.net.out_channels])

        midChList.append([self.regression[0].net[0].net.out_channels])

        architect_dict = {'prune_config':retList, 'prune_midchs':midChList}
        return architect_dict

    # def save_index_mask_score(self, filepath):
    #     save_dict = dict()
    #     for name, module in self.named_modules():
    #         if isinstance(module, PruneConv2d):
    #             #weightKey = name + '.net.0.weight'
    #             #biastKey = name + '.net.0.bias'

    #             # Change to original key
    #             '''
    #             if weightKey in self.changedKeyDict.keys():
    #                 saveWeightKey = self.changedKeyDict[weightKey]
    #             if biastKey in self.changedKeyDict.keys():
    #                 saveBiasKey = self.changedKeyDict[biastKey]

    #             saveKey = saveWeightKey[0:-7]
    #             '''

    #             #saveKey = weightKey[0:-7]

    #             save_dict[name] = module.index_mask_info

    #     with open(filepath, "wb") as fw:
    #         pickle.dump(save_dict, fw)

    def forward(self, x):
        x = self.features(x)

        return self.regression(x).flatten(start_dim=1)

    
def efficientnet(pretrained=True, progress=False, num_classes=1000):
    model = EfficientNet(num_classes)
    if pretrained:
        #path = './models_imagenet/efficientnet/checkpoint/KETI_efficientnet-enlight-snu_77.92.pth'
        path = './models_imagenet/efficientnet/checkpoint/KETI_efficientnet-enlight-snu_76.24_v2.pth'
        state_dict = torch.load(path)
      
        model.load_state_dict(state_dict)
    return model

