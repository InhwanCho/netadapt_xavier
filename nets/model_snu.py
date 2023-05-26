import os
import re
import math
import torch
from nets import sn
from collections.abc import Iterable


class Model(sn.Module):
    presets = {
        'default': {}
    }

    def __init__(self, preset='default', params=None):
        super().__init__()

        # merge network parameters
        self.params = {
            'postfix': None,

            'width': 224,
            'height': 224,
        }

        if 'default' in self.presets.keys():
            self.apply_params(self.presets['default'])

        if preset in self.presets.keys():
            self.apply_params(self.presets[preset])
        else:
            keys = [k for k in self.presets.keys()]
            prog = re.compile('^(.+?)(-v[1-9]+)?(-large|-small)?$')

            for k in reversed(keys):
                matched = prog.match(k)

                prefix = matched[1]
                suffix = matched[2] if matched[2] is not None else ''
                suffix += matched[3] if matched[3] is not None else ''

                # target pattern is ...
                #    'prefix(input_size)-suffix(-nose)(-width_multiplier)'
                #
                # ex) ssdlite300-v3-nose-1.25
                #
                pattern  = '^'
                pattern += prefix              # ssdlite
                pattern += '([0-9]+)?'         # none or 300
                pattern += suffix              # none or -v3
                pattern += '(-nose)?'          # none or -nose
                pattern += '(-[0-9.]+)?'       # none or -1.25
                pattern += '(-voc|-coco)?'     # none or -coco or -voc
                pattern += '$'

                matched = re.match(pattern, preset)

                if not matched:
                    continue

                self.apply_params(self.presets[k])

                size = matched[1]
                no_se = matched[2]
                width_multiplier = matched[3][1:] if matched[3] else None
                dataset = matched[4][1:] if matched[4] else None

                if size:
                    self.apply_params(self.presets[size])
                if dataset:
                    self.apply_params(self.presets[dataset])
                if no_se:
                    self.params['override_se'] = None
                if width_multiplier:
                    self.params['width_multiplier'] = float(width_multiplier)

                break
            else:
                raise Exception('Unrecognized preset')

        self.apply_params(params)

        postfix = self.params['postfix']

        name = preset
        if postfix is not None:
            name = name + '-' + postfix

        self.name = name

    def initialize_model(self, pretrained=False):
        if pretrained:
            filename = os.path.join('checkpoints', self.name + '_pretrained.pth')
            self.load_state_dict(torch.load(filename))
        else:
            self.initialize_weights()

    def initialize_weights(self):
        pass

    def get_input_size(self):
        return self.params['width'], self.params['height']

    @staticmethod
    def get_regression_method():
        return "softmax"

    @classmethod
    def get_out_channels(cls, layer):
        if hasattr(layer, 'out_channels'):
            return layer.out_channels
        if isinstance(layer, sn.Sequential):
            return cls.get_out_channels(layer[-1])
        elif isinstance(layer, sn.BatchNorm2d):
            return layer.num_features
        else:
            raise Exception("failed to guess input channel width")

    def apply_params(self, params):
        if params is None:
            return

        if 'inherit' in params.keys():
            parents = params['inherit']

            if not (isinstance(parents, tuple) or isinstance(parents, list)):
                parents = (parents, )

            for parent in parents:
                self.apply_params(self.presets[parent])

        for k, v in params.items():
            if k == 'inherit':
                continue

            self.params[k] = v

    def stochastic_depth(self, drop_path_rate):
        num_residual = 0
        for m in self.modules():
            if isinstance(m, sn.Residual):
                num_residual += 1

        if num_residual == 0:
            return

        delta_drop_path_rate = drop_path_rate / (num_residual - 1)
        drop_path_rate = 0.
        for m in self.modules():
            if isinstance(m, sn.Residual):
                m.drop_path_rate = drop_path_rate
                drop_path_rate += delta_drop_path_rate

    @classmethod
    def _align(cls, value, to):
        return int(cls._ceil(value) + to - 1) & (~(to - 1))

    @staticmethod
    def _ceil(value):
        return math.ceil(value)
