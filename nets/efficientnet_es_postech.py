# from KETI_Prune.KETI_Pruner import ConvNormActivation
import torch
import torch.fx
import math
import numpy as np
import copy
from functools import partial
from torch import nn, Tensor
from typing import Any, Callable, List, Optional, Sequence
import pickle
from collections import OrderedDict

__all__ = ["EfficientNet_ES_Postech","efficientnet_es_postech"]

IMAGENET_DEFAULT_MEAN_OPENEDGE = (0.5019, 0.5019, 0.5019)
IMAGENET_DEFAULT_STD_OPENEDGE = (0.5019, 0.5019, 0.5019)

class ConvNormActivation(torch.nn.Sequential):
    ''' Conv2d, Normalization, activation 을 순서대로 실행하는 Sequential 모듈 '''
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int = 3,
            stride: int = 1,
            padding: Optional[int] = None,
            groups: int = 1,
            norm_layer: Optional[Callable[..., torch.nn.Module]] = torch.nn.BatchNorm2d,
            activation_layer: Optional[Callable[..., torch.nn.Module]] = torch.nn.ReLU,
            dilation: int = 1,
            inplace: bool = True,
            no_5x5: bool = False
    ) -> None:
        if padding is None:
            padding = (kernel_size - 1) // 2 * dilation

        if kernel_size == 5 and no_5x5:
            kernel_size = 3
            padding = (kernel_size - 1) // 2 * dilation
            layers = [torch.nn.Conv2d(in_channels, out_channels, kernel_size, 1, padding,
                                      dilation=dilation, groups=groups, bias=norm_layer is None)]

            if norm_layer is not None:
                layers.append(norm_layer(out_channels))

            layers.append(torch.nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding,
                                          dilation=dilation, groups=groups, bias=norm_layer is None))

        else:
            layers = [torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding,
                                      dilation=dilation, groups=groups, bias=norm_layer is None)]
        if norm_layer is not None:
            layers.append(norm_layer(out_channels))
        if activation_layer is not None:
            layers.append(activation_layer(inplace=inplace))
        super().__init__(*layers)
        self.out_channels = out_channels
        
def stochastic_depth(input: Tensor, p: float, mode: str, training: bool = True) -> Tensor:
    """
    Implements the Stochastic Depth from `"Deep Networks with Stochastic Depth"
    <https://arxiv.org/abs/1603.09382>`_ used for randomly dropping residual
    branches of residual architectures.

    Args:
        input (Tensor[N, ...]): The input tensor or arbitrary dimensions with the first one
                    being its batch i.e. a batch with ``N`` rows.
        p (float): probability of the input to be zeroed.
        mode (str): ``"batch"`` or ``"row"``.
                    ``"batch"`` randomly zeroes the entire input, ``"row"`` zeroes
                    randomly selected rows from the batch.
        training: apply stochastic depth if is ``True``. Default: ``True``

    Returns:
        Tensor[N, ...]: The randomly zeroed tensor.
    """
    '''
    if not torch.jit.is_scripting() and not torch.jit.is_tracing():
        _log_api_usage_once(stochastic_depth)
    '''
    if p < 0.0 or p > 1.0:
        raise ValueError(f"drop probability has to be between 0 and 1, but got {p}")
    if mode not in ["batch", "row"]:
        raise ValueError(f"mode has to be either 'batch' or 'row', but got {mode}")
    if not training or p == 0.0:
        return input

    survival_rate = 1.0 - p
    if mode == "row":
        size = [input.shape[0]] + [1] * (input.ndim - 1)
    else:
        size = [1] * input.ndim
    noise = torch.empty(size, dtype=input.dtype, device=input.device)
    noise = noise.bernoulli_(survival_rate)
    if survival_rate > 0.0:
        noise.div_(survival_rate)
    return input * noise



torch.fx.wrap("stochastic_depth")


class StochasticDepth(nn.Module):
    """
    See :func:`stochastic_depth`.
    """
    def __init__(self, p: float, mode: str) -> None:
        super().__init__()
        #_log_api_usage_once(self)
        self.p = p
        self.mode = mode

    def forward(self, input: Tensor) -> Tensor:
        return stochastic_depth(input, self.p, self.mode, self.training)

    def __repr__(self) -> str:
        s = f"{self.__class__.__name__}(p={self.p}, mode={self.mode})"
        return s


def _make_divisible(v: float, divisor: int, min_value: Optional[int] = None) -> int:
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

class MBConvConfig:
    # Stores information listed at Table 1 of the EfficientNet paper
    def __init__(
        self,
        num_layers: int,
        kernel: int,
        stride: int,
        expand_ratio: float,
        in_channels: int,
        out_channels: int,
        fused_conv  : bool,
        identity : bool,
        width_mult: float,
        depth_mult: float,
        expanded_channels: int = None,
        expanded_channels_fix : bool = False
    ) -> None:
        self.expand_ratio = expand_ratio
        self.kernel = kernel
        self.stride = stride
        self.in_channels = self.adjust_channels(in_channels, width_mult)

        self.expanded_channels_fix = expanded_channels_fix

        # Pruning 결과로 channel 을 fix 할때 사용
        if expanded_channels_fix is True:
            self.expanded_channels = expanded_channels

        self.out_channels = self.adjust_channels(out_channels, width_mult)
        self.num_layers = self.adjust_depth(num_layers, depth_mult)
        self.fused_conv = fused_conv
        self.identity = identity

    def __repr__(self) -> str:
        s = self.__class__.__name__ + "("
        s += "num_layers={num_layers}"
        s += ", kernel={kernel}"
        s += ", stride={stride}"
        s += ", expand_ratio={expand_ratio}"
        s += ", in_channels={in_channels}"
        s += ", expanded_channels={expanded_channels}"
        s += ", out_channels={out_channels}"
        s += ", fused={fused_conv}"
        s += ", identity={identity}"
        s += ")"
        return s.format(**self.__dict__)

    @staticmethod
    def adjust_channels(channels: int, width_mult: float, min_value: Optional[int] = None) -> int:
        return _make_divisible(channels * width_mult, 8, min_value)

    @staticmethod
    def adjust_depth(num_layers: int, depth_mult: float):
        return int(math.ceil(num_layers * depth_mult))


class MBConvBlockWithoutDepthwise(nn.Module):
    def __init__(
            self,
            cnf: MBConvConfig,
            stochastic_depth_prob: float,
            norm_layer: Callable[..., nn.Module],
    ):
        super().__init__()
        if not (1 <= cnf.stride <= 2):
            raise ValueError("illegal stride value")

        self.use_res_connect = (cnf.stride == 1 and cnf.in_channels == cnf.out_channels)

        layers: List[nn.Module] = []
        activation_layer = nn.ReLU

        # expand
        if not cnf.expanded_channels_fix:
            expanded_channels = cnf.adjust_channels(cnf.in_channels, cnf.expand_ratio)
        else:
            expanded_channels = cnf.expanded_channels

        if expanded_channels != cnf.in_channels:
            layers.append(
                ConvNormActivation(
                    cnf.in_channels,
                    expanded_channels,
                    stride=cnf.stride,
                    kernel_size=3,
                    norm_layer=norm_layer,
                    activation_layer=activation_layer,
                )
            )

        # project
        layers.append(
            ConvNormActivation(
                expanded_channels,
                cnf.out_channels,
                kernel_size=1,
                norm_layer=norm_layer,
                activation_layer=None,
            )
        )

        self.block = nn.Sequential(*layers)
        self.stochastic_depth = StochasticDepth(stochastic_depth_prob, "row")
        self.out_channels = cnf.out_channels
        self._is_cn = cnf.stride > 1
        self.index_mask_info1 = {'mask':torch.ones([1, expanded_channels, 1, 1]),
                                'KL':np.zeros(expanded_channels),
                                'L1':np.zeros(expanded_channels),
                                'L2':np.zeros(expanded_channels)}
        self.index_mask_info2 = {'mask': torch.ones([1, cnf.out_channels, 1, 1]),
                                 'KL': np.zeros(cnf.out_channels),
                                 'L1': np.zeros(cnf.out_channels),
                                 'L2': np.zeros(cnf.out_channels)}
        self.importanceCalMode = False

    def forward(self, input: Tensor) -> Tensor:
        if self.importanceCalMode:
            result = self.block[0](input)
            mask = self.index_mask_info1['mask'].expand(result.size()).cuda()
            result = result.mul(mask)

            result = self.block[1](result)
            mask = self.index_mask_info2['mask'].expand(result.size()).cuda()
            result = result.mul(mask)
        else:
            result = self.block(input)

        if self.use_res_connect:
            result = self.stochastic_depth(result)
            result += input
        return result

class MBConvBlock(nn.Module):
    def __init__(
        self,
        cnf: MBConvConfig,
        stochastic_depth_prob: float,
        norm_layer: Callable[..., nn.Module],
    ):
        super().__init__()
        if not (1 <= cnf.stride <= 2):
            raise ValueError("illegal stride value")

        self.use_res_connect = (
            cnf.stride == 1 and cnf.in_channels == cnf.out_channels
        )

        layers: List[nn.Module] = []
        activation_layer = nn.ReLU

        # expand
        if not cnf.expanded_channels_fix:
            expanded_channels = cnf.adjust_channels(cnf.in_channels, cnf.expand_ratio)
        else:
            expanded_channels = cnf.expanded_channels

        if expanded_channels != cnf.in_channels:
            layers.append(
                ConvNormActivation(
                    cnf.in_channels,
                    expanded_channels,
                    kernel_size=1,
                    norm_layer=norm_layer,
                    activation_layer=activation_layer,
                )
            )

        # depthwise
        layers.append(
            ConvNormActivation(
                expanded_channels,
                expanded_channels,
                kernel_size=cnf.kernel,
                stride=cnf.stride,
                groups=expanded_channels,
                norm_layer=norm_layer,
                activation_layer=activation_layer,
            )
        )

        # project
        layers.append(
            ConvNormActivation(
                expanded_channels,
                cnf.out_channels,
                kernel_size=1,
                norm_layer=norm_layer,
                activation_layer=None,
            )
        )

        self.block = nn.Sequential(*layers)
        self.stochastic_depth = StochasticDepth(stochastic_depth_prob, "row")
        self.out_channels = cnf.out_channels
        self._is_cn = cnf.stride > 1

        self.index_mask_info1 = {'mask': torch.ones([1, expanded_channels, 1, 1]),
                                 'KL': np.zeros(expanded_channels),
                                 'L1': np.zeros(expanded_channels),
                                 'L2': np.zeros(expanded_channels)}
        self.index_mask_info2 = {'mask': torch.ones([1, expanded_channels, 1, 1]),
                                 'KL': np.zeros(expanded_channels),
                                 'L1': np.zeros(expanded_channels),
                                 'L2': np.zeros(expanded_channels)}
        self.index_mask_info3 = {'mask': torch.ones([1, cnf.out_channels, 1, 1]),
                                 'KL': np.zeros(cnf.out_channels),
                                 'L1': np.zeros(cnf.out_channels),
                                 'L2': np.zeros(cnf.out_channels)}

        self.importanceCalMode = False

    def forward(self, input: Tensor) -> Tensor:
        if self.importanceCalMode:
            result = self.block[0](input)

            mask = self.index_mask_info1['mask'].expand(result.size()).cuda()
            result = result.mul(mask)

            result = self.block[1](result)
            mask = self.index_mask_info2['mask'].expand(result.size()).cuda()
            result = result.mul(mask)

            result = self.block[2](result)
            mask = self.index_mask_info3['mask'].expand(result.size()).cuda()
            result = result.mul(mask)
        else:
            result = self.block(input)

        if self.use_res_connect:
            result = self.stochastic_depth(result)
            result += input
        return result


class EfficientNet_ES_Postech(nn.Module):
    def __init__(
        self,
        inverted_residual_setting: List[MBConvConfig],
        dropout: float,
        stochastic_depth_prob: float = 0.2,
        num_classes: int = 1000,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        prune_config=None,
    ) -> None:
        """
        EfficientNetb0_postech main class

        Args:
            inverted_residual_setting (List[InvertedResidualConfig]): Network structure
            last_channel (int): The number of channels on the penultimate layer
            num_classes (int): Number of classes
            block (Optional[Callable[..., nn.Module]]): Module specifying inverted residual building block for mobilenet
            norm_layer (Optional[Callable[..., nn.Module]]): Module specifying the normalization layer to use
        """
        super().__init__()

        if not inverted_residual_setting:
            raise ValueError("The inverted_residual_setting should not be empty")
        elif not (
                isinstance(inverted_residual_setting, Sequence)
                and all([isinstance(s, MBConvConfig) for s in inverted_residual_setting])
        ):
            raise TypeError("The inverted_residual_setting should be List[MBConvConfig]")

        if norm_layer is None:
            norm_layer = partial(nn.BatchNorm2d, eps=0.001, momentum=0.01)

        layers: List[nn.Module] = []

        # building first layer
        if prune_config is not None:
            layers.append(
                ConvNormActivation(prune_config['features.0']['in'], prune_config['features.0']['out'], kernel_size=3,
                                 stride=2, norm_layer=norm_layer, activation_layer=nn.ReLU))
        else:
            #layers.append(
            #    ConvNormActivation(3, 32, kernel_size=3, stride=2, norm_layer=norm_layer, activation_layer=nn.ReLU))
            self.conv_stem = ConvNormActivation(3, 32, kernel_size=3, stride=2, norm_layer=norm_layer, activation_layer=nn.ReLU)
            self.conv_stem_index_mask_info = {'mask':torch.ones([1, 32, 1, 1]),
                                              'KL':np.zeros(32),
                                              'L1':np.zeros(32),
                                              'L2':np.zeros(32)}

        # building edge residual blocks
        block = MBConvBlockWithoutDepthwise


        if prune_config is not None:
            featuresID = 1
            for cnf in mb_block_without_dw_setting:
                cnf.input_channels = prune_config[f'features.{featuresID}']['in']
                cnf.expanded_channels = prune_config[f'features.{featuresID}']['mid']
                cnf.out_channels = prune_config[f'features.{featuresID}']['out']
                layers.append(block(cnf, norm_layer))
                featuresID += 1
            # building inverted residual blocks
            block = MBConvBlock

            for cnf in mb_block_setting:
                cnf.input_channels = prune_config[f'features.{featuresID}']['in']
                cnf.expanded_channels = prune_config[f'features.{featuresID}']['mid']
                cnf.out_channels = prune_config[f'features.{featuresID}']['out']
                layers.append(block(cnf, norm_layer))
                featuresID += 1

        else:
            total_stage_blocks = sum(cnf.num_layers for cnf in inverted_residual_setting)
            stage_block_id = 0
            for cnf in inverted_residual_setting:
                stage: List[nn.Module] = []
                for _ in range(cnf.num_layers):
                    # copy to avoid modifications. shallow copy is enough
                    block_cnf = copy.copy(cnf)

                    # overwrite info if not the first conv in the stage
                    if stage:
                        block_cnf.in_channels = block_cnf.out_channels
                        block_cnf.stride = 1
                        block_cnf.identity = True

                    # adjust stochastic depth probability based on the depth of the stage block
                    sd_prob = stochastic_depth_prob * float(stage_block_id) / total_stage_blocks

                    if block_cnf.fused_conv:
                        stage.append(MBConvBlockWithoutDepthwise(block_cnf, sd_prob, norm_layer))
                    else:
                        stage.append(MBConvBlock(block_cnf, sd_prob, norm_layer))
                    stage_block_id += 1

                layers.append(nn.Sequential(*stage))

        # building last several layers
        lastconv_input_channels = inverted_residual_setting[-1].out_channels
        if prune_config is not None:
            lastconv_output_channels = prune_config[f'features.19']['out']
        else:
            lastconv_output_channels = 1280

        #layers.append(
        self.last_layer = ConvNormActivation(
                lastconv_input_channels,
                lastconv_output_channels,
                kernel_size=1,
                norm_layer=norm_layer,
                activation_layer=nn.ReLU,
            )
        #)

        self.last_layer_index_mask_info = {'mask': torch.ones([1, lastconv_output_channels, 1, 1]),
                                                'KL': np.zeros(lastconv_output_channels),
                                                'L1': np.zeros(lastconv_output_channels),
                                                'L2': np.zeros(lastconv_output_channels)}
        self.features = nn.Sequential(*layers)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout, inplace=True),
            nn.Linear(lastconv_output_channels, num_classes),
        )

        self.importanceCalMode = False


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                init_range = 1.0 / math.sqrt(m.out_features)
                nn.init.uniform_(m.weight, -init_range, init_range)
                nn.init.zeros_(m.bias)

    def _forward_impl(self, x: Tensor) -> Tensor:

        if self.importanceCalMode:
            x = self.conv_stem(x)
            mask = self.conv_stem_index_mask_info['mask'].expand(x.size()).cuda()
            x = x.mul(mask)
            x = self.features(x)
            x = self.last_layer(x)
            mask = self.last_layer_index_mask_info['mask'].expand(x.size()).cuda()
            x = x.mul(mask)
        else:
            x = self.conv_stem(x)
            x = self.features(x)
            x = self.last_layer(x)

        x = self.avgpool(x)
        x = x.flatten(1)

        x = self.classifier(x)

        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)

    def change_keys_state_dict(self, ckpt_path):
        tmp_dict = OrderedDict()
        load_dict = torch.load(ckpt_path)

        for i, j in load_dict.items():
            if 'features.0' in i:
                newKey = i.replace('features.0', 'conv_stem')
                tmp_dict[newKey] = j
            elif 'features.7' in i:
                newKey = i.replace('features.7', 'last_layer')
                tmp_dict[newKey] = j
            elif 'classifier' in i:
                tmp_dict[i] = j
            else:
                for o in range(1,7):
                    if f'features.{o}' in i:
                        newKey = i.replace(f'features.{o}', f'features.{o-1}')
                        tmp_dict[newKey] = j

        return tmp_dict


    def save_index_mask_score(self, filepath):

        save_dict = dict()


        for name, module in self.named_modules():
            if name == 'conv_stem':
                save_dict[name] = self.conv_stem_index_mask_info

            elif isinstance(module, MBConvBlockWithoutDepthwise):
                save_dict[ name + '.block.0.0'] = module.index_mask_info1
                save_dict[ name + '.block.1.0'] = module.index_mask_info2
            elif isinstance(module, MBConvBlock):
                save_dict[name + '.block.0.0'] = module.index_mask_info1
                save_dict[name + '.block.1.0'] = module.index_mask_info2
                save_dict[name + '.block.2.0'] = module.index_mask_info3
            elif name == 'last_layer':
                save_dict[name] = self.last_layer_index_mask_info
        '''
        for name, module in self.named_modules():
            if name == 'conv_stem':
                save_dict[name] = np.random.rand( self.conv_stem_index_mask_info['KL'].shape[0] )

            elif isinstance(module, MBConvBlockWithoutDepthwise):
                save_dict[name + '.block.0.0'] = np.random.rand( module.index_mask_info1['KL'].shape[0] )
                save_dict[name + '.block.1.0'] = np.random.rand( module.index_mask_info2['KL'].shape[0] )
            elif isinstance(module, MBConvBlock):
                save_dict[name + '.block.0.0'] = np.random.rand( module.index_mask_info1['KL'].shape[0] )
                save_dict[name + '.block.1.0'] = np.random.rand( module.index_mask_info2['KL'].shape[0] )
                save_dict[name + '.block.2.0'] = np.random.rand( module.index_mask_info3['KL'].shape[0] )
            elif name == 'last_layer':
                save_dict[name] = np.random.rand( self.last_layer_index_mask_info['KL'].shape[0] )
        '''

        with open( filepath , "wb") as fw:
            pickle.dump(save_dict, fw)

def change_keys_state_dict(ckpt_path):
        tmp_dict = OrderedDict()
        load_dict = torch.load(ckpt_path)

        for i, j in load_dict.items():
            if 'features.0' in i:
                newKey = i.replace('features.0', 'conv_stem')
                tmp_dict[newKey] = j
            elif 'features.7' in i:
                newKey = i.replace('features.7', 'last_layer')
                tmp_dict[newKey] = j
            elif 'classifier' in i:
                tmp_dict[i] = j
            else:
                for o in range(1,7):
                    if f'features.{o}' in i:
                        newKey = i.replace(f'features.{o}', f'features.{o-1}')
                        tmp_dict[newKey] = j

        return tmp_dict

def efficientnet_es_postech(pretrained: bool = True, prune_config=None, num_classes=1000,**kwargs: Any) :
    """
    Constructs a small MobileNetV3 architecture from
    `"Searching for MobileNetV3" <https://arxiv.org/abs/1905.02244>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    bneck_conf = partial(MBConvConfig, width_mult=1.0, depth_mult=1.0)

    inverted_residual_setting = [
        bneck_conf(1, 3, 1, 3, 32, 24, True, False),
        bneck_conf(2, 3, 2, 8, 24, 32, True, True),
        bneck_conf(4, 3, 2, 8, 32, 48, True, True),
        bneck_conf(5, 3, 2, 8, 48, 96, False, True),
        bneck_conf(4, 3, 1, 8, 96, 144, False, True),
        bneck_conf(2, 3, 2, 8, 144, 192, False, True),
    ]
    model = EfficientNet_ES_Postech(inverted_residual_setting, dropout=0.2, num_classes=1000,**kwargs)
    
    if pretrained:
        path = './models_imagenet/efficientnet_es_postech/checkpoint/efficientnet_es_postech_original.pth'
        state_dict= change_keys_state_dict(path)
        #state_dict = torch.load(path)
        model.load_state_dict(state_dict)

    return model


