import copy

from typing import List, Union

import torch.nn as nn

import brevitas.nn as qnn
from brevitas.inject import ExtendedInjector

def conv2d_to_qconv2d(conv: nn.Conv2d, **kwargs):
    return qnn.QuantConv2d(conv.in_channels, conv.out_channels,
                           conv.kernel_size, conv.stride, conv.padding,
                           conv.dilation, conv.groups, conv.bias is not None,
                           conv.padding_mode, **kwargs)


def conv1d_to_qconv1d(conv: nn.Conv1d, **kwargs):
    return qnn.QuantConv1d(conv.in_channels, conv.out_channels,
                           conv.kernel_size, conv.stride, conv.padding,
                           conv.dilation, conv.groups, conv.bias is not None,
                           conv.padding_mode, **kwargs)


def linear_to_qlinear(linear: nn.Linear, **kwargs):
    return qnn.QuantLinear(linear.in_features, linear.out_features, linear.bias
                           is not None, **kwargs)


def modules_to_qmodules(model: nn.Module,
                        weight_quant: ExtendedInjector,
                        act_quant: ExtendedInjector,
                        bias_quant: Union[ExtendedInjector, None] = None,
                        skip_modules: List[type[nn.Module]] = [],
                        inplace=False):
    if not inplace:
        model = copy.deepcopy(model)

    for name, module in model.named_modules():
        if isinstance(module, tuple(skip_modules)):
            continue

        new_child_module = None

        if isinstance(module, nn.Conv1d):
            new_child_module = conv1d_to_qconv1d(module,
                                                 weight_quant=weight_quant,
                                                #  weight_scaling_per_output_channel=True,
                                                 bias_quant=bias_quant)
        elif isinstance(module, nn.Conv2d):
            new_child_module = conv2d_to_qconv2d(module,
                                                 weight_quant=weight_quant,
                                                 bias_quant=bias_quant)
        elif isinstance(module, nn.Linear):
            new_child_module = linear_to_qlinear(module,
                                                 weight_quant=weight_quant,
                                                 bias_quant=bias_quant)
        elif isinstance(module, nn.ReLU):
            new_child_module = qnn.QuantReLU(act_quant=act_quant,
                                             return_quant_tensor=True)

        if new_child_module is not None:
            replace_node_module(model, name, new_child_module)

    return model