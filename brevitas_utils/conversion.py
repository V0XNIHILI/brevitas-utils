import copy

from typing import List, Optional, Type, Callable, Dict

import torch.nn as nn

import brevitas.nn as qnn
from brevitas.nn.mixin.parameter import WeightQuantType, BiasQuantType
from brevitas.nn.mixin.act import ActQuantType

from .layer_editing_utils import replace_node_module
from .quant_activation import QuantBSiLU, QuantHardsigmoid, QuantHardswish

def conv2d_to_qconv2d(conv: nn.Conv2d, **kwargs):
    return qnn.QuantConv2d(conv.in_channels, conv.out_channels,
                           conv.kernel_size, conv.stride, conv.padding,
                           conv.dilation, conv.groups, conv.padding_mode,
                           conv.bias is not None, **kwargs)


def conv1d_to_qconv1d(conv: nn.Conv1d, **kwargs):
    return qnn.QuantConv1d(conv.in_channels, conv.out_channels,
                           conv.kernel_size, conv.stride, conv.padding,
                           conv.dilation, conv.groups, conv.padding_mode,
                           conv.bias is not None, **kwargs)


def linear_to_qlinear(linear: nn.Linear, **kwargs):
    return qnn.QuantLinear(linear.in_features, linear.out_features, linear.bias
                           is not None, **kwargs)

def act_to_qact(act: nn.Module, **kwargs):
    act_map = {
        nn.ReLU: qnn.QuantReLU,
        nn.Hardsigmoid: QuantHardsigmoid,
        nn.Hardswish: QuantHardswish
    }
    return act_map[act.__class__](**kwargs)

def adapt2davgpool2d_to_qavgpool2d(pool: nn.AdaptiveAvgPool2d, **kwargs):
    return qnn.TruncAdaptiveAvgPool2d(pool.output_size, **kwargs)

                
def modules_to_qmodules(model = None,
                        weight_quant: Optional[WeightQuantType]= None,
                        act_quant: Optional[ActQuantType] = None,
                        bias_quant: Optional[BiasQuantType] = None,
                        skip_modules: Optional[List[type[nn.Module]]] = None,
                        custom_qact_mapping: CustomQModuleMapping = None,
                        custom_qmodule_mapping: CustomQModuleMapping = None,
                        inplace: bool =False):
    if not inplace:
        model = copy.deepcopy(model)

    for name, module in model.named_modules():
        if (skip_modules != None and isinstance(module, tuple(skip_modules))) or isinstance(module, nn.Identity) or isinstance(module, nn.Sequential):
            continue

        # If the module is just some container module
        if name == "":
            continue

        if module.__class__.__name__ == "Module":
            continue

        new_child_module = None

        if isinstance(module, nn.Conv1d):
            new_child_module = conv1d_to_qconv1d(module,
                                                 weight_quant=weight_quant,
                                                 bias_quant=bias_quant)
        elif isinstance(module, nn.Conv2d):
            new_child_module = conv2d_to_qconv2d(module,
                                                 weight_quant=weight_quant,
                                                 bias_quant=bias_quant)
        elif isinstance(module, nn.Linear):
            new_child_module = linear_to_qlinear(module,
                                                 weight_quant=weight_quant,
                                                 bias_quant=bias_quant)
        elif isinstance(module, nn.AdaptiveAvgPool2d):
            new_child_module = adapt2davgpool2d_to_qavgpool2d(module,
                                                             trunc_quant = act_quant,
                                                             return_quant_tensor=True)
            
        elif isinstance(module, (nn.ReLU, nn.Hardsigmoid, nn.Hardswish)):
            if act_quant is None:
                continue
            
            new_child_module = act_to_qact(module, act_quant=act_quant, return_quant_tensor=True)

        if new_child_module is not None:
            replace_node_module(model, name, new_child_module)
        else:
            raise NotImplementedError(f"Module '{module.__class__.__name__}' not supported yet.")

    return model