import copy

from typing import List, Optional, Type, Callable, Dict

import torch.nn as nn

import brevitas.nn as qnn
from brevitas.nn.mixin.parameter import WeightQuantType, BiasQuantType
from brevitas.nn.mixin.act import ActQuantType

from .layer_editing_utils import replace_node_module
from .quant_activation import QuantHardsigmoid, QuantHardswish


CustomQModuleMapping = Dict[Type[nn.Module], Callable[[nn.Module, Dict], nn.Module]]


def conv2d_to_qconv2d(conv: nn.Conv2d, kwargs: Dict):
    return qnn.QuantConv2d(conv.in_channels, conv.out_channels,
                           conv.kernel_size, conv.stride, conv.padding,
                           conv.dilation, conv.groups, conv.padding_mode,
                           conv.bias is not None, **kwargs)


def conv1d_to_qconv1d(conv: nn.Conv1d, kwargs: Dict):
    return qnn.QuantConv1d(conv.in_channels, conv.out_channels,
                           conv.kernel_size, conv.stride, conv.padding,
                           conv.dilation, conv.groups, conv.padding_mode,
                           conv.bias is not None, **kwargs)


def linear_to_qlinear(linear: nn.Linear, kwargs: Dict):
    return qnn.QuantLinear(linear.in_features, linear.out_features, linear.bias
                           is not None, **kwargs)


def adapt2davgpool2d_to_qavgpool2d(pool: nn.AdaptiveAvgPool2d, kwargs: Dict):
    return qnn.TruncAdaptiveAvgPool2d(pool.output_size, **kwargs)


base_qmodule_mapping: CustomQModuleMapping = {
    nn.Conv2d: conv2d_to_qconv2d,
    nn.Conv1d: conv1d_to_qconv1d,
    nn.Linear: linear_to_qlinear,
    nn.AdaptiveAvgPool2d: adapt2davgpool2d_to_qavgpool2d,
}

base_qact_mapping: CustomQModuleMapping = {
    nn.ReLU: lambda _, kwargs: qnn.QuantReLU(**kwargs),
    nn.Hardsigmoid: lambda _, kwargs: QuantHardsigmoid(**kwargs),
    nn.Hardswish: lambda _, kwargs: QuantHardswish(**kwargs)
}


def modules_to_qmodules(model: nn.Module,
                        weight_quant: Optional[WeightQuantType] = None,
                        act_quant: Optional[ActQuantType] = None,
                        bias_quant: Optional[BiasQuantType] = None,
                        skip_modules: Optional[List[type[nn.Module]]] = None,
                        custom_qact_mapping: Optional[CustomQModuleMapping] = None,
                        custom_qmodule_mapping: Optional[CustomQModuleMapping] = None,
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

        # Give priority to custom mappings if provided
        if custom_qact_mapping is not None and type(module) in custom_qact_mapping:
            if act_quant is None:
                continue

            new_child_module = custom_qact_mapping[type(module)](module, {
                'act_quant': act_quant,
                "return_quant_tensor": True
            })
        elif custom_qmodule_mapping is not None and type(module) in custom_qmodule_mapping:
            new_child_module = custom_qmodule_mapping[type(module)](module, {
                'weight_quant': weight_quant,
                'bias_quant': bias_quant
            })
        elif type(module) in base_qmodule_mapping:
            new_child_module = base_qmodule_mapping[type(module)](module, {
                'weight_quant': weight_quant,
                'bias_quant': bias_quant
            })
        elif type(module) in base_qact_mapping:
            if act_quant is None:
                continue

            new_child_module = base_qact_mapping[type(module)](module, {
                'act_quant': act_quant,
                "return_quant_tensor": True
            })

        if new_child_module is not None:
            replace_node_module(model, name, new_child_module)
        else:
            raise NotImplementedError(f"Module '{module.__class__.__name__}' not supported yet.")

    return model