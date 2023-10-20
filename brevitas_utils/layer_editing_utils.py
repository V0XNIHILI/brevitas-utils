from typing import Any, Dict, Tuple

import copy

from torch.fx.experimental.optimization import (fuse_conv_bn_eval,
                                                matches_module_pattern)

import torch.fx as fx
import torch.nn as nn

DROPOUT_MODULES = (nn.Dropout, nn.Dropout1d, nn.Dropout2d, nn.Dropout3d)


def replace_node_module(top_level_module: nn.Module, name: str,
                        new_module: nn.Module):
    # Example name: "tcn.0.0.temp_layer1.4"

    # Split string into list of strings
    name_list = name.split(".")

    for name in name_list[:-1]:
        top_level_module = getattr(top_level_module, name)

    setattr(top_level_module, name_list[-1], new_module)


def remove_dropout(model: nn.Module, inplace=False) -> nn.Module:
    if not inplace:
        model = copy.deepcopy(model)

    for name, module in model.named_modules():
        if isinstance(module, DROPOUT_MODULES):
            replace_node_module(model, name, nn.Identity())

    return model


def fold_conv_bn(model: nn.Module, inplace=False) -> nn.Module:
    """Fuses convolution/BN layers for inference purposes (or, in this case, quantization purposes).

    Based on: https://github.com/pytorch/pytorch/blob/40cbf342d3c000712da92cfafeaca651b3e0bd3e/torch/fx/experimental/optimization.py#L50

    Will deepcopy your model by default, but can modify the model inplace as well.
    """

    patterns = [(nn.Conv1d, nn.BatchNorm1d), (nn.Conv2d, nn.BatchNorm2d),
                (nn.Conv3d, nn.BatchNorm3d)]
    
    if not inplace:
        model = copy.deepcopy(model)

    fx_model = fx.symbolic_trace(model)
    modules = dict(fx_model.named_modules())
    new_graph = copy.deepcopy(fx_model.graph)

    for pattern in patterns:
        for node in new_graph.nodes:
            if matches_module_pattern(pattern, node, modules):
                # Check if output of conv is used by other nodes
                if len(node.args[0].users) > 1:
                    continue

                conv = modules[node.args[0].target]
                bn = modules[node.target]
                fused_conv = fuse_conv_bn_eval(conv, bn)

                replace_node_module(model, node.args[0].target, fused_conv)
                replace_node_module(model, node.target, nn.Identity())

    return model
