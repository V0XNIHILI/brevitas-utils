from collections import OrderedDict

import torch
import torch.nn as nn

import brevitas.nn as qnn


def get_quant_weights_and_biases(quant_model: nn.Module, input_shape: tuple, as_numpy: bool = False):
    quant_model = quant_model.eval()

    layers = {}

    for name, module in quant_model.named_modules():
        if isinstance(module, qnn.QuantConv1d) or isinstance(module, qnn.QuantConv2d) or isinstance(module, qnn.QuantLinear):
            layers[name] = module

    activations = {}

    for name, module in quant_model.named_modules():
        if name.endswith('act_quant'):
            scale = module.scale().cpu().detach()
            zero_point = module.zero_point().cpu().detach()

            if as_numpy:
                scale = scale.numpy()
                zero_point = zero_point.numpy()

            if len(scale.shape) == 0:
                scale = scale.item()

            if len(zero_point.shape) == 0:
                zero_point = zero_point.item()

            activations[name] = {
                'scale': scale,
                'zero_point': zero_point
            }

    # Make sure bias can be accessed. Enable this flag on all modules,
    # independently of whether they have a bias or not.
    for name, module in layers.items():
        module.cache_inference_quant_bias = True

    quant_model.cpu()(torch.randn(*input_shape))

    parameters = {}

    # TODO: add input and output quant scale and zero-point

    for name, layer in layers.items():
        quant_weight = layer.quant_weight().value.cpu().detach()
        quant_bias = layer.quant_bias()

        if quant_bias is not None:
            quant_bias = quant_bias.value.cpu().detach()

            if as_numpy:
                quant_bias = quant_bias.numpy()

        scale = layer.quant_weight().scale.cpu().detach()

        if as_numpy:
            quant_weight = quant_weight.numpy()
            scale = scale.numpy()

        if len(scale.shape) == 0:
            scale = scale.item()

        parameters[name] = {
            'weight': quant_weight,
            'bias': quant_bias,
            'scale': scale
        }

    return OrderedDict(parameters.items()), OrderedDict(activations.items())
