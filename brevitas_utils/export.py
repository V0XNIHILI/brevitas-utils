import torch
import torch.nn as nn

import brevitas.nn as qnn

def get_quant_weights_and_biases(quant_model: nn.Module, input_shape: tuple):
    layers = {}

    for name, module in quant_model.named_modules():
        if isinstance(module, qnn.QuantConv1d) or isinstance(module, qnn.QuantConv2d) or isinstance(module, qnn.QuantLinear):
            layers[name] = module

    activations = {}

    for name, module in quant_model.named_modules():
        if name.endswith('act_quant'):
            activations[name] = {
                'scale': module.scale().cpu().detach(),
                'zero_point': module.zero_point().cpu().detach()
            }

    # Make sure bias can be accessed
    for name, module in layers.items():
        module.cache_inference_quant_bias = True

    quant_model.eval()
    quant_model.cpu()(torch.randn(*input_shape))

    parameters = {}

    for name, layer in layers.items():
        quant_weight = layer.quant_weight().value.cpu().detach()
        quant_bias = layer.quant_bias()

        if quant_bias is not None:
            quant_bias = quant_bias.value.cpu().detach()

        scale = layer.quant_weight().scale.cpu().detach()

        parameters[name] = {
            'weight': quant_weight,
            'bias': quant_bias,
            'scale': scale
        }

    return parameters, activations
