from collections import OrderedDict

import torch
import torch.nn as nn

import brevitas.nn as qnn


def is_supported_weight_layer(module: nn.Module):
    return isinstance(module, qnn.QuantConv1d) or isinstance(module, qnn.QuantConv2d) or isinstance(module, qnn.QuantLinear)


def get_quant_state_dict(quant_model: nn.Module, input_shape: tuple, as_numpy: bool = False):
    """Retrieves the quantized state dictionary of a quantized model.

    Args:
        quant_model (nn.Module): The quantized model.
        input_shape (tuple): The shape of the input tensor.
        as_numpy (bool, optional): If True, converts the quantized parameters from Torch tensors to Numpy arrays. Defaults to False.

    Returns:
        OrderedDict: The quantization state dictionary containing the activations, weight, and bias parameters of the quantized model.
    """
    
    quant_model = quant_model.eval()

    # Make sure bias can be accessed. Enable this flag on all modules,
    # independently of whether they have a bias or not.
    for name, module in quant_model.named_modules():
        if is_supported_weight_layer(module):
            module.cache_inference_quant_bias = True

    quant_model.cpu()(torch.randn(*input_shape))

    quant_state_dict = {}

    for name, module in quant_model.named_modules():
        # TODO: add input and output quant scale and zero-point
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

            quant_state_dict[name] = {
                'scale': scale,
                'zero_point': zero_point
            }
        elif is_supported_weight_layer(module):
            quant_weight_full = module.quant_weight()
            quant_weight = quant_weight_full.value.cpu().detach()
            quant_bias_full = module.quant_bias()

            scale = quant_weight_full.scale.cpu().detach()        

            if as_numpy:
                quant_weight = quant_weight.numpy()
                scale = scale.numpy()

            if len(scale.shape) == 0:
                scale = scale.item()

            quant_state_dict[name] = {
                'weight': {
                    'value': quant_weight,
                    'scale': scale,
                    'bit_width': int(quant_weight_full.bit_width.item()),
                    'signed': quant_weight_full.signed_t.item(),
                    'zero_point': quant_weight_full.zero_point.item()
                }
            }

            if quant_bias_full is not None:
                quant_bias = quant_bias_full.value.cpu().detach()
                scale = quant_bias_full.scale.cpu().detach()

                if as_numpy:
                    quant_bias = quant_bias.numpy()
                    scale = scale.numpy()

                if len(scale.shape) <= 1:
                    scale = scale.item()

                quant_state_dict[name]['bias'] = {
                    'value': quant_bias,
                    'scale': scale,
                    'bit_width': int(quant_bias_full.bit_width.item()),
                    'signed': quant_bias_full.signed_t.item(),
                    'zero_point': quant_bias_full.zero_point.item()
                }

    return OrderedDict(quant_state_dict.items())
