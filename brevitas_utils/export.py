from collections import OrderedDict
import pickle
from typing import Union
from pathlib import Path

import torch
import torch.nn as nn

from brevitas.quant_tensor import QuantTensor


def is_supported_weight_layer(module: nn.Module):
    import brevitas.nn as qnn

    return isinstance(module, qnn.QuantConv1d) or isinstance(module, qnn.QuantConv2d) or isinstance(module, qnn.QuantLinear) # or isinstance(module, qnn.QuantLSTM)


def is_gate_params(module: nn.Module):
    import brevitas.nn.quant_rnn as qnnr

    return isinstance(module, qnnr.GateParams)


def save_quant_state_dict(quant_state_dict: OrderedDict, path: str):
    """Saves the quantization state dictionary to a file.

    Args:
        quant_state_dict (OrderedDict): The quantization state dictionary.
        path (str): The path to save the quantization state dictionary.
    """

    with open(path, 'wb') as handle:
        pickle.dump(quant_state_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_quant_state_dict(path: Union[str, Path]):
    """Loads the quantization state dictionary from a file.

    Args:
        path (Union[str, Path]): The path to load the quantization state dictionary.

    Returns:
        OrderedDict: The quantization state dictionary.
    """

    with open(path, 'rb') as handle:
        quant_state_dict = pickle.load(handle)

    return quant_state_dict


def store_quant_tensor(quant_tensor_full: QuantTensor, param_type: str, quant_state_dict: dict, name: str, as_numpy: bool):
    """Stores the quantized tensor and its quantization parameters in the quantization state dictionary.

    Args:
        quant_tensor_full (QuantTensor): The full quantized tensor containing the quantized values and quantization parameters.
        param_type (str): The type of the parameter (e.g., 'weight' or 'bias').
        quant_state_dict (dict): The quantization state dictionary to store the quantized tensor and its parameters.
        name (str): The name of the module to which the quantized tensor belongs.
        as_numpy (bool): If True, converts the quantized tensor and its parameters from Torch tensors to Numpy arrays.

    Returns:
        dict: The updated quantization state dictionary with the stored quantized tensor and its parameters.
    """

    quant_tensor = quant_tensor_full.value.cpu().detach()
    scale = quant_tensor_full.scale.cpu().detach()

    if as_numpy:
        quant_tensor = quant_tensor.numpy()
        scale = scale.numpy()

    if len(scale.shape) <= 1:
        scale = scale.item()

    if name not in quant_state_dict:
        quant_state_dict[name] = {}

    quant_state_dict[name][param_type] = {
        'value': quant_tensor,
        'scale': scale,
        'bit_width': int(quant_tensor_full.bit_width.item()),
        'signed': quant_tensor_full.signed_t.item(),
        'zero_point': quant_tensor_full.zero_point.item()
    }

    return quant_state_dict


def get_quant_state_dict(quant_model: nn.Module, input_shape_or_quant_tensor: Union[tuple[int, ...], QuantTensor], as_numpy: bool = False):
    """Retrieves the quantized state dictionary of a quantized model.

    Args:
        quant_model (nn.Module): The quantized model.
        input_shape_or_quant_tensor (Union[tuple, QuantTensor]): The shape of the input tensor or a QuantTensor to be used for a forward pass.
        as_numpy (bool, optional): If True, converts the quantized parameters from Torch tensors to Numpy arrays. Defaults to False.

    Returns:
        OrderedDict: The quantization state dictionary containing the activations, weight, and bias parameters of the quantized model.
    """

    quant_model = quant_model.eval()

    # Make sure bias can be accessed. Enable this flag on all modules,
    # independently of whether they have a bias or not.
    for name, module in quant_model.named_modules():
        if is_supported_weight_layer(module) or is_gate_params(module):
            module.cache_inference_quant_bias = True

    if isinstance(input_shape_or_quant_tensor, QuantTensor):
        input_tensor = input_shape_or_quant_tensor
    else:
        input_tensor = torch.randn(*input_shape_or_quant_tensor)

    quant_model.cpu()(input_tensor)

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
            if quant_weight_full is not None:
                quant_state_dict = store_quant_tensor(quant_weight_full, 'weight', quant_state_dict, name, as_numpy)

            quant_bias_full = module.quant_bias()
            if quant_bias_full is not None:
                quant_state_dict = store_quant_tensor(quant_bias_full, 'bias', quant_state_dict, name, as_numpy)
                
        if is_gate_params(module):
            # TODO: Brevitas does not store _cached_bias when cache_inference_quant_bias is True for GateParams.  
            # quant_bias_full = module.quant_bias()
            # if quant_bias_full is not None:
            #     quant_state_dict = store_quant_tensor(quant_bias_full, 'bias', quant_state_dict, name, as_numpy)

            quant_input_weight_full = module.input_weight.quant_weight()
            if quant_input_weight_full is not None:
                quant_state_dict = store_quant_tensor(quant_input_weight_full, 'input_weight', quant_state_dict, name, as_numpy)

            quant_hidden_weight_full = module.hidden_weight.quant_weight()
            if quant_hidden_weight_full is not None:
                quant_state_dict = store_quant_tensor(quant_hidden_weight_full, 'hidden_weight', quant_state_dict, name, as_numpy)

    return OrderedDict(quant_state_dict.items())