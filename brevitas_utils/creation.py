import copy

from typing import Dict, List, Optional, Tuple, Any

from collections import OrderedDict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.fx.experimental.optimization import remove_dropout, fuse as fold_conv_bn

import brevitas.nn as qnn
from brevitas import config
from brevitas.nn.mixin.parameter import WeightQuantType, BiasQuantType
from brevitas.nn.mixin.act import ActQuantType

from .brevitas_class_mapping import get_brevitas_class_by_name
from .conversion import modules_to_qmodules, base_qact_mapping, base_qmodule_mapping
from .typing import OptionalBatchTransform
from .calibration import calibrate_model


# For reference, see: https://xilinx.github.io/brevitas/tutorials/tvmcon2021.html#Inheriting-from-a-quantizer
def create_quantizer(base_classes: List[str], kwargs: Dict[str, Any] = None):
    """Function to dynamically create a quantizer class from a list of base class names and optional kwargs. 

    Args:
        base_classes (List[str]): Earlier base classes override properties of later ones.
        kwargs (Dict[str, Any], optional): _description_. Defaults to None.

    Raises:
        TypeError: _description_

    Returns:
        _type_: _description_
    """

    if not isinstance(base_classes, list):
        raise TypeError("base_classes must be a list of strings")

    base_classes = [get_brevitas_class_by_name(base_class) for base_class in base_classes]

    return type('QuantBase', tuple(base_classes), {} if kwargs is None else kwargs)


def quantize_io(model: nn.Module,
                in_quant: Optional[ActQuantType],
                out_quant: Optional[ActQuantType],
                inplace=False):
    if not inplace:
        model = copy.deepcopy(model)

    modules = []

    if in_quant is not None:
        modules.append(qnn.QuantIdentity(act_quant=in_quant,
                                        return_quant_tensor=True))

    modules.append(model)

    if out_quant is not None:
        modules.append(qnn.QuantIdentity(act_quant=out_quant,
                                        return_quant_tensor=False))

    if len(modules) == 1:
        return model

    return nn.Sequential(*modules)


def load_float_weights(quant_model: nn.Module, float_model: nn.Module):
    config.IGNORE_MISSING_KEYS = True
    quant_model.load_state_dict(float_model.state_dict())
    config.IGNORE_MISSING_KEYS = False

    return quant_model


def prepare_model_for_quant(model: nn.Module, remove_dropout_layers: bool = True, fold_batch_norm_layers: bool = True):
    eval_net = model.eval()

    if remove_dropout_layers == True:
        eval_net = remove_dropout(eval_net)

    # TODO: Support weight-only (float act, bias) and weight+act (float bias) quantization
    # If bias quant is not provided, also make sure to not return a quant tensor from the activations

    folded_net = eval_net

    if fold_batch_norm_layers == True:
        folded_net = fold_conv_bn(folded_net)

    return folded_net


class QuantNet(nn.Sequential):
    def __init__(self, net: nn.Module,
                 weight_quant: Optional[WeightQuantType] = None,
                 act_quant: Optional[ActQuantType] = None,
                 bias_quant: Optional[BiasQuantType] = None,
                 in_quant: Optional[ActQuantType] = None,
                 out_quant: Optional[ActQuantType] = None,
                 skip_modules: Optional[List[type[nn.Module]]] = None,
                 return_quant_out_tensor: Optional[bool] = None):
        modules = OrderedDict()

        if in_quant is not None:
            modules.update({"in_quant": qnn.QuantIdentity(act_quant=in_quant, return_quant_tensor=True)})

        modules.update({"quant_net": modules_to_qmodules(net, weight_quant, act_quant, bias_quant, skip_modules).train()})

        if out_quant is not None:
            modules.update({"out_quant": qnn.QuantIdentity(act_quant=out_quant, return_quant_tensor=bool(return_quant_out_tensor))})
        elif return_quant_out_tensor is not None:
            raise ValueError("return_quant_out_tensor can only be set if out_quant is also provided.")
        
        super(QuantNet, self).__init__(modules)


def create_qat_ready_model(model: nn.Module,
                           weight_quant: Optional[WeightQuantType] = None,
                           act_quant: Optional[ActQuantType] = None,
                           bias_quant: Optional[BiasQuantType] = None,
                           in_quant: Optional[ActQuantType] = None,
                           out_quant: Optional[ActQuantType] = None,
                           load_float_weights_into_model: bool = True,
                           remove_dropout_layers: bool = True,
                           fold_batch_norm_layers: bool = True,
                           calibration_setup: Optional[Tuple[DataLoader, torch.device, OptionalBatchTransform, Optional[int]]] = None,
                           apply_bias_correction: bool = False,
                           apply_norm_correction: bool = False,
                           skip_modules: Optional[List[type[nn.Module]]] = None,
                           return_quant_out_tensor: Optional[bool] = None):
    """Create a quantization-aware training model, ready for training. At minimum, only the weights should be quantized.

    For more details on how a custom quantizer is created, see:  (see for more details: https://xilinx.github.io/brevitas/tutorials/tvmcon2021.html#Inheriting-from-a-quantizer

    Note that if calibration is performed, the value of collect_stats_steps will be ignored.

    Args:
        model (nn.Module): Model to quantize.
        weight_quant (WeightQuantType, optional): Weight quantizer. Defaults to None.
        act_quant (ActQuantType, optional): Activation quantizer. Defaults to None.
        bias_quant (BiasQuantType, optional): Bias quantizer. Defaults to None.
        in_quant (ActQuantType, optional): Input quantizer. Defaults to None.
        out_quant (ActQuantType, optional): Output quantizer. Defaults to None.
        load_float_weights_into_model (bool, optional): Whether or not to reuse the weights from the floating point model. Defaults to True.
        remove_dropout_layers (bool, optional): Whether or not to remove dropout layers. Defaults to True.
        fold_batch_norm_layers (bool, optional): Whether or not to fold batch norm layers. Defaults to True.
        calibration_setup (Tuple[DataLoader, torch.device, OptionalBatchTransform, Optional[int]]], optional): Dataloader, device, batch transform and max calibration batches to be used for calibration before training. See [here](https://xilinx.github.io/brevitas/tutorials/tvmcon2021.html#Calibration-based-post-training-quantization) for more information. Defaults to None.
        apply_bias_correction (bool, optional): Whether or not to apply bias correction. Defaults to False.
        apply_norm_correction (bool, optional): Whether or not to apply norm correction. Defaults to False.
        skip_modules (List[type[nn.Module]]], optional): Torch modules that should not be quantized. Defaults to None.
        return_quant_out_tensor (bool, optional): Whether or not to return a quant tensor from the output quantizer (if provided). Defaults to None.
    """

    # If only a single layer has to be quantized, tracing it effectively empties the model (and thus removes the layer).
    if not type(model) in base_qact_mapping | base_qmodule_mapping:
        folded_net = prepare_model_for_quant(model, remove_dropout_layers, fold_batch_norm_layers)
    else:
        # Since we first wrap the model in order to make the quantization of the model work
        # and then unwrap only the quantized part at the end, we do not support input and output quantization here.
        assert in_quant is None and out_quant is None, "Input and output quantization is only supported for full models."
        folded_net = nn.Sequential(OrderedDict([("___brevitas_utils_name_wrapper___", model)]))

    quant_net = QuantNet(folded_net, weight_quant, act_quant, bias_quant, in_quant, out_quant, skip_modules, return_quant_out_tensor)

    # Taken from: https://xilinx.github.io/brevitas/tutorials/tvmcon2021.html#Retraining-from-floating-point
    if load_float_weights_into_model == True:
        load_float_weights(quant_net.quant_net, folded_net)

    if apply_bias_correction == True and calibration_setup == None:
        raise ValueError("Bias correction can only be applied if calibration is also performed.")

    if calibration_setup != None:
        calibration_loader, device, batch_transform, max_calibration_batches = calibration_setup

        quant_net = calibrate_model(quant_net.to(device),
                                      calibration_loader,
                                      device,
                                      batch_transform,
                                      max_calibration_batches,
                                      apply_bias_correction,
                                      apply_norm_correction)
        
    if hasattr(quant_net.quant_net, '___brevitas_utils_name_wrapper___'):
        quant_net = quant_net.quant_net.___brevitas_utils_name_wrapper___

    return quant_net.to('cpu')
