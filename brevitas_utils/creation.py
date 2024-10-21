import copy

from typing import Dict, List, Optional, Tuple, Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import brevitas.nn as qnn
from brevitas import config
from brevitas.nn.mixin.parameter import WeightQuantType, BiasQuantType
from brevitas.nn.mixin.act import ActQuantType

from .brevitas_class_mapping import get_brevitas_class_by_name
from .layer_editing_utils import fold_conv_bn, remove_dropout
from .conversion import modules_to_qmodules
from .typing import OptionalBatchTransform
from .calibration import calibrate_model


# For reference, see: https://xilinx.github.io/brevitas/tutorials/tvmcon2021.html#Inheriting-from-a-quantizer
def create_quantizer(base_classes: List, kwargs: Dict[str, Any] = None):
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


class QuantNet(nn.Module):
    def __init__(self, net: nn.Module,
                 weight_quant: Optional[WeightQuantType] = None,
                 act_quant: Optional[ActQuantType] = None,
                 bias_quant: Optional[BiasQuantType] = None,
                 in_quant: Optional[ActQuantType] = None,
                 out_quant: Optional[ActQuantType] = None,
                 skip_modules: Optional[List[type[nn.Module]]] = None):
        super(QuantNet, self).__init__()

        if in_quant is not None:
            self.in_quant = qnn.QuantIdentity(act_quant=in_quant, return_quant_tensor=True)

        self.quant_net = modules_to_qmodules(net, weight_quant, act_quant, bias_quant, skip_modules).train()

        if out_quant is not None:
            self.out_quant = qnn.QuantIdentity(act_quant=out_quant, return_quant_tensor=False)

    def forward(self, x):
        if hasattr(self, 'in_quant'):
            x = self.in_quant(x)

        x = self.quant_net(x)

        if hasattr(self, 'out_quant'):
            x = self.out_quant(x)

        return x


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
                           skip_modules: Optional[List[type[nn.Module]]] = None):
    """Create a quantization-aware training model, ready for training. At minimum, only the weights should be quantized.

    For more details on how a custom quantizer is created, see:  (see for more details: https://xilinx.github.io/brevitas/tutorials/tvmcon2021.html#Inheriting-from-a-quantizer

    Args:
        model (nn.Module): Model to quantize.
        weight_quant (WeightQuantType, optional): Weight quantizater. Defaults to None.
        act_quant (ActQuantType, optional): Activation quantizater. Defaults to None.
        bias_quant (BiasQuantType, optional): Bias quantizater. Defaults to None.
        in_quant (ActQuantType, optional): Input quantizater. Defaults to None.
        out_quant (ActQuantType, optional): Output quantizater. Defaults to None.
        load_float_weights_into_model (bool, optional): Whether or not to reuse the weights from the floating point model. Defaults to True.
        remove_dropout_layers (bool, optional): Whether or not to remove dropout layers. Defaults to True.
        fold_batch_norm_layers (bool, optional): Whether or not to fold batch norm layers. Defaults to True.
        calibration_setup (Tuple[DataLoader, torch.device, OptionalBatchTransform, Optional[int]]], optional): Dataloader, device, batch transform and max calibration batches to be used for calibration before training. See [here](https://xilinx.github.io/brevitas/tutorials/tvmcon2021.html#Calibration-based-post-training-quantization) for more information. Defaults to None.
        apply_bias_correction (bool, optional): Whether or not to apply bias correction. Defaults to False.
        apply_norm_correction (bool, optional): Whether or not to apply norm correction. Defaults to False.
        skip_modules (List[type[nn.Module]]], optional): Torch modules that should not be quantized. Defaults to None.
    """

    folded_net = prepare_model_for_quant(model, remove_dropout_layers, fold_batch_norm_layers)
    quant_net = QuantNet(folded_net, weight_quant, act_quant, bias_quant, in_quant, out_quant, skip_modules)

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

    return quant_net.to('cpu')
