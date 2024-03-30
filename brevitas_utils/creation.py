import copy

from typing import Dict, List, Optional, NamedTuple, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import brevitas.nn as qnn
from brevitas import config
from brevitas.inject import ExtendedInjector

from .brevitas_class_mapping import get_brevitas_class_by_name
from .layer_editing_utils import fold_conv_bn, remove_dropout
from .convert_to_qmodules import modules_to_qmodules
from .typing import OptionalBatchTransform
from .calibration import calibrate_model


class QuantConfig(NamedTuple):
    base_classes: List[str]
    kwargs: Dict


# For reference, see: https://xilinx.github.io/brevitas/tutorials/tvmcon2021.html#Inheriting-from-a-quantizer
def create_quant_class(base_classes: List, kwargs: Dict):
    base_classes = [
        get_brevitas_class_by_name(base_class) for base_class in base_classes
    ]

    return type('QuantBase', tuple(base_classes), kwargs)


def prepend_qinput(model: nn.Module,
                   in_quant: Optional[ExtendedInjector],
                   inplace=False):
    if not inplace:
        model = copy.deepcopy(model)

    if in_quant is None:
        return model

    quantize_input = qnn.QuantIdentity(act_quant=in_quant,
                                       return_quant_tensor=True)

    model = nn.Sequential(quantize_input, model)

    return model


def create_qat_ready_model(model: nn.Module,
                           weight_quant_cfg: QuantConfig,
                           act_quant_cfg: Optional[QuantConfig] = None,
                           bias_quant_cfg: Optional[QuantConfig] = None,
                           in_quant_cfg: Optional[QuantConfig] = None,
                           load_float_weights: bool = True,
                           remove_dropout_layers: bool = True,
                           fold_batch_norm_layers: bool = True,
                           calibration_setup: Optional[Tuple[DataLoader, torch.device, OptionalBatchTransform]] = None,
                           skip_modules: List[type[nn.Module]] = []):
    """Create a quantization-aware training model, ready for training. At minimum, only the weights should be quantized.

    For more details on how a custom quantizer is created, see:  (see for more details: https://xilinx.github.io/brevitas/tutorials/tvmcon2021.html#Inheriting-from-a-quantizer

    Args:
        model (nn.Module): Model to quantize.
        weight_quant_cfg (QuantConfig): Weight quantization configuration
        act_quant_cfg (Optional[QuantConfig]): Activation quantization configuration. Defaults to None.
        bias_quant_cfg (Optional[QuantConfig], optional): Bias quantization configuration. Defaults to None.
        in_quant_cfg (Optional[QuantConfig], optional): Input quantization configuration. Defaults to None.
        load_float_weights (bool, optional): Whether or not to reuse the weights from the floating point model. Defaults to True.
        remove_dropout_layers (bool, optional): Whether or not to remove dropout layers. Defaults to True.
        fold_batch_norm_layers (bool, optional): Whether or not to fold batch norm layers. Defaults to True.
        calibration_setup (Optional[Tuple[DataLoader, torch.device, OptionalBatchTransform]], optional): Dataloader, device and batch transform to be use for calibration before training. See [here](https://xilinx.github.io/brevitas/tutorials/tvmcon2021.html#Calibration-based-post-training-quantization) for more information. Defaults to None.
        skip_modules (List[type[nn.Module]], optional): Torch modules that should not be quantized. Defaults to [].
    """

    weight_quant, act_quant, bias_quant, in_quant = [create_quant_class(quant_cfg.base_classes, dict(quant_cfg.kwargs)) if quant_cfg else None for quant_cfg in [weight_quant_cfg, act_quant_cfg, bias_quant_cfg, in_quant_cfg]]

    eval_model = model.eval()

    if remove_dropout_layers == True:
        eval_model = remove_dropout(eval_model)

    # Support weight-only (float act, bias) and weight+act (float bias) quantization
    # If biase quant not provided, also make sure to not return a quant tensor from the activations

    folded_model = eval_model

    if fold_batch_norm_layers == True:
        folded_model = fold_conv_bn(folded_model)

    quant_model = prepend_qinput(modules_to_qmodules(folded_model, weight_quant, act_quant, bias_quant, skip_modules).train(), in_quant)

    # Taken from: https://xilinx.github.io/brevitas/tutorials/tvmcon2021.html#Retraining-from-floating-point
    if load_float_weights == True:
        config.IGNORE_MISSING_KEYS = True

        quant_model[1].load_state_dict(folded_model.state_dict())

    if calibration_setup != None:
        calibration_loader, device, batch_transform = calibration_setup

        quant_model = quant_model.to(device)
        quant_model = calibrate_model(quant_model, calibration_loader, device, batch_transform)

    return quant_model.to('cpu')
