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


def quantize_io(model: nn.Module,
                in_quant: Optional[ExtendedInjector],
                out_quant: Optional[ExtendedInjector],
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


def create_qat_ready_model(model: nn.Module,
                           weight_quant_cfg: QuantConfig,
                           act_quant_cfg: Optional[QuantConfig] = None,
                           bias_quant_cfg: Optional[QuantConfig] = None,
                           in_quant_cfg: Optional[QuantConfig] = None,
                           out_quant_cfg: Optional[QuantConfig] = None,
                           load_float_weights_into_model: bool = True,
                           remove_dropout_layers: bool = True,
                           fold_batch_norm_layers: bool = True,
                           calibration_setup: Optional[Tuple[DataLoader, torch.device, OptionalBatchTransform]] = None,
                           apply_bias_correction: bool = False,
                           skip_modules: List[type[nn.Module]] = []):
    """Create a quantization-aware training model, ready for training. At minimum, only the weights should be quantized.

    For more details on how a custom quantizer is created, see:  (see for more details: https://xilinx.github.io/brevitas/tutorials/tvmcon2021.html#Inheriting-from-a-quantizer

    Args:
        model (nn.Module): Model to quantize.
        weight_quant_cfg (QuantConfig): Weight quantization configuration
        act_quant_cfg (Optional[QuantConfig]): Activation quantization configuration. Defaults to None.
        bias_quant_cfg (Optional[QuantConfig], optional): Bias quantization configuration. Defaults to None.
        in_quant_cfg (Optional[QuantConfig], optional): Input quantization configuration. Defaults to None.
        out_quant_cfg (Optional[QuantConfig], optional): Output quantization configuration. Defaults to None.
        load_float_weights_into_model (bool, optional): Whether or not to reuse the weights from the floating point model. Defaults to True.
        remove_dropout_layers (bool, optional): Whether or not to remove dropout layers. Defaults to True.
        fold_batch_norm_layers (bool, optional): Whether or not to fold batch norm layers. Defaults to True.
        calibration_setup (Optional[Tuple[DataLoader, torch.device, OptionalBatchTransform]], optional): Dataloader, device and batch transform to be use for calibration before training. See [here](https://xilinx.github.io/brevitas/tutorials/tvmcon2021.html#Calibration-based-post-training-quantization) for more information. Defaults to None.
        apply_bias_correction (bool, optional): Whether or not to apply bias correction. Defaults to False.
        skip_modules (List[type[nn.Module]], optional): Torch modules that should not be quantized. Defaults to [].
    """

    weight_quant, act_quant, bias_quant, in_quant, out_quant = [create_quant_class(quant_cfg.base_classes, dict(quant_cfg.kwargs)) if quant_cfg else None for quant_cfg in [weight_quant_cfg, act_quant_cfg, bias_quant_cfg, in_quant_cfg, out_quant_cfg]]

    eval_model = model.eval()

    if remove_dropout_layers == True:
        eval_model = remove_dropout(eval_model)

    # Support weight-only (float act, bias) and weight+act (float bias) quantization
    # If bias quant is not provided, also make sure to not return a quant tensor from the activations

    folded_model = eval_model

    if fold_batch_norm_layers == True:
        folded_model = fold_conv_bn(folded_model)

    quant_model = quantize_io(modules_to_qmodules(folded_model, weight_quant, act_quant, bias_quant, skip_modules).train(), in_quant, out_quant)

    # Taken from: https://xilinx.github.io/brevitas/tutorials/tvmcon2021.html#Retraining-from-floating-point
    if load_float_weights_into_model == True:
        if isinstance(quant_model, nn.Sequential):
            quant_model_for_float_loading = quant_model[0] if in_quant_cfg is None else quant_model[1]
        else:
            quant_model_for_float_loading = quant_model

        load_float_weights(quant_model_for_float_loading, folded_model)

    if apply_bias_correction == True and calibration_setup == None:
        raise ValueError("Bias correction can only be applied if calibration is also performed.")

    if calibration_setup != None:
        calibration_loader, device, batch_transform = calibration_setup

        quant_model = quant_model.to(device)
        quant_model = calibrate_model(quant_model, calibration_loader, device, batch_transform, apply_bias_correction)

    return quant_model.to('cpu')
