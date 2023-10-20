import copy

from typing import Callable, Dict, List, Union, NamedTuple, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import brevitas.nn as qnn
from brevitas import config
from brevitas.graph.calibrate import bias_correction_mode, calibration_mode
from brevitas.inject import ExtendedInjector

from .brevitas_class_mapping import get_brevitas_class_by_name
from .layer_editing_utils import replace_node_module, fold_conv_bn, remove_dropout
from .convert_to_qmodules import modules_to_qmodules
from .typing import OptionalBatchTransform

from metalarena.common.loop import evaluate


class QuantConfig(NamedTuple):
    base_classes: List[str]
    kwargs: Dict


def create_quant_class(base_classes: List, kwargs: Dict):
    base_classes = [
        get_brevitas_class_by_name(base_class) for base_class in base_classes
    ]

    return type('QuantBase', tuple(base_classes), kwargs)


def prepend_qinput(model: nn.Module,
                   act_quant: ExtendedInjector,
                   inplace=False):
    if not inplace:
        model = copy.deepcopy(model)

    quantize_input = qnn.QuantIdentity(act_quant=act_quant,
                                       return_quant_tensor=True)

    model = nn.Sequential(quantize_input, model)

    return model


def create_qat_ready_model(model: nn.Module,
                           weight_quant_cfg: QuantConfig,
                           act_quant_cfg: QuantConfig,
                           bias_quant_cfg: Union[QuantConfig, None] = None,
                           from_float_weights: bool = True,
                           calibration_setup: Union[Tuple[DataLoader, torch.device, OptionalBatchTransform], None] = None,
                           skip_modules: List[type[nn.Module]] = []):
    weight_quant, act_quant, bias_quant = [create_quant_class(quant_cfg.base_classes, dict(quant_cfg.kwargs)) for quant_cfg in [weight_quant_cfg, act_quant_cfg, bias_quant_cfg]]

    folded_model = fold_conv_bn(remove_dropout(model.eval()))
    quant_model = prepend_qinput(modules_to_qmodules(folded_model, weight_quant, act_quant, bias_quant, skip_modules).train(), act_quant)

    if from_float_weights == True:
        config.IGNORE_MISSING_KEYS = True

        quant_model[1].load_state_dict(folded_model.state_dict())

    if calibration_setup != None:
        calibration_loader, device, batch_transform = calibration_setup

        quant_model = quant_model.to(device)
        quant_model = calibrate_model(quant_model, calibration_loader, device, batch_transform)

    return quant_model.to('cpu')
