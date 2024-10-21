from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from brevitas.graph.calibrate import bias_correction_mode, calibration_mode, norm_correction_mode
from brevitas_examples.llm.llm_quant.prepare_for_quantize import add_zero_bias_to_linear

from tqdm import tqdm

from .typing import OptionalBatchTransform, NestedTupleOfTensors


def nested_tuple_to_device(item: NestedTupleOfTensors, device: torch.device, non_blocking=False):
    """Move a (nested) tuple of tensors to the device.

    Args:
        item (NestedTupleOfTensors): (Nested) tuple of tensors to move.
        device (torch.device): Device to move the tensors to.
        non_blocking (bool, optional): If True and this copy is between CPU and GPU, the copy may occur asynchronously with respect to the host. For other cases, this argument has no effect. Defaults to False.

    Returns:
        NestedTupleOfTensors: (Nested) tuple of tensors moved to the device.
    """

    if type(item) is tuple or type(item) is list:
        return tuple(nested_tuple_to_device(e, device, non_blocking) for e in item)
    else:
        return item.to(device, non_blocking=non_blocking)


def forward_all(quant_model: nn.Module, calibration_loader: DataLoader, device: torch.device, batch_transform: OptionalBatchTransform = None, max_calibration_batches: Optional[int] = None, desc: Optional[str] = None):
    for i, (X, _) in tqdm(enumerate(calibration_loader), desc=desc, total=len(calibration_loader) if max_calibration_batches is None else max_calibration_batches):
        X = nested_tuple_to_device(X, device)
        if batch_transform is not None:
            X = batch_transform(X)
        quant_model(X)

        if max_calibration_batches is not None:
            if i + 1 >= max_calibration_batches:
                break
            

@torch.no_grad()
def calibrate_model(quant_model: nn.Module,
                    calibration_loader: DataLoader,
                    device: torch.device,
                    batch_transform: OptionalBatchTransform = None,
                    max_calibration_batches: Optional[int] = None,
                    apply_bias_correction: bool = False,
                    apply_norm_correction: bool = False) -> nn.Module:
    # Based on (but modified): https://xilinx.github.io/brevitas/tutorials/tvmcon2021.html#Calibration-based-post-training-quantization

    if apply_bias_correction:
        quant_model = add_zero_bias_to_linear(quant_model)

    # Put the model in calibration mode to collect statistics
    # Quantization is automatically disabled
    # during the calibration, and re-enabled at the end
    # Based on: https://github.com/huggingface/optimum-amd/blob/ca32e8e4f7f0c8321d0380304697c08d60c6edf9/optimum/amd/brevitas/quantizer.py#L305
    with calibration_mode(quant_model), torch.no_grad():
        forward_all(quant_model, calibration_loader, device, batch_transform, max_calibration_batches, "Calibrating")

    if apply_bias_correction:
        # Apply bias correction
        # Based on: https://github.com/huggingface/optimum-amd/blob/ca32e8e4f7f0c8321d0380304697c08d60c6edf9/optimum/amd/brevitas/quantizer.py#L313
        with bias_correction_mode(quant_model):
            forward_all(quant_model, calibration_loader, device, batch_transform, max_calibration_batches, "Correcting biases")

    if apply_norm_correction:
        # Apply (batch) norm correction
        with norm_correction_mode(quant_model):
            forward_all(quant_model, calibration_loader, device, batch_transform, max_calibration_batches, "Correcting norms")

    return quant_model
