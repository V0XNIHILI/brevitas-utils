from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from brevitas.graph.calibrate import bias_correction_mode, calibration_mode, norm_correction_mode

from tqdm import tqdm

from .typing import OptionalBatchTransform


def forward_all(quant_model: nn.Module, calibration_loader: DataLoader, device: torch.device, batch_transform: OptionalBatchTransform = None, desc: Optional[str] = None):
    for (X, _) in tqdm(calibration_loader, desc=desc):
        X = X.to(device)
        if batch_transform is not None:
            X = batch_transform(X)
        quant_model(X)
            

@torch.no_grad()
def calibrate_model(quant_model: nn.Module,
                    calibration_loader: DataLoader,
                    device: torch.device,
                    batch_transform: OptionalBatchTransform = None,
                    apply_bias_correction: bool = False,
                    apply_norm_correction: bool = False) -> nn.Module:
    # Based on (but modified): https://xilinx.github.io/brevitas/tutorials/tvmcon2021.html#Calibration-based-post-training-quantization

    # Put the model in calibration mode to collect statistics
    # Quantization is automatically disabled
    # during the calibration, and re-enabled at the end
    # Based on: https://github.com/huggingface/optimum-amd/blob/ca32e8e4f7f0c8321d0380304697c08d60c6edf9/optimum/amd/brevitas/quantizer.py#L305
    with calibration_mode(quant_model), torch.no_grad():
        forward_all(quant_model, calibration_loader, device, batch_transform, "Calibrating")

    if apply_bias_correction:
        # Apply bias correction
        # Based on: https://github.com/huggingface/optimum-amd/blob/ca32e8e4f7f0c8321d0380304697c08d60c6edf9/optimum/amd/brevitas/quantizer.py#L313
        with bias_correction_mode(quant_model):
            forward_all(quant_model, calibration_loader, device, batch_transform, "Correcting biases")

    if apply_norm_correction:
        # Apply (batch) norm correction
        with norm_correction_mode(quant_model):
            forward_all(quant_model, calibration_loader, device, batch_transform, "Correcting norms")

    return quant_model
