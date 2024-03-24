import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from brevitas.graph.calibrate import bias_correction_mode, calibration_mode

from tqdm import tqdm

from .typing import OptionalBatchTransform

@torch.no_grad()
def calibrate_model(quant_model: nn.Module,
                    calibration_loader: DataLoader,
                    device: torch.device,
                    batch_transform: OptionalBatchTransform = None):
    # Based on (but modified): https://xilinx.github.io/brevitas/tutorials/tvmcon2021.html#Calibration-based-post-training-quantization

    # Put the model in calibration mode to collect statistics
    # Quantization is automatically disabled
    # during the calibration, and re-enabled at the end
    # Based on: https://github.com/huggingface/optimum-amd/blob/ca32e8e4f7f0c8321d0380304697c08d60c6edf9/optimum/amd/brevitas/quantizer.py#L305
    with calibration_mode(quant_model):
        with torch.no_grad():
            for (X, _) in tqdm(calibration_loader, desc="Calibrating"):
                X = X.to(device)
                if batch_transform is not None:
                    X = batch_transform(X)
                quant_model(X)

    # Apply bias correction
    # Based on: https://github.com/huggingface/optimum-amd/blob/ca32e8e4f7f0c8321d0380304697c08d60c6edf9/optimum/amd/brevitas/quantizer.py#L313
    with bias_correction_mode(quant_model):
        for (X, _) in tqdm(calibration_loader, desc="Correcting biases"):
            X = X.to(device)
            if batch_transform is not None:
                X = batch_transform(X)
            quant_model(X)
            
    return quant_model
