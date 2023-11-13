import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from brevitas.graph.calibrate import bias_correction_mode, calibration_mode

from torch_mate.flow.main import evaluate

from .typing import OptionalBatchTransform

def calibrate_model(quant_model: nn.Module,
                    calibration_loader: DataLoader,
                    device: torch.device,
                    batch_transform: OptionalBatchTransform = None):
    # Taken from: https://xilinx.github.io/brevitas/tutorials/tvmcon2021.html#Calibration-based-post-training-quantization

    with torch.no_grad():
        zero_loss_fn = lambda _0, _1: torch.tensor([0.0])

        # Put the model in calibration mode to collect statistics
        # Quantization is automatically disabled
        # during the calibration, and re-enabled at the end
        with calibration_mode(quant_model):
            evaluate(quant_model, zero_loss_fn, calibration_loader, device,
                     batch_transform, None)

        # Apply bias correction
        with bias_correction_mode(quant_model):
            evaluate(quant_model, zero_loss_fn, calibration_loader, device,
                     batch_transform, None)

    return quant_model
