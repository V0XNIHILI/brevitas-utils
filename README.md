# Brevitas utils (`brevitas_utils`)

Library with utilities for [Brevitas](https://github.com/Xilinx/brevitas/).

Features:

- 1-line conversion of a floating point PyTorch model to a model that can be used for quantization-aware training (QAT)
  - Automatic batch normalization folding
  - Automatic removal of dropout layers
- 1-line extraction of quantized weights and biases from a QAT model
- Power-of-two weight quantization ([`Int8WeightPerTensorPowerOfTwo.py`](brevitas_utils/custom_quantizers/Int8WeightPerTensorPowerOfTwo.py))

## Installation

```bash
git clone git@github.com:V0XNIHILI/brevitas-utils.git
cd brevitas-utils
pip install -e .
```

## Usage

Please see a mini-tutorial below:

```python
import torch.nn as nn

from brevitas_utils import create_qat_ready_model, get_quant_state_dict, QuantConfig

model = nn.Sequential(
    nn.Linear(10, 20),
    nn.ReLU(),
    nn.Linear(20, 10)
)

# Define quantization configurations (see for more details: https://xilinx.github.io/brevitas/tutorials/tvmcon2021.html#Inheriting-from-a-quantizer)
weight_quant_cfg = QuantConfig(base_classes=["Int8WeightPerTensorPowerOfTwo"], kwargs={"bit_width": 4, "narrow_range": False})
act_quant_cfg = QuantConfig(base_classes=["ShiftedParamFromPercentileUintQuant"], kwargs={"bit_width": 4, "collect_stats_steps": 1500})

# Optional parameters for quantization
bias_quant_cfg = QuantConfig(base_classes= ["Int16Bias"])
from_float_weights = False # Do not reuse weights from the floating point model
calibration_setup = None # Do not calibrate (via: https://xilinx.github.io/brevitas/tutorials/tvmcon2021.html#Calibration-based-post-training-quantization) the model
skip_modules = [] # Quantize all modules

# Create a QAT-ready model
qat_ready_model = create_qat_ready_model(model,
                                         weight_quant_cfg,
                                         act_quant_cfg,
                                         bias_quant_cfg,
                                         from_float_weights,
                                         calibration_setup,
                                         skip_modules)

# Train as usual...

# Get quantized weights and biases
quant_state_dict = get_quant_state_dict(qat_ready_model, (10,))

# Can either save and load via torch
torch.save(quant_state_dict, "quant_model.pth")
quant_state_dict_loaded = torch.load("quant_model.pth")

# Or, can use built-in functions
from brevitas_utils import save_quant_state_dict, load_quant_state_dict 
save_quant_state_dict(quant_state_dict, "quant_model.pkl")
quant_state_dict_loaded = load_quant_state_dict("quant_model.pkl")
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
