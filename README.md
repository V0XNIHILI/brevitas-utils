# Brevitas utils (`brevitas_utils`)

Library with utilities for [Brevitas](https://github.com/Xilinx/brevitas/).

## Features

- 1-line conversion of a floating point PyTorch model to a model with post-training quantization(PTQ) applied that can be used for quantization-aware training (QAT)
  - Automatic batch normalization folding
  - Automatic removal of dropout layers
- 1-line extraction of quantized weights and biases from a QAT model
- Power-of-two weight quantization ([`Int8WeightPerTensorPowerOfTwo.py`](brevitas_utils/custom_quantizers/Int8WeightPerTensorPowerOfTwo.py))
- Allowing `QuantTensor`s to be sliced

## Installation

```bash
git clone git@github.com:V0XNIHILI/brevitas-utils.git
cd brevitas-utils
pip install -e .
```

## Usage

Please see a small tutorial below.

### 1. Define model to be quantized

```python
import torch.nn as nn

model = nn.Sequential(
    nn.Linear(10, 20),
    nn.ReLU(),
    nn.Linear(20, 10)
)
```

### 2. Define quantization configurations

See [here](https://xilinx.github.io/brevitas/tutorials/tvmcon2021.html#Inheriting-from-a-quantizer) for more details.

```python
from brevitas_utils import create_quantizer

# Mandatory parameters for quantization

weight_quant = create_quantizer(base_classes=["Int8WeightPerTensorPowerOfTwo"], kwargs={"bit_width": 4, "narrow_range": False})
act_quant = create_quantizer(base_classes=["ShiftedParamFromPercentileUintQuant"], kwargs={"bit_width": 4, "collect_stats_steps": 1500})

# Optional parameters for quantization

bias_quant = create_quantizer(base_classes= ["Int16Bias"])
# Do not reuse weights from the floating point model
from_float_weights = False
# Do not calibrate the model, calibration is necessary for PTQ (via: https://xilinx.github.io/brevitas/tutorials/tvmcon2021.html#Calibration-based-post-training-quantization)
calibration_setup = None
# Quantize all modules; dont skip any
skip_modules = []
```

### 3. Create a QAT-ready model & apply PTQ

Note: `create_qat_ready_model` applies PTQ to the model if `calibration_setup` is not `None`.

```python
from brevitas_utils import create_qat_ready_model

qat_ready_model = create_qat_ready_model(
  model,
  weight_quant,
  act_quant,
  bias_quant,
  load_float_weights_into_model=from_float_weights,
  calibration_setup=calibration_setup,
  skip_modules=skip_modules
)
```

### 4. Training (or not)

#### 4.1 Use PTQ model without further QAT

```python
# You do not have to do anything else in this case:
# as long as you specify the calibration_setup, PTQ
# will be performed automatically in the
# create_qat_ready_model function.
```

#### 4.2. Continue training/finetuning the model via QAT

Note: this can also be done after performing a PTQ step.

```python
for epoch in range(10):
    ...
```

### 5. Export quantized weights and biases

```python
from brevitas_utils import get_quant_state_dict

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

### 6. Extras

Allow for slicing of `QuantTensor`s, which is currently not possible since their base class is a `NamedTuple`:

```python
from brevitas_utils import allow_slicing

allow_slicing()
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
