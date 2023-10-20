# Brevitas utils (`brevitas_utils`)

Library with utilities for [Brevitas](https://github.com/Xilinx/brevitas/).

Features:

- 1-line conversion of a floating point PyTorch model to a model that can be used for quantization-aware training (QAT)

## Installation

```bash
git clone git@github.com:V0XNIHILI/brevitas-utils.git
cd brevitas-utils
pip install -e .
```

## Usage

```python
import torch.nn as nn

from brevitas_utils import create_qat_ready_model

model = nn.Sequential(
    nn.Linear(10, 20),
    nn.ReLU(),
    nn.Linear(20, 10)
)

qat_ready_model = create_qat_ready_model(model)

# Train as usual...
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
