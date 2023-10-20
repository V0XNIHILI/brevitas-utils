from typing import Callable, Union

import torch

OptionalBatchTransform = Union[Callable[[torch.Tensor], torch.Tensor], None]
