from typing import Tuple, Union, Callable, Optional

import torch


OptionalBatchTransform = Optional[Callable[[torch.Tensor], torch.Tensor]]
TupleOfTensors = Tuple[torch.Tensor, ...]
NestedTupleOfTensors = Union['NestedTupleOfTensors', TupleOfTensors]
