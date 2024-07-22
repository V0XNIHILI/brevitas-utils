from typing import Tuple, Union, Optional, Callable

import torch

OptionalBatchTransform = Union[Callable[[torch.Tensor], torch.Tensor], None]
TupleOfTensors = Tuple[torch.Tensor, ...]
NestedTupleOfTensors = Union['NestedTupleOfTensors', TupleOfTensors]
