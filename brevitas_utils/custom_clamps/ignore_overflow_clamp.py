import torch
from torch import Tensor

import brevitas


@brevitas.jit.script
def ignore_overflow_clamp(x: Tensor, min_val: Tensor, max_val: Tensor) -> Tensor:
    """
    Generalized overflow function that wraps values exceeding the specified range.

    Args:
        x: Input tensor on which to apply the overflow behavior.
        min_val: Minimum values for the range.
        max_val: Maximum values for the range.

    Notes:
        x, min_val, max_val need to be broadcastable.
        Differentiable w.r.t. x, min_val, max_val.

    Returns:
        Tensor where values are wrapped around when exceeding min_val or max_val.

    Examples:
        >>> tensor_overflow(torch.tensor([1.7, -0.5, 0.1]), torch.tensor(0.0), torch.tensor(1.0))
        tensor([0.7000, 0.5000, 0.1000])
    """
    range_val = max_val - min_val
    wrapped = (x - min_val) % range_val + min_val
    return wrapped


class IgnoreOverflowClamp(brevitas.jit.ScriptModule):
    """
    ScriptModule wrapper for :func:`~brevitas.function.ops.tensor_clamp`.

    Examples:
        >>> tensor_clamp = TensorClamp()
        >>> min_val = torch.tensor(-2.0)
        >>> max_val = torch.tensor(2.0)
        >>> tensor_clamp(torch.tensor([-3.0, 3.0]), min_val, max_val)
        tensor([-2.,  2.])
    """

    def __init__(self) -> None:
        super(IgnoreOverflowClamp, self).__init__()

    @brevitas.jit.script_method
    def forward(self, x: Tensor, min_val: Tensor, max_val: Tensor):
        return ignore_overflow_clamp(x, min_val=min_val, max_val=max_val)
