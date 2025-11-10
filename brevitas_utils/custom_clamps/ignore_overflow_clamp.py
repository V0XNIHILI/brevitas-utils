import torch
from torch import Tensor

import brevitas


class ModWithClampGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: Tensor, min_val: Tensor, max_val: Tensor, gradient_outside_range: float = 1.0):
        range_val = max_val - min_val + 1
        wrapped = (x - min_val) % range_val + min_val

        ctx.save_for_backward(min_val, max_val, x)
        ctx.gradient_outside_range = gradient_outside_range

        return wrapped

    @staticmethod
    def backward(ctx, grad_output):
        min_val, max_val, x = ctx.saved_tensors
        gradient_outside_range = ctx.gradient_outside_range

        if gradient_outside_range == 1.0:
            return grad_output, None, None, None
        
        grad_input = grad_output.clone()

        if gradient_outside_range == 0.0:
            grad_input[x < min_val] = 0
            grad_input[x > max_val] = 0

            return grad_input, None, None, None
        
        grad_input[x < min_val] = gradient_outside_range * grad_output[x < min_val]
        grad_input[x > max_val] = -gradient_outside_range * grad_output[x > max_val]

        return grad_input, None, None, None


mod_with_clamp_grad_func = ModWithClampGrad.apply


@brevitas.jit.script
def ignore_overflow_clamp(x: Tensor, min_val: Tensor, max_val: Tensor, gradient_outside_range: float = 1.0) -> Tensor:
    """
    Generalized overflow function that wraps values exceeding the specified range.

    Args:
        x: Input tensor on which to apply the overflow behavior.
        min_val: Minimum values for the range.
        max_val: Maximum values for the range.
        zero_gradients_outside_range: If set to 0, gradients are zeroed outside the range. This can help with convergence, especially for really deep networks.
            If set to 1, gradients are passed through unchanged. If set to -1.0, the gradients look like this: [-1 (before min), 1 (inside range), 2 (after max)].

    Notes:
        x, min_val, max_val need to be broadcastable.
        Differentiable w.r.t. x, min_val, max_val.

    Returns:
        Tensor where values are wrapped around when exceeding min_val or max_val.

    Examples:
        >>> tensor_overflow(torch.tensor([1.7, -0.5, 0.1]), torch.tensor(0.0), torch.tensor(1.0))
        tensor([0.7000, 0.5000, 0.1000])
    """

    return mod_with_clamp_grad_func(x, min_val, max_val, gradient_outside_range)


class IgnoreOverflowClamp(brevitas.jit.ScriptModule):
    """
    ScriptModule wrapper for :func:`~brevitas_utils.custom_clamps.ignore_overflow_clamp.ignore_overflow_clamp`.

    Examples:
        >>> ignore_overflow_clamp = IgnoreOverflowClamp()
        >>> min_val = torch.tensor(0.0)
        >>> max_val = torch.tensor(15.0)
        >>> ignore_overflow_clamp(torch.tensor([1.0, 16.0]), min_val, max_val)
        tensor([1.,  0.])
    """

    def __init__(self, gradient_outside_range: float = 1.0):
        super(IgnoreOverflowClamp, self).__init__()

        assert gradient_outside_range in (0.0, 1.0) or gradient_outside_range < 0.0, "gradient_outside_range can only be 0.0, 1.0 or smaller than 0."

        self.gradient_outside_range = gradient_outside_range

    @brevitas.jit.script_method
    def forward(self, x: Tensor, min_val: Tensor, max_val: Tensor):
        return ignore_overflow_clamp(x, min_val=min_val, max_val=max_val, gradient_outside_range=self.gradient_outside_range)
