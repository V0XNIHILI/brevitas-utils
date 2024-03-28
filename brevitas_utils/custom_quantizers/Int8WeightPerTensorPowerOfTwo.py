"""Complete approach based on: https://github.com/Xilinx/brevitas/blob/master/notebooks/03_anatomy_of_a_quantizer.ipynb
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import brevitas

from brevitas.core.quant.delay import DelayWrapper
from brevitas.proxy import WeightQuantProxyFromInjector
from brevitas.quant.fixed_point import Int8WeightPerTensorFixedPoint
from brevitas.core.zero_point import ZeroZeroPoint


@torch.jit.script
def clamped_quantize_power_of_two(input: torch.Tensor, bit_width: int):
    sign = input.sign()
    input_abs = input.abs()

    max = 2**(2**(bit_width-1)-1)
    min = -max

    input_clamped = input_abs.clamp(min=min, max=max)

    rounded_log2_values = torch.round(torch.log2(input_clamped))
    clamped_rounded_log2_values = F.relu(rounded_log2_values)

    # As sign is 0 or 1, we need to convert it to -1 or 1
    zero_corrected_sign = (sign + 1.0).sign() * 2.0 - 1.0
    
    return torch.exp2(clamped_rounded_log2_values) * zero_corrected_sign


class ClampedQuantizePowerOfTwo(torch.autograd.Function):
    """Clamp all inputs between -1 and 1 and to the closest power of two
    """

    @staticmethod
    def forward(_, input: torch.Tensor, bit_width: int):
        return clamped_quantize_power_of_two(input, bit_width)

    @staticmethod
    def backward(_, grad_output: torch.Tensor):
        return grad_output, None


potquant = ClampedQuantizePowerOfTwo.apply


class ClampedPoTQuantizer(brevitas.jit.ScriptModule):

    def __init__(self,
                 scaling_impl: nn.Module,
                 int_scaling_impl: nn.Module,
                 zero_point_impl: nn.Module,
                 bit_width: int,
                 signed: bool,
                 narrow_range: bool = False,
                 quant_delay_steps: int = 0):
        super(ClampedPoTQuantizer, self).__init__()

        self.scaling_impl = scaling_impl
        self.int_scaling_impl = int_scaling_impl
        self.zero_point_impl = zero_point_impl

        # TODO: Also use a bit_width_impl function instead of computing it in here?

        self.bit_width = bit_width
        # Define the bit width for Brevitas that is used for scaling differently,
        # as for example with 4 bits of signed logarithmic weights, you can represent
        # values of signed 8 bit regular integer.
        self.brevitas_bit_width = 2**(bit_width-1)

        self.signed = signed
        self.narrow_range = narrow_range

        self.delay_wrapper = DelayWrapper(quant_delay_steps)

        if not isinstance(self.zero_point_impl, ZeroZeroPoint):
            raise NotImplementedError("Zero-point must be ZeroZeroPointImpl for ClampedPoTQuantizer")
        elif not self.signed:
            raise NotImplementedError(
                "Unsigned quantization not implemented yet for ClampedPoTQuantizer")
        elif self.narrow_range:
            raise ValueError("Power-of-two quantization only works correctly when narrow_range is False")

    @brevitas.jit.script_method
    def forward(self, x: torch.Tensor):
        scale = self.scaling_impl(x) / self.int_scaling_impl(self.brevitas_bit_width)

        zero_point = self.zero_point_impl(x, scale, self.brevitas_bit_width)

        y = potquant(x / scale, int(self.bit_width)) * scale
        y = self.delay_wrapper(x, y)

        return y, scale, zero_point, self.brevitas_bit_width


class ClampedPoTWeightQuantizer(Int8WeightPerTensorFixedPoint):
    tensor_quant = ClampedPoTQuantizer
    bit_width = 8


class Int8WeightPerTensorPowerOfTwo(ClampedPoTWeightQuantizer):
    proxy_class = WeightQuantProxyFromInjector
