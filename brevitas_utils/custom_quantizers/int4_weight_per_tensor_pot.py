"""Complete approach based on: https://github.com/Xilinx/brevitas/blob/master/notebooks/03_anatomy_of_a_quantizer.ipynb
"""
from typing import Tuple
import platform

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

import brevitas

from brevitas.core.quant.delay import DelayWrapper
from brevitas.proxy import WeightQuantProxyFromInjector
from brevitas.quant.fixed_point import Int8WeightPerTensorFixedPoint
from brevitas.core.zero_point import ZeroZeroPoint


class FakeIntQuant:
    def __init__(self):
        self.input_view_impl = nn.Identity()


def clamped_quantize_power_of_two(input: torch.Tensor, bit_width: int, floor: bool = False, allow_less_than_one: bool = False):
    sign = input.sign()
    input_abs = input.abs()

    max = 2**(2**(bit_width-1)-1)
    min = -max

    input_clamped = input_abs.clamp(min=min, max=max)

    log2_values = torch.log2(input_clamped)
    rounded_floored_log2_values = torch.floor(log2_values) if floor else torch.round(log2_values)
    clamped_rounded_log2_values = rounded_floored_log2_values if allow_less_than_one else F.relu(rounded_floored_log2_values)

    # As sign is 0 or 1, we need to convert it to -1 or 1
    zero_corrected_sign = (sign + 1.0).sign() * 2.0 - 1.0
    
    return torch.exp2(clamped_rounded_log2_values) * zero_corrected_sign

# Optionally apply the decorator, do not apply it on the aarch64 architecture because it
# leads to "Illegal instruction (core dumped)"
if platform.processor() != 'aarch64':
    clamped_quantize_power_of_two = torch.jit.script(clamped_quantize_power_of_two)


class ClampedQuantizePowerOfTwo(torch.autograd.Function):
    """Clamp all inputs between -2**(2**(bit_width-1)-1)) and 2**(2**(bit_width-1)-1))
    and to the closest power of two"""

    @staticmethod
    def forward(_, input: torch.Tensor, bit_width: int, floor: bool = False, allow_less_than_one: bool = False):
        return clamped_quantize_power_of_two(input, bit_width, floor, allow_less_than_one)

    @staticmethod
    def backward(_, grad_output: torch.Tensor):
        return grad_output, None, None, None


potquant = ClampedQuantizePowerOfTwo.apply


# Create small helper class for 32-bit quantization for quantization-aware training with ProtoNets
class PotQuant32BFloorFloat(nn.Module):
    def __init__(self, is_input_quant_tensor: bool = False):
        super(PotQuant32BFloorFloat, self).__init__()
        self.is_input_quant_tensor = is_input_quant_tensor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return potquant(x.value if self.is_input_quant_tensor else x, 32, True, True) # type: ignore[arg-type]


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
        self.int_quant = FakeIntQuant()

        # TODO: Also use a bit_width_impl function instead of computing it in here?

        self.bit_width = bit_width
        # Define the bit width for Brevitas that is used for scaling differently,
        # as for example with 4 bits of signed logarithmic weights, you can represent
        # values of signed 8 bit regular integer.
        self.brevitas_bit_width = 2**(bit_width-1)

        self.signed = signed
        self.narrow_range = narrow_range

        self.delay_wrapper = DelayWrapper(quant_delay_steps)

        self.observer_only = brevitas.jit.Attribute(False, bool)

        if not isinstance(self.zero_point_impl, ZeroZeroPoint):
            raise NotImplementedError("Zero-point must be ZeroZeroPointImpl for ClampedPoTQuantizer")
        elif not self.signed:
            raise NotImplementedError("Unsigned quantization not implemented yet for ClampedPoTQuantizer")
        elif self.narrow_range:
            raise ValueError("Power-of-two quantization only works correctly when narrow_range is False")

    @brevitas.jit.script_method
    def forward(self, x: torch.Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        scale = self.scaling_impl(x) / self.int_scaling_impl(self.brevitas_bit_width)
        zero_point = self.zero_point_impl(x, scale, self.brevitas_bit_width)

        if self.observer_only:
            y = x
        else:
            y_int = potquant(x / scale + zero_point, int(self.bit_width))
            y = (y_int - zero_point) * scale
            y = self.delay_wrapper(x, y)

        return y, scale, zero_point, self.brevitas_bit_width


class Int4WeightPerTensorPowerOfTwo(Int8WeightPerTensorFixedPoint):
    bit_width = 4
    narrow_range = False
    tensor_quant = ClampedPoTQuantizer
    proxy_class = WeightQuantProxyFromInjector
