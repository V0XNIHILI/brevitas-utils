import torch
import torch.nn as nn

import brevitas
from brevitas.core.quant.delay import DelayWrapper
from brevitas.proxy import WeightQuantProxyFromInjector
from brevitas.quant.fixed_point import Int8WeightPerTensorFixedPoint


class ClampedQuantizePowerOfTwo(torch.autograd.Function):
    # Clamp all inputs clamped between -1 and 1 to the closest power of two

    @staticmethod
    def forward(_, input: torch.Tensor, bit_width: int):
        sign = input.sign()
        input_abs = input.abs()

        input_clamped = input_abs.clamp(min=-1, max=1)

        minimum_log2_value: int = -(2**(bit_width-1) - 1)
        maximum_log2_value = 0

        rounded_log2_values = torch.round(torch.log2(input_clamped))
        clamped_rounded_log2_values = torch.clamp(rounded_log2_values,
                                                  minimum_log2_value,
                                                  maximum_log2_value)

        # As sign is 0 or 1, we need to convert it to -1 or 1
        zero_corrected_sign = (sign+1.0).sign()*2-1.0
        
        return torch.exp2(clamped_rounded_log2_values) * zero_corrected_sign

    @staticmethod
    def backward(_, grad_output: torch.Tensor):
        return grad_output, None

potquant = ClampedQuantizePowerOfTwo.apply

class ClampedPoTQuantizer(brevitas.jit.ScriptModule):

    def __init__(self,
                 scaling_impl: nn.Module,
                 bit_width,
                 zero_point_impl,
                 signed,
                 quant_delay_steps: int = 0):
        super(ClampedPoTQuantizer, self).__init__()
        self.scaling_impl = scaling_impl
        self.bit_width = bit_width
        self.zero_point_impl = zero_point_impl
        self.delay_wrapper = DelayWrapper(quant_delay_steps)

        if not signed:
            raise NotImplementedError(
                "Unsigned quantization not implemented yet")
        self.signed = signed

    @brevitas.jit.script_method
    def forward(self, x: torch.Tensor):
        scale = self.scaling_impl(x)
        # TODO: check if zero-point is ZeroZeroPointImpl
        zero_point = self.zero_point_impl(x, scale, self.bit_width) # TODO: not sure if passing this bitwidth is correct

        y = potquant(x / scale, int(self.bit_width)) * scale
        y = self.delay_wrapper(x, y)

        return y, scale, zero_point, self.bit_width


class ClampedPoTWeightQuantizer(Int8WeightPerTensorFixedPoint):
    tensor_quant = ClampedPoTQuantizer
    bit_width = 8


class Int8WeightPerTensorPowerOfTwo(ClampedPoTWeightQuantizer):
    proxy_class = WeightQuantProxyFromInjector
