
from typing import Optional

import torch
from torch import nn
import torch.nn.functional as F

from brevitas.inject.defaults import Int8ActPerTensorFloatMinMaxInit
from brevitas.inject.defaults import Uint8ActPerTensorFloat

from brevitas.nn.quant_layer import ActQuantType
from brevitas.nn.quant_layer import QuantNonLinearActLayer as QuantNLAL


class BSiLU(nn.Module):
    def __init__(self, alpha: float = 1.67):
        # per: https://arxiv.org/pdf/2505.22074
        super().__init__()
        self.alpha = alpha

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # σ(x) = sigmoid(x)
        return (x + self.alpha) * torch.sigmoid(x) - self.alpha / 2


class QuantBSiLU(QuantNLAL):

    def __init__(
            self,
            act_quant: Optional[ActQuantType] = Uint8ActPerTensorFloat,
            input_quant: Optional[ActQuantType] = None,
            return_quant_tensor: bool = False,
            **kwargs):
        QuantNLAL.__init__(
            self,
            act_impl=BSiLU,
            passthrough_act=True,
            input_quant=input_quant,
            act_quant=act_quant,
            return_quant_tensor=return_quant_tensor,
            **kwargs)


class QuantHardsigmoid(QuantNLAL):
    def __init__(
            self,
            act_quant: Optional[ActQuantType] = Int8ActPerTensorFloatMinMaxInit,
            input_quant: Optional[ActQuantType] = None,
            return_quant_tensor: bool = False,
            **kwargs):
        QuantNLAL.__init__(
            self,
            act_impl=nn.Hardsigmoid,
            passthrough_act=True,
            input_quant=input_quant,
            act_quant=act_quant,
            return_quant_tensor=return_quant_tensor,
            **kwargs)


class QuantHardswish(QuantNLAL):
    def __init__(
            self,
            act_quant: Optional[ActQuantType] = Uint8ActPerTensorFloat,
            input_quant: Optional[ActQuantType] = None,
            return_quant_tensor: bool = False,
            **kwargs):
        QuantNLAL.__init__(
            self,
            act_impl=nn.Hardswish,
            passthrough_act=True,
            input_quant=input_quant,
            act_quant=act_quant,
            return_quant_tensor=return_quant_tensor,
            **kwargs)
