
from typing import Optional

from torch import nn

from brevitas.inject.defaults import Int8ActPerTensorFloat
from brevitas.inject.defaults import Int8ActPerTensorFloatMinMaxInit
from brevitas.inject.defaults import Uint8ActPerTensorFloat


from brevitas.nn.quant_layer import ActQuantType
from brevitas.nn.quant_layer import QuantNonLinearActLayer as QuantNLAL



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