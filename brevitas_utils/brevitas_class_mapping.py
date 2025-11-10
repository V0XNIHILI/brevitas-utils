import brevitas.quant
from brevitas.quant.base import __all__ as base_quant_all
from brevitas.quant.binary import __all__ as binary_quant_all
from brevitas.quant.fixed_point import __all__ as fixed_point_quant_all
from brevitas.quant.none import __all__ as none_quant_all
from brevitas.quant.scaled_int import __all__ as scaled_int_quant_all
from brevitas.quant.shifted_scaled_int import \
    __all__ as shifted_scaled_int_quant_all
from brevitas.quant.ternary import __all__ as ternary_quant_all

from .custom_quantizers.pot4_weight_per_tensor_fixed_point import \
    PoT4WeightPerTensorFixedPoint

# Create a dict with key = module name and value = list of classes in the module
quant_classes = {
    'shifted_scaled_int': shifted_scaled_int_quant_all,
    'scaled_int': scaled_int_quant_all,
    'fixed_point': fixed_point_quant_all,
    'binary': binary_quant_all,
    'ternary': ternary_quant_all,
    'none': none_quant_all,
    'base': base_quant_all,
    'scaled_int': scaled_int_quant_all
}

custom_quant_classes = {
    'PoT4WeightPerTensorFixedPoint': PoT4WeightPerTensorFixedPoint
}


def get_brevitas_class_by_name(class_name: str):
    for module_name, classes in quant_classes.items():
        if class_name in classes:
            return getattr(getattr(brevitas.quant, module_name), class_name)

    if class_name in custom_quant_classes:
        return custom_quant_classes[class_name]

    raise ValueError(
        f"Class {class_name} not found in any of the modules (brevitas.quant.{list(quant_classes.keys())}) or in custom classes ({list(custom_quant_classes.keys())})"
    )
