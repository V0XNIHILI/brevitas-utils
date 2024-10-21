import torch

import brevitas

if int(brevitas.__version__.split('.')[1]) == 11:
    from brevitas.quant_tensor import IntQuantTensor as QT
else:
    from brevitas.quant_tensor import QuantTensor as QT


def _get_item(self, indices):
    # Only allow indexing on QuantTensors with scalar scale
    if self.scale == None or self.scale.shape == torch.Size([]):
        return QT(self.value[indices], self.scale, self.zero_point, self.bit_width, self.signed, self.training)

    # Do not yet support indexing on QuantTensors with scale per channel, as it is not directly clear how to handle this
    raise RuntimeError("QuantTensor with scale of shape {} is not supported.".format(self.scale.shape))


def allow_quant_tensor_slicing():
    QT.__getitem__ = _get_item
