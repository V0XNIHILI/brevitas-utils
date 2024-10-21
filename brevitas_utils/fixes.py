import torch

from brevitas.quant_tensor import IntQuantTensor


def __getitem__(self, indices):
    # Only allow indexing on QuantTensors with scalar scale
    if self.scale == None or self.scale.shape == torch.Size([]):
        return IntQuantTensor(self.value[indices], self.scale, self.zero_point, self.bit_width, self.signed, self.training)

    # Do not yet support indexing on QuantTensors with scale per channel, as it is not directly clear how to handle this
    raise RuntimeError("QuantTensor with scale of shape {} is not supported.".format(self.scale.shape))


def allow_quant_tensor_slicing():
    IntQuantTensor.__getitem__ = __getitem__
