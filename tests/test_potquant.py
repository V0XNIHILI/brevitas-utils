import torch

from brevitas_utils.custom_quantizers.Int8WeightPerTensorPowerOfTwo import potquant


def test_potquant_values():
    for bit_width in range(1, 16):
        inputs = 2**(-torch.tensor(range(0, 2**(bit_width-1)))).float()

        inputs = torch.cat([inputs, -inputs])

        outputs = potquant(inputs, bit_width)

        assert torch.allclose(outputs, inputs), f"Power-of-two valued number do not match for bit width {bit_width}"

        if bit_width < 9:
            assert len(torch.unique(outputs)) == 2**bit_width, f"Number of possible values do not match for bit width {bit_width}"


def test_potquant_clamp():
    inputs = torch.tensor([2.0, -2.0])

    outputs = potquant(inputs, 8)

    assert torch.max(torch.abs(outputs)) <= 1.0, f"Clamping does not work"
